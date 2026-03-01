import os
import SimpleITK as sitk
import nibabel as nib
import numpy as np
from tqdm import tqdm

class MSPreprocessor:
    def __init__(self, output_dir, target_spacing=(1.0, 1.0, 1.0)):
        self.output_dir = output_dir
        self.target_spacing = target_spacing
        os.makedirs(output_dir, exist_ok=True)

    def reorient_to_ras(self, image):
        """Reorients image to RAS+ (standard orientation)."""
        return sitk.DICOMOrient(image, "RAS")

    def resample_image(self, image, is_mask=False):
        """Resamples image to target spacing using B-spline (images) or Nearest Neighbor (masks)."""
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        
        # Calculate new size
        new_size = [
            int(round(osz * ospc / tspc))
            for osz, ospc, tspc in zip(original_size, original_spacing, self.target_spacing)
        ]
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(self.target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        
        if is_mask:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resampler.SetInterpolator(sitk.sitkBSpline)
            
        return resampler.Execute(image)

    def n4_bias_correction(self, image, mask=None):
        """Applies N4 Bias Field Correction, optionally within a brain mask."""
        if mask:
            # Mask must be same size and spacing
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            # Ensure mask is 0 or 1
            mask = sitk.Cast(mask, sitk.sitkUInt8)
            return corrector.Execute(image, mask)
        else:
            return sitk.N4BiasFieldCorrection(image)

    def register_images(self, fixed_image, moving_image, is_mask=False):
        """
        Registers moving_image to fixed_image using Rigid 6-DOF and Mutual Information.
        """
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, 
            moving_image, 
            sitk.Euler3DTransform(), 
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )

        registration_method = sitk.ImageRegistrationMethod()

        # Similarity metric settings
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.01)

        registration_method.SetInterpolator(sitk.sitkLinear)

        # Optimizer settings
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=1.0, 
            numberOfIterations=100, 
            convergenceMinimumValue=1e-6, 
            convergenceWindowSize=10
        )
        registration_method.SetOptimizerScalesFromPhysicalShift()

        # Setup for multi-resolution optimization
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        registration_method.SetInitialTransform(initial_transform, inPlace=False)

        final_transform = registration_method.Execute(fixed_image, moving_image)

        # Apply final transform
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_image)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkBSpline)
        resampler.SetTransform(final_transform)
        
        return resampler.Execute(moving_image)

    def z_score_normalize(self, image):
        """Applies Z-score normalization."""
        array = sitk.GetArrayFromImage(image)
        mean = np.mean(array)
        std = np.std(array)
        normalized_array = (array - mean) / (std + 1e-8)
        
        normalized_image = sitk.GetImageFromArray(normalized_array)
        normalized_image.CopyInformation(image)
        return normalized_image

    def run_pipeline(self, flair_path, t1_path=None, t2_path=None, mask_path=None, subject_id="sub-001"):
        """
        Executes the full pipeline for a single subject.
        """
        from .skull_strip import simple_skull_strip
        
        print(f"Processing subject: {subject_id}...")
        
        # 1. Load FLAIR as the fixed reference
        flair = sitk.ReadImage(flair_path)
        flair = self.reorient_to_ras(flair)
        flair = self.resample_image(flair)
        
        # 2. Skull Stripping
        print("Skull stripping...")
        flair_mask = simple_skull_strip(flair)
        
        # 3. Apply N4 Bias Correction inside the brain mask
        print("N4 Bias Correction...")
        flair = self.n4_bias_correction(flair, mask=flair_mask)
        
        # 4. Normalize
        print("Intensity normalization...")
        flair_final = self.z_score_normalize(flair)
        
        # 5. Handle T1, T2 if provided
        processed_images = {'flair': flair_final}
        
        if t1_path:
            print("Processing T1...")
            t1 = sitk.ReadImage(t1_path)
            t1 = self.reorient_to_ras(t1)
            t1 = self.resample_image(t1)
            t1 = self.register_images(flair_final, t1)
            t1 = self.n4_bias_correction(t1, mask=flair_mask)
            t1 = self.z_score_normalize(t1)
            processed_images['t1'] = t1
            
        if t2_path:
            print("Processing T2...")
            t2 = sitk.ReadImage(t2_path)
            t2 = self.reorient_to_ras(t2)
            t2 = self.resample_image(t2)
            t2 = self.register_images(flair_final, t2)
            t2 = self.n4_bias_correction(t2, mask=flair_mask)
            t2 = self.z_score_normalize(t2)
            processed_images['t2'] = t2

        if mask_path:
            print("Processing Lesion Mask...")
            mask = sitk.ReadImage(mask_path)
            mask = self.reorient_to_ras(mask)
            mask = self.resample_image(mask, is_mask=True)
            # Re-register mask if it was not in the same space as FLAIR
            mask = self.register_images(flair_final, mask, is_mask=True)
            processed_images['mask'] = mask

        # 6. Save results to subject-specific output directory
        subject_output_dir = os.path.join(self.output_dir, subject_id)
        os.makedirs(subject_output_dir, exist_ok=True)
        
        for name, img in processed_images.items():
            out_file = os.path.join(subject_output_dir, f"{subject_id}_{name}.nii.gz")
            sitk.WriteImage(img, out_file)
            print(f"Saved: {out_file}")
        
        return processed_images


if __name__ == "__main__":
    pass
