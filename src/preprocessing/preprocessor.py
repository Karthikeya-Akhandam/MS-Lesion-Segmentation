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

    def run_pipeline(self, flair_path, t1_path=None, t2_path=None, mask_path=None):
        """
        Executes the full pipeline for a single subject.
        """
        # Load FLAIR as the fixed reference
        flair = sitk.ReadImage(flair_path)
        flair = self.reorient_to_ras(flair)
        flair = self.resample_image(flair)
        
        # Placeholder for Skull Stripping (Integration required with a chosen tool)
        # Assuming flair_mask is generated here
        # flair_mask = self.skull_strip(flair) 
        
        # Apply N4 Bias Correction
        # flair = self.n4_bias_correction(flair, mask=flair_mask)
        
        # Normalize
        flair_final = self.z_score_normalize(flair)
        
        # Save results
        # sitk.WriteImage(flair_final, os.path.join(self.output_dir, "flair_processed.nii.gz"))
        
        return flair_final

if __name__ == "__main__":
    pass
