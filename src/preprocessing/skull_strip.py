import os
import SimpleITK as sitk
import numpy as np

def simple_skull_strip(image):
    """
    A robust skull-stripping method using Otsu's thresholding and morphological operations.
    Note: For high-accuracy research, HD-BET or a deep-learning model is preferred.
    """
    # 1. Cast image to Float32 for processing
    image = sitk.Cast(image, sitk.sitkFloat32)
    
    # 2. Rescale intensity to [0, 1]
    rescale_filter = sitk.RescaleIntensityImageFilter()
    rescale_filter.SetOutputMaximum(1.0)
    rescale_filter.SetOutputMinimum(0.0)
    rescaled_image = rescale_filter.Execute(image)
    
    # 3. Apply Otsu Thresholding to find a rough brain/skull mask
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetOutsideValue(1)
    binary_mask = otsu_filter.Execute(rescaled_image)
    
    # 4. Fill holes and remove small components (morphological cleanup)
    # Using BinaryMorphologicalOpening to remove small background noise
    opening_filter = sitk.BinaryMorphologicalOpeningImageFilter()
    opening_filter.SetKernelRadius(2)
    binary_mask = opening_filter.Execute(binary_mask)
    
    # Fill holes
    hole_filling_filter = sitk.BinaryFillholeImageFilter()
    binary_mask = hole_filling_filter.Execute(binary_mask)
    
    # 5. Connected Component Analysis to keep only the largest component (the brain)
    cc_filter = sitk.ConnectedComponentImageFilter()
    label_image = cc_filter.Execute(binary_mask)
    
    label_stats = sitk.LabelShapeStatisticsImageFilter()
    label_stats.Execute(label_image)
    
    largest_label = 0
    max_pixels = 0
    for label in label_stats.GetLabels():
        pixels = label_stats.GetNumberOfPixels(label)
        if pixels > max_pixels:
            max_pixels = pixels
            largest_label = label
            
    if largest_label == 0:
        print("Warning: No brain tissue found in simple skull stripping.")
        return binary_mask
        
    brain_mask = sitk.Equal(label_image, largest_label)
    
    # 6. Apply smoothing to the mask for a better result
    smoothing_filter = sitk.MedianImageFilter()
    smoothing_filter.SetRadius(3)
    brain_mask = smoothing_filter.Execute(brain_mask)
    
    return brain_mask

if __name__ == "__main__":
    pass
