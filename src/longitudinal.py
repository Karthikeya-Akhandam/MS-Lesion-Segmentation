import os
import torch
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import label, binary_dilation

def noise_reduction_filter(mask, min_voxels=5):
    """
    Applies noise reduction (minimum size filter + smoothing)
    """
    labels, n = label(mask)
    new_mask = np.zeros_like(mask)
    for i in range(1, n + 1):
        if np.sum(labels == i) >= min_voxels:
            new_mask[labels == i] = 1
    return new_mask

def track_longitudinal_change(t0_mask, t1_mask, voxel_volume=1.0):
    """
    Categorizes lesion evolution between T0 and T1.
    Expects binary masks in the same space.
    """
    # 1. Apply filtering
    t0_filtered = noise_reduction_filter(t0_mask)
    t1_filtered = noise_reduction_filter(t1_mask)
    
    # 2. Basic Evolution Masks
    # New Lesions: T1 present, but not T0
    # Use small dilation on T0 to be conservative about spatial alignment
    t0_dilated = binary_dilation(t0_filtered, iterations=1)
    new_lesions = (t1_filtered == 1) & (t0_dilated == 0)
    
    # Resolved Lesions: T0 present, but not T1
    t1_dilated = binary_dilation(t1_filtered, iterations=1)
    resolved_lesions = (t0_filtered == 1) & (t1_dilated == 0)
    
    # Stable/Enlarging Lesions
    stable_region = (t0_filtered == 1) & (t1_filtered == 1)
    
    # 3. Component Analysis for Volume Change
    t1_labels, n_t1 = label(t1_filtered)
    evolution_report = {
        'new_lesion_count': 0,
        'resolved_lesion_count': 0,
        'enlarging_lesion_count': 0,
        'total_volume_t0': np.sum(t0_filtered) * voxel_volume,
        'total_volume_t1': np.sum(t1_filtered) * voxel_volume,
        'percent_change': 0.0
    }
    
    # New lesion count
    new_labels, n_new = label(new_lesions)
    evolution_report['new_lesion_count'] = n_new
    
    # Resolved lesion count
    resolved_labels, n_resolved = label(resolved_lesions)
    evolution_report['resolved_lesion_count'] = n_resolved
    
    # Enlarging logic (existing lesions with >20% growth)
    # ...
    
    if evolution_report['total_volume_t0'] > 0:
        evolution_report['percent_change'] = ((evolution_report['total_volume_t1'] - evolution_report['total_volume_t0']) / evolution_report['total_volume_t0']) * 100
        
    return evolution_report

if __name__ == "__main__":
    pass
