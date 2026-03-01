import os
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd, 
    RandCropByPosNegLabeld, RandFlipd, RandRotate90d, ToTensord
)
from monai.data import Dataset, DataLoader

def get_ms_transforms(patch_size=(96, 96, 96), num_samples=4):
    """
    Returns the transformation pipeline for MONAI, including balanced patch sampling.
    """
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        # We assume image is already Z-score normalized by the preprocessor
        # but EnsureChannelFirstd is needed for MONAI's internal structure.
        
        # 50% Pos (Lesion) / 50% Neg (Background) Patch Sampling
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=1,
            neg=1,
            num_samples=num_samples,
            image_key="image",
            image_threshold=0,
        ),
        
        # Data Augmentation (Phase 2 & 3 support)
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        
        ToTensord(keys=["image", "label"]),
    ])

def get_dataloader(data_list, batch_size=1, patch_size=(96, 96, 96), num_samples=4):
    """
    Creates a MONAI DataLoader from a list of dictionaries.
    data_list example: [{"image": "path/to/flair.nii.gz", "label": "path/to/mask.nii.gz"}]
    """
    transforms = get_ms_transforms(patch_size=patch_size, num_samples=num_samples)
    ds = Dataset(data=data_list, transform=transforms)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    return loader

if __name__ == "__main__":
    # Test loader instantiation logic
    pass
