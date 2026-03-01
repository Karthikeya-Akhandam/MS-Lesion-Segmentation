from monai.networks.nets import UNet
import torch.nn as nn

def get_model(in_channels=1, out_channels=1):
    """
    Returns a 3D Residual U-Net based on Phase 2 specifications.
    """
    model = UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="INSTANCE", # InstanceNorm3D for small batch stability
        act="LEAKYRELU",
        dropout=0.1,
        bias=True
    )
    return model

if __name__ == "__main__":
    # Test model instantiation
    model = get_model()
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters.")
