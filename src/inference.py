import os
import torch
import SimpleITK as sitk
from monai.inferers import sliding_window_inference
from .model import get_model
from .metrics import get_lesion_wise_metrics

def run_zero_shot_eval(model_path, test_files, in_channels=1, device="cuda"):
    """
    Evaluates a frozen model on unseen clinical datasets.
    test_files: list of dicts with 'image' (multi-channel) and 'label' paths.
    """
    model = get_model(in_channels=in_channels, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for f in test_files:
            # Load images as multi-channel input
            # Assume images are already preprocessed (Resampled, Normalized)
            image_paths = f['image'] # List of paths
            images = [sitk.GetArrayFromImage(sitk.ReadImage(p)) for p in image_paths]
            image_tensor = torch.tensor(np.stack(images)).float().unsqueeze(0).to(device)
            
            mask_path = f['label']
            mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
            
            # Sliding window inference
            roi_size = (96, 96, 96)
            pred = sliding_window_inference(image_tensor, roi_size, 4, model)
            pred = (pred.sigmoid() > 0.5).cpu().numpy().squeeze()
            
            # Compute Metrics
            metrics = get_lesion_wise_metrics(pred, mask)
            results.append(metrics)
            
    return results

if __name__ == "__main__":
    pass
