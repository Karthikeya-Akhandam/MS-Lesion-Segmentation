import os
import torch
import SimpleITK as sitk
import numpy as np
from .preprocessing.preprocessor import MSPreprocessor
from .model import get_model
from .longitudinal import track_longitudinal_change
from .visual_qc import generate_qc_montage
from monai.inferers import sliding_window_inference

class ClinicalPipeline:
    def __init__(self, model_path, output_dir, device="cuda"):
        self.output_dir = output_dir
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained model (from Phase 2)
        self.model = get_model(in_channels=1, out_channels=1).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Preprocessor (from Phase 1)
        self.preprocessor = MSPreprocessor(output_dir=os.path.join(output_dir, "processed"))

    def process_patient(self, flair_path, subject_id="patient_001", t0_mask_path=None):
        """
        Runs the end-to-end pipeline: Preprocess -> Inference -> Report
        """
        print(f"
--- Processing {subject_id} ---")
        
        # 1. Preprocessing (Phase 1)
        # Returns sitk.Image
        flair_processed = self.preprocessor.run_pipeline(flair_path, subject_id=subject_id)['flair']
        
        # 2. Inference (Phase 2 & 5)
        # Convert to tensor for MONAI
        flair_array = sitk.GetArrayFromImage(flair_processed)
        flair_tensor = torch.tensor(flair_array).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        print("Inference (Sliding Window)...")
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                # ROI size from Phase 2
                prob_map = sliding_window_inference(flair_tensor, (96, 96, 96), 4, self.model)
                prob_map = prob_map.sigmoid().cpu().numpy().squeeze()
        
        # Apply optimal threshold (e.g., 0.5)
        pred_mask = (prob_map >= 0.5).astype(np.uint8)
        
        # 3. Save Mask & Generate QC (Phase 5)
        subject_dir = os.path.join(self.output_dir, subject_id)
        os.makedirs(subject_dir, exist_ok=True)
        
        mask_out = os.path.join(subject_dir, f"{subject_id}_pred_mask.nii.gz")
        pred_img = sitk.GetImageFromArray(pred_mask)
        pred_img.CopyInformation(flair_processed)
        sitk.WriteImage(pred_img, mask_out)
        
        # Visual QC Montage
        qc_out = os.path.join(subject_dir, f"{subject_id}_qc_montage.png")
        generate_qc_montage(mask_out, qc_out, title=f"Segmentation Overlay: {subject_id}")
        
        # 4. Longitudinal Analysis (if baseline exists)
        report = {
            'subject_id': subject_id,
            'total_lesion_volume': np.sum(pred_mask) * 1.0, # 1mm3 per voxel
            'lesion_count': 0 # From metrics
        }
        
        if t0_mask_path:
            print("Running Longitudinal Comparison...")
            t0_mask = sitk.GetArrayFromImage(sitk.ReadImage(t0_mask_path))
            long_report = track_longitudinal_change(t0_mask, pred_mask)
            report.update(long_report)
            
        return report

if __name__ == "__main__":
    pass
