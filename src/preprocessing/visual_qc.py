import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

def generate_qc_montage(nifti_path, output_path, title="QC Montage"):
    """
    Generates a 2D montage of Axial, Sagittal, and Coronal slices from a 3D NIfTI volume.
    """
    try:
        img = nib.load(nifti_path)
        data = img.get_fdata()
        
        # Get center indices
        shape = data.shape
        mid_x, mid_y, mid_z = shape[0] // 2, shape[1] // 2, shape[2] // 2
        
        # Extract slices
        slices = [
            data[mid_x, :, :].T,   # Sagittal
            data[:, mid_y, :].T,   # Coronal
            data[:, :, mid_z].T    # Axial
        ]
        
        titles = ['Sagittal', 'Coronal', 'Axial']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"{title}
{os.path.basename(nifti_path)}", fontsize=12)
        
        for i, (slice_data, t) in enumerate(zip(slices, titles)):
            axes[i].imshow(slice_data, cmap='gray', origin='lower')
            axes[i].set_title(t)
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"QC Montage saved to: {output_path}")
        
    except Exception as e:
        print(f"Failed to generate QC montage for {nifti_path}: {e}")

if __name__ == "__main__":
    # Example usage logic
    pass
