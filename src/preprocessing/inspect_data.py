import nibabel as nib
import os
from glob import glob
import pandas as pd
from tqdm import tqdm

def inspect_voxels(data_dir):
    """
    Inspects all NIfTI files in a directory and reports their voxel dimensions.
    """
    stats = []
    # Search for all .nii and .nii.gz files recursively
    files = glob(os.path.join(data_dir, "**/*.nii*", recursive=True))
    
    if not files:
        print(f"No NIfTI files found in {data_dir}")
        return None

    print(f"Inspecting {len(files)} files...")
    for f in tqdm(files):
        try:
            img = nib.load(f)
            header = img.header
            zooms = header.get_zooms()
            dims = img.shape
            stats.append({
                'file': os.path.basename(f),
                'path': f,
                'v_x': zooms[0],
                'v_y': zooms[1],
                'v_z': zooms[2],
                'dim_x': dims[0],
                'dim_y': dims[1],
                'dim_z': dims[2]
            })
        except Exception as e:
            print(f"Error reading {f}: {e}")

    df = pd.DataFrame(stats)
    return df

if __name__ == "__main__":
    # Example usage:
    # DATA_DIR = "data/raw"
    # df = inspect_voxels(DATA_DIR)
    # if df is not None:
    #     print(df.describe())
    #     df.to_csv("voxel_report.csv", index=False)
    pass
