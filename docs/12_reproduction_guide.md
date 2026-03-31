# Reproduction Guide

Step-by-step instructions to reproduce the project results from scratch. Written for someone who has never run the project before.

---

## Prerequisites

### Hardware
- GPU with at least 8 GB VRAM (16 GB recommended for batch_size=1 with 96³ patches)
- Tested on: NVIDIA RTX 4500 Ada (25.76 GB), Google Colab T4 (16 GB)

### Software
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install monai[all]
pip install SimpleITK nibabel tqdm gradio matplotlib scipy
```

Or install all at once:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
pip install "monai[all]" SimpleITK nibabel tqdm gradio matplotlib scipy
```

---

## Step 1: Download Datasets

### Option A: Use Data_Downloader.ipynb
Run `Data_Downloader.ipynb` — this uses `aria2c` to download all 3 datasets to Google Drive or local storage.

### Option B: Manual Download
- **MSLesSeg**: Download 5-part ZIP from Figshare, extract to `data/raw/MSLesSeg/`
- **Mendeley**: Download from Mendeley Data, extract to `data/raw/Mendeley/`
- **Long-MR-MS**: Download from public source, extract to `data/raw/Long-MR-MS/`

Expected structure after download:
```
data/raw/
├── MSLesSeg/
│   ├── {Subject}_{Time}_FLAIR.nii.gz
│   └── {Subject}_{Time}_MASK.nii.gz
├── Mendeley/
│   ├── {ID}-Flair.nii
│   └── {ID}-LesionSeg-Flair.nii
└── Long-MR-MS/
    └── patient{N}/
        ├── {N}_study1_FLAIRreg.nii.gz
        └── {N}_gt.nii.gz
```

---

## Step 2: Open Project_Training_v3.ipynb

This is the main notebook. Run cells in order from top to bottom.

### Cell order and purpose:

| Cell | Purpose | Run? |
|---|---|---|
| Cell 1 | Environment detection (local vs Colab) | Always |
| Cell 2 | Path configuration (set DRIVE_ROOT) | Always |
| Cell 3 | GPU check + library imports | Always |
| Cell 4 | Telegram bot setup (optional) | Optional |
| Cells 5–10 | Preprocessing functions | Always |
| Cell (ABLATION config) | Set ABLATION = "C" | Always |
| Preprocessing cell | Run preprocessing pipeline | **First time only** |
| Model cell | Define BasicUNet | Always |
| Loss cell | Define MSLesionLoss | Always |
| Training cell | Run training | Always |

---

## Step 3: Set Environment

In Cell 2, set:
```python
ENVIRONMENT = "local"   # or "colab" if running on Google Colab
DRIVE_ROOT  = "G:/My Drive"  # path to your Google Drive sync folder
                              # or local data directory
```

---

## Step 4: Configure Ablation

Find the cell containing `ABLATION = "C"` and verify it is set to `"C"` for the 3-channel (FLAIR+T1+T2) run.

```python
ABLATION = "C"   # A=FLAIR only, B=FLAIR+T1, C=FLAIR+T1+T2
```

---

## Step 5: Run Preprocessing

Run the preprocessing cell. This will:
1. Scan all 3 dataset folders
2. Apply N4 bias correction, reorientation, resampling, normalisation
3. Save processed volumes to `data/processed/sub-000/` through `sub-{N}/`

**Expected output:**
```
✅ Ablation C | 3 channel(s) | 266 pairs found
0 cached, 266 still need processing — running pipeline...
[Processing sub-000...] ...
Preprocessing complete. 266 volumes saved.
```

This step takes ~2–4 hours on CPU. It only needs to run once — subsequent runs detect cached files and skip.

**IMPORTANT**: If you change the ABLATION setting after preprocessing, you must delete `data/processed/` and re-run. The cached files contain the wrong number of channels.

---

## Step 6: Train the Model

### Fresh training (default):
```python
EPOCHS        = 200
RESUME_CHECKPOINT = False  # start fresh
LR            = 1e-4
```

Run the training cell. Training will:
- Print epoch-by-epoch metrics
- Save `models/best_model.pth` when a new best Dice is achieved
- Save `models/checkpoint_latest.pth` every epoch
- Send Telegram messages every 5 epochs (if bot configured)

**Expected epoch 1 output:**
```
Training engine ready.
🔍 Dry run: Testing dataset load...
✅ Dataset OK.
Epoch   1/200 [train]: 100%|██| 227/227 [03:03]
[INTENSITY CHECK] mean=0.027  std=0.459  max=2.896
  [diag e1] pred_mean=0.0143  lt0.1=1.00  pred_pos_ratio=0.0000  label_pos=3858
Epoch   1 | Loss: 0.8545 | BestDice@0.5: 0.0245 | TPR@0.3: 0.167 | 4.6m
  New best Dice: 0.0245 -> ./models/best_model.pth
```

### Resuming from checkpoint:
If training is interrupted, set `RESUME_CHECKPOINT = True` and re-run the training cell. It will load `models/checkpoint_latest.pth` and continue from where it stopped.

---

## Step 7: Switch to DiceFocalLoss (Fine-tuning)

After 150 epochs of DiceCELoss, switch the loss function for fine-tuning:

1. Find the loss cell (contains `class MSLesionLoss`) and change it to DiceFocalLoss (already done in the current notebook)
2. Update training cell hyperparameters:
   ```python
   EPOCHS        = 200   # epochs 151-200 will run
   LR            = 5e-5  # reduced LR for fine-tuning
   RESUME_CHECKPOINT = True
   ```
3. Re-run the loss cell and training cell

---

## Step 8: Expected Results Checkpoints

| By Epoch | Expected Dice |
|---|---|
| 5 | > 0.04 (no collapse) |
| 10 | > 0.10 |
| 25 | > 0.15 |
| 75 | > 0.18 |
| 150 | > 0.21 (DiceCELoss best) |
| 200 | ~0.23 (DiceFocalLoss) |

If Dice is not meeting these checkpoints by epoch 25, check:
1. `label_pos` in diagnostic output — should be > 1000 (not zero)
2. `pred_pos_ratio` — should be > 0.001 by epoch 5
3. That `RESUME_CHECKPOINT=False` on a fresh run (no stale checkpoint loading)

---

## Step 9: Run the Demo

After training completes, open `Project_Demo.ipynb` and run all cells. This launches a Gradio web interface where you can:
1. Upload a patient's FLAIR, T1, and T2 scans
2. Get the predicted lesion mask
3. View an overlay visualisation

The demo loads `models/best_model.pth` automatically.

---

## Troubleshooting

| Error | Likely Cause | Fix |
|---|---|---|
| `RuntimeError: CUDA error: unknown error` | GPU crash | Restart kernel, set RESUME=True, re-run |
| `Trying to resize storage` | DataLoader collation | Ensure `pad_list_collate_clone` is used |
| `label_pos=0` every epoch | Wrong dataset path | Check `DATA_DIR` points to raw data |
| Dice stuck at 0.02 after ep10 | Model not learning | Check loss function is not collapsed |
| `ModuleNotFoundError: einops` | Missing dependency | `pip install einops` |
| `KeyError` loading checkpoint | Architecture mismatch | Set RESUME=False, delete old checkpoint |

---

## Model Files

| File | Contents | Size |
|---|---|---|
| `models/best_model.pth` | Best validation Dice weights | ~75 MB |
| `models/checkpoint_latest.pth` | Full training state (model+optimizer+scheduler) | ~225 MB |
| `models/top2_ep{N}_dice{D}.pth` | Second best checkpoint | ~75 MB |
| `models/top3_ep{N}_dice{D}.pth` | Third best checkpoint | ~75 MB |

Note: `models/` is in `.gitignore` — these files are not tracked by git and must be preserved manually.
