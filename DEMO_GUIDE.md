# MS Lesion Segmentation — Demo Guide

## What the Demo Does

`Project_Demo.ipynb` launches a Gradio web dashboard with two tabs:
- **Single-Scan Analysis** — upload one FLAIR `.nii` / `.nii.gz`, get axial/coronal/sagittal overlay + lesion count/volume report
- **Longitudinal Tracking** — upload Baseline (T0) + Follow-up (T1), auto-registers them and shows new (green) vs resolved (blue) lesions

---

## Running Locally

### 1. Prerequisites

```bash
pip install monai[all] SimpleITK nibabel gradio matplotlib torch
```

### 2. Place the model file

The notebook looks for the model at:
```
./models/best_model.pth
```

Put your `.pth` file there:
```
Minor Project/code/
├── models/
│   └── best_model.pth   ← here
├── Project_Demo.ipynb
└── ...
```

If you don't have `models/` yet:
```bash
mkdir models
```
Then copy your old/current model into it.

### 3. Run the notebook

Open `Project_Demo.ipynb` in Jupyter or VS Code and run all cells top to bottom (Cell 1 → Cell 4).

- **Cell 2** — installs packages, detects environment, sets model path
- **Cell 3** — defines the preprocessing + inference + visualisation engine
- **Cell 4** — loads the model and launches Gradio

### 4. Access the dashboard

After Cell 4 runs you will see two URLs printed:
```
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://xxxx.gradio.live   ← share link (valid 72h)
```

Open the local URL in your browser. Use the public URL to demo on any device on the same network or share it.

### 5. Upload a scan

- Input format: `.nii` or `.nii.gz` (FLAIR MRI)
- The pipeline runs N4 bias correction + skull strip + 1mm resampling automatically before inference
- For testing you can use any FLAIR volume from the `data/processed/` folder (the `image.nii.gz` files)

---

## Running on Google Colab

### 1. Upload the notebook

Upload `Project_Demo.ipynb` to your Google Drive or open it directly in Colab via File → Open → GitHub/Drive.

### 2. Place the model on Google Drive

The notebook expects:
```
My Drive/
└── MS_Project/
    └── models/
        └── best_model.pth   ← here
```

Upload your model file to that path before running.

### 3. Run all cells

Cell 2 auto-detects Colab and mounts your Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

It will ask you to authenticate with your Google account — click the link, sign in, copy the code back.

After that, all cells run identically to local.

### 4. Get the public URL

Colab blocks `localhost`, so always use the public Gradio link:
```
Running on public URL: https://xxxx.gradio.live
```

This link works from any browser for 72 hours.

### 5. GPU runtime (recommended)

Go to Runtime → Change runtime type → T4 GPU.
Inference on CPU is slow (~2–3 min per volume). GPU brings it to ~15–30 sec.

---

## Using an Old / Interim Model

The current training run is still in progress. You can demo with an older checkpoint — the pipeline works the same, segmentation quality will just be lower (fewer lesions detected, smaller masks).

**Expected behaviour with an undertrained model:**
- Dice ~0.03–0.06 → lesions will be partially detected, some missed
- Large lesions are usually found; small lesions (<10 vox) often missed
- The Gradio UI, overlays, and report format all work correctly regardless

When the new `best_model.pth` is ready, just drop it into `./models/` (local) or Drive (Colab) and re-run Cell 4 only — no restart needed.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `WARNING: Model not found` | Check path: local → `./models/best_model.pth`, Colab → `My Drive/MS_Project/models/best_model.pth` |
| Gradio won't launch | Port 7860 in use → restart kernel and re-run Cell 4 |
| `CUDA out of memory` | Inference uses `sw_batch_size=8` — reduce to `4` in `segment_volume()` in Cell 3 |
| N4 bias correction hangs | Very large volumes (>300mm per axis) can be slow on CPU — wait up to 2 min |
| Upload fails for `.nii.gz` | Gradio file picker sometimes needs the full extension typed manually — use `.gz` |
| Drive not mounting on Colab | Re-run Cell 2 and re-authenticate |

---

## Quick Test Without Real Data

If you don't have a patient scan handy, use any preprocessed volume from training:

```
data/processed/sub-001/image.nii.gz   ← already bias-corrected and normalised
```

Note: the demo re-runs preprocessing on whatever you upload, so preprocessed files will be double-processed — results will still look reasonable for a quick UI check.
