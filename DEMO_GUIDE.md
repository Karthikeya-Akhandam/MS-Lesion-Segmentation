# MS Lesion Segmentation — Demo Guide

## File Structure

```
code/
├── utils.py            ← shared engine (model, preprocessing, inference, metrics)
├── Project_Demo.ipynb  ← Gradio UI (imports from utils.py)
├── app.py              ← Streamlit UI (imports from utils.py)
└── models/
    └── best_model.pth
```

`utils.py` is the single source of truth — both UIs import from it. No duplicated logic.

---

## Option A — Gradio (Notebook)

### What it does

`Project_Demo.ipynb` launches a Gradio web dashboard with two tabs:
- **Single-Scan Analysis** — upload one FLAIR `.nii` / `.nii.gz`, get axial/coronal/sagittal overlay + lesion count/volume/confidence report. Optional ground-truth mask for Dice/sensitivity/precision metrics.
- **Longitudinal Tracking** — upload Baseline (T0) + Follow-up (T1), auto-registers them and shows new (green) vs resolved (blue) lesions

### Running locally

```bash
pip install monai[all] SimpleITK nibabel gradio matplotlib torch streamlit
```

Place the model at `./models/best_model.pth`, then open `Project_Demo.ipynb` and run all cells top to bottom.

After the last cell runs you will see:
```
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://xxxx.gradio.live   ← share link (valid 72h)
```

### Running on Google Colab

1. Upload `Project_Demo.ipynb` **and `utils.py`** to the same Colab working directory (or sync both via Drive)
2. Place model at `My Drive/MS_Project/models/best_model.pth`
3. Cell 2 auto-detects Colab and mounts Drive — authenticate when prompted
4. Run all cells — use the public Gradio URL (Colab blocks localhost)
5. GPU runtime recommended: Runtime → Change runtime type → T4 GPU

---

## Option B — Streamlit (app.py)

### Running locally

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Same two modes as Gradio (Single-Scan + Longitudinal).

### Running on Colab / remote

```python
# In a notebook cell:
!streamlit run app.py &
# Then use a tunnel (e.g. localtunnel or ngrok) to expose the port
```

---

## Model Path

| Environment | Path |
|-------------|------|
| Local | `./models/best_model.pth` |
| Colab | `/content/drive/MyDrive/MS_Project/models/best_model.pth` |

---

## Using an Interim Model

Training is still in progress. You can demo with an older checkpoint — the UI works correctly regardless of model quality.

**Expected behaviour with current model (~Dice 0.07):**
- Large lesions usually detected
- Small lesions (<10 vox) often missed
- Overlays, reports, and confidence metrics all display correctly

When a better `best_model.pth` is ready, drop it into `./models/` and restart the app — no code changes needed.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `WARNING: Model not found` | Check path: local → `./models/best_model.pth`, Colab → Drive path above |
| Gradio won't launch | Port 7860 in use → restart kernel and re-run last cell |
| Streamlit won't launch | Port 8501 in use → `streamlit run app.py --server.port 8502` |
| `CUDA out of memory` | Reduce `sw_batch_size` from `4` to `2` in `utils.segment_volume()` |
| N4 bias correction slow | Large volumes (>300mm/axis) can take 1–2 min on CPU — wait it out |
| Colab: `utils not found` | Ensure `utils.py` is in the same directory as the notebook (`/content/`) |
| Upload fails for `.nii.gz` | Use `.gz` extension in the file picker |
| Drive not mounting | Re-run setup cell and re-authenticate |

---

## Quick Test Without Real Data

Use any preprocessed volume from training:
```
data/processed/sub-001/image.nii.gz
```

Note: the demo re-runs preprocessing on upload, so preprocessed files will be double-processed — results still work for a quick UI check.
