# Issues Encountered and Fixes Applied

This document is a chronological log of every major engineering problem encountered during the project and how each was resolved. This is useful for the paper's "Implementation Challenges" or "Engineering Decisions" section.

---

## Issues 1–5: Preprocessing Pipeline Failures

**Symptom**: Data loading crashes with MetaTensor shape errors; `EnsureChannelFirstd` was double-applying the channel dimension.

**Root cause**: MONAI's `LoadImaged` returns `MetaTensor` objects that already carry metadata about channel dimensions. Applying `EnsureChannelFirstd` after the tensor had already been channel-expanded caused shape mismatch `(1, 1, D, H, W)` instead of `(1, D, H, W)`.

**Fix**: Restructured the transform pipeline to handle MetaTensor correctly. For multi-channel inputs, used `ConcatItemsd` to combine per-modality tensors along dim=0 after individual `EnsureChannelFirstd` calls.

**Lesson**: MONAI's MetaTensor has implicit channel tracking — transforms must be applied in strict order.

---

## Issues 6–17: DataLoader Collation Errors

**Symptom**: `RuntimeError: Trying to resize storage that is not resizable` during DataLoader batch assembly. Occurred because different brain volumes have different spatial dimensions — they can't be directly stacked into a batch tensor.

**Fix**: Custom collation function that clones all tensors before padding:

```python
def pad_list_collate_clone(batch):
    """Clone tensors before collating to avoid storage resize errors."""
    flat = []
    for b in batch:
        if isinstance(b, list):
            flat.extend(b)
        else:
            flat.append(b)
    for b in flat:
        for k, v in b.items():
            if hasattr(v, "as_tensor"):
                b[k] = v.as_tensor().clone()
            elif isinstance(v, torch.Tensor):
                b[k] = v.clone()
    return pad_list_data_collate(flat)

train_loader = DataLoader(..., collate_fn=pad_list_collate_clone)
```

The `.clone()` call creates a new contiguous memory allocation, making the tensor resizable for padding.

---

## Issues 18–23: Loss Function Collapses

### Issue 18: All-positive collapse (v1)
**Symptom**: After epoch 45, model predicted everything as lesion. Dice converged to 0.067 and stopped improving.

**Cause**: `BCEWithLogitsLoss(pos_weight=50)` applied 50× weight to positive class. With only 0.5% positive voxels, the model learned that predicting everything positive minimised the loss more efficiently than learning true boundaries.

**Fix**: Switched to Focal Tversky Loss which is inherently class-imbalance resistant.

### Issue 19: Precision collapse (v2 Focal Tversky α=0.3)
**Symptom**: By epoch 19, PPV (precision) dropped to 1.5%. Model was predicting ~80% of voxels as lesion.

**Cause**: `TverskyLoss(alpha=0.3, beta=0.7)` penalises false positives only 0.3× but false negatives 0.7×. The model found it cheaper to have unlimited false positives (low FP penalty) than miss any true lesion.

**Fix**: Changed to `alpha=0.5, beta=0.5` (symmetric — equal FP/FN penalty, equivalent to Dice Loss).

### Issue 20: Plateau at 0.19 (v2)
**Symptom**: Dice oscillated between 0.15–0.19 for 50+ epochs with no improvement.

**Cause** (later identified): Long-MR-MS dataset noisy labels (see Issue 27). Also, the network was capacity-limited with the v1 UNet architecture.

**Fix**: Upgraded to BasicUNet + identified and fixed the data bug.

---

## Issue 24: Stale Checkpoint Contamination (Critical Bug)

**Symptom**: Every new architecture experiment failed to improve beyond the previous run's ceiling, even with a "fresh" model. SwinUNETR started with Dice 0.058, AttentionUnet with 0.065 — both worse than v1 UNet.

**Root cause**: The training cell contained this resume logic:

```python
# BUG — this was in the code:
if RESUME_CHECKPOINT:
    model.load_state_dict(torch.load(ckpt_path))
elif os.path.exists(best_path):               # ← THIS LINE
    model.load_state_dict(torch.load(best_path))  # ← LOADED WRONG WEIGHTS
```

When `RESUME_CHECKPOINT=False`, the `elif` branch still loaded weights from the **previous experiment's** `best_model.pth`. A SwinUNETR model was loading weights shaped for a BasicUNet — causing silent failure (PyTorch's `load_state_dict` matched keys by name and ignored shape mismatches with `strict=False`).

**Fix**: Removed the `elif` block entirely. Fresh runs always start from random initialisation.

```python
# FIXED:
if RESUME_CHECKPOINT:
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    # ... restore optimizer/scheduler state
# NOTE: No elif block — fresh runs always start from random init.
```

**Impact**: This single bug invalidated ~5 days of experiments. Once fixed, BasicUNet immediately began training correctly.

---

## Issues 25–26: AttentionUnet Safe Mode Collapse

**Symptom**: After switching to Focal Tversky Loss with AttentionUnet, TPR declined from 0.428 at epoch 1 to 0.057 by epoch 25. The model learned to predict nothing (Safe Mode Collapse).

**Cause**:
1. Focal Tversky α=0.6 for AttentionUnet was too aggressive — the FP penalty was so severe that the model learned predicting zero was safer than risking false positives
2. AttentionUnet uses BatchNorm which with batch_size=1 produces meaningless normalisation statistics

**Fix**: Switched loss to DiceCELoss (symmetric, no directional bias) and switched architecture to BasicUNet (Instance Norm works correctly with batch_size=1).

---

## Issue 27: Long-MR-MS Noisy Label Bug

**Symptom**: Training stalled at Dice 0.166 despite all other fixes. Dataset analysis revealed that `label_pos` (number of positive lesion voxels per batch) was inconsistently zero in some batches.

**Root cause**: Long-MR-MS dataset has 2 timepoints per patient:
- `study1_FLAIRreg.nii.gz` — baseline scan
- `study2_FLAIRreg.nii.gz` — follow-up scan (1–2 years later)

Only ONE ground truth mask exists per patient (`*_gt.nii.gz`, corresponding to study1). The data loader was globbing ALL `*_FLAIRreg.nii.gz` files, so study2 scans were paired with the study1 mask.

Study2 scans show the patient's brain at a different timepoint — lesions have changed (some resolved, some new). Pairing with study1 mask meant the model was trained with incorrect labels for ~20 patients × 2 = ~40 samples.

**Fix**:
```python
# Filter to study1 only
flairs = [f for f in glob(os.path.join(p_folder, '*_FLAIRreg.nii.gz'))
          if 'study1' in os.path.basename(f).lower()]
```

Same filter applied to T1 and T2 modality lookups.

**Impact**: Combined with adding T2 channel (Ablation C), this fix contributed to pushing Dice from 0.166 → 0.2326.

---

## CUDA Unknown Error (Hardware Crash)

**Symptom**: At epoch 29 of the clean BasicUNet run, training stopped with:
```
RuntimeError: CUDA error: unknown error
```

**Cause**: GPU hardware/driver crash (not a code bug). Possibly thermal throttling or a driver fault on the university system.

**Recovery procedure**:
1. Restart the Jupyter kernel
2. Re-run all cells up to the training cell
3. Set `RESUME_CHECKPOINT = True`
4. Re-run training cell — resumes from `checkpoint_latest.pth` at epoch 28

The async checkpoint saving (every epoch in a background thread) meant only 1 epoch of progress was lost.

---

## Einops Not Installed (SwinUNETR)

**Symptom**: `ImportError: No module named 'einops'` when trying to import SwinUNETR.

**Cause**: SwinUNETR uses the `einops` library for tensor rearrangement, which is not included in standard PyTorch/MONAI installations.

**Fix**: `pip install einops` in a terminal cell.

**Secondary issue**: After installing einops, MONAI 1.5.2's SwinUNETR API had changed — `img_size` parameter was removed in this version.

**Fix**: Remove `img_size=(96,96,96)` from the constructor call.

---

## Summary Table

| Issue | Symptom | Root Cause | Fix | Epochs Lost |
|---|---|---|---|---|
| MetaTensor shape error | Crash on load | Double EnsureChannelFirst | Restructure transforms | ~3 |
| DataLoader resize error | Crash on batch | Non-contiguous tensor storage | pad_list_collate_clone | ~2 |
| All-positive collapse | Dice stall 0.067 | BCE pos_weight=50 too high | Switch to Focal Tversky | 150 |
| Precision collapse | PPV→1.5% | Tversky α=0.3 too low | α→0.5 (symmetric) | 25 |
| Stale checkpoint contamination | All experiments fail | elif block loading wrong weights | Remove elif block | ~5 days |
| AttentionUnet collapse | TPR→0.057 | BatchNorm+batch_size=1 | Switch to BasicUNet | 86 |
| Long-MR-MS noisy labels | Plateau at 0.166 | study2 paired with study1 mask | Filter to study1 only | indirect |
| CUDA crash | Training stops ep29 | Hardware/driver fault | Restart + resume | 1 |
