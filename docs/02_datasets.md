# Datasets

## What is an MRI Modality?

An MRI scanner can be programmed to highlight different tissue types by changing its pulse sequence. Each "modality" is the same brain scanned with different settings:

| Modality | Full Name | What it highlights |
|---|---|---|
| **FLAIR** | Fluid Attenuated Inversion Recovery | MS lesions appear **bright white** — primary diagnostic sequence |
| **T1** | T1-weighted | Grey/white matter contrast — healthy anatomy |
| **T2** | T2-weighted | Water content — lesions also bright, but less specific than FLAIR |

This project uses all three modalities stacked as separate input channels to the neural network, like using RGB channels in a colour image.

---

## Dataset 1 — MSLesSeg

- **Source**: Public Figshare dataset (5-part ZIP)
- **Patients**: 78
- **Modalities**: FLAIR, T1, T2
- **Ground truth masks**: Binary NIfTI files with `*MASK*` in filename
- **Role**: Primary training dataset

**File naming convention:**
```
{SubjectID}_{TimePoint}_FLAIR.nii.gz  → image
{SubjectID}_{TimePoint}_MASK.nii.gz   → lesion mask
```

Pairing logic: Match by `{SubjectID}_{TimePoint}` prefix.

---

## Dataset 2 — Mendeley

- **Source**: Mendeley Data (public)
- **Patients**: 60
- **Modalities**: FLAIR, T1, T2
- **Ground truth masks**: `{ID}-LesionSeg-Flair.nii`
- **Role**: Cross-dataset validation (different scanner)

**File naming convention:**
```
{ID}-Flair.nii           → FLAIR image
{ID}-T1.nii              → T1 image
{ID}-T2.nii              → T2 image
{ID}-LesionSeg-Flair.nii → ground truth mask
```

---

## Dataset 3 — Long-MR-MS (Longitudinal)

- **Source**: Public longitudinal MS dataset
- **Patients**: 20 (each has 2 timepoints: study1 = baseline, study2 = follow-up)
- **Modalities**: FLAIR, T1, T2
- **Ground truth masks**: ONE mask per patient (only for study1/baseline)
- **Role**: Longitudinal tracking

### Critical Bug Found and Fixed

**The bug:** The original data loader loaded both `study1_FLAIRreg.nii.gz` and `study2_FLAIRreg.nii.gz` for each patient, but both were paired with the same single `*_gt.nii.gz` mask.

**Why this is wrong:** The study2 scan is a follow-up scan from a later date. The patient's lesions have changed between study1 and study2. Pairing study2's image with study1's mask means the model is trained on **incorrect labels** — it gets penalized for finding real lesions that appear in study2 but weren't in the study1 mask.

**Impact:** ~40 training samples (20 patients × 2 timepoints) were contaminated with noisy labels, acting as a noise floor that capped Dice improvement.

**Fix applied:**
```python
# Before (BUG):
flairs = glob(os.path.join(p_folder, '*_FLAIRreg.nii.gz'))

# After (FIX):
flairs = [f for f in glob(os.path.join(p_folder, '*_FLAIRreg.nii.gz'))
          if 'study1' in os.path.basename(f).lower()]
```

---

## Preprocessing Pipeline

Each raw NIfTI volume goes through this pipeline before training:

```
Raw NIfTI
    ↓
1. N4 Bias Field Correction (SimpleITK)
   — removes low-frequency intensity inhomogeneity from scanner coil
    ↓
2. RAS+ Reorientation
   — standardises axes: Right-Anterior-Superior coordinate system
    ↓
3. Resample to 1mm Isotropic
   — bilinear interpolation for images, nearest-neighbour for masks
   — ensures all volumes have the same voxel size regardless of scanner
    ↓
4. Z-score Normalisation + ScaleIntensity [0, 1]
   — mean=0, std=1 per volume; then rescaled to [0,1]
    ↓
Saved to data/processed/sub-{idx:03d}/image.nii.gz + label.nii.gz
```

---

## Final Dataset Statistics

| Ablation | Channels | Pairs Found | Notes |
|---|---|---|---|
| A (FLAIR only) | 1 | ~158 | All patients included |
| B (FLAIR+T1) | 2 | ~190 | Drops patients missing T1 |
| **C (FLAIR+T1+T2)** | **3** | **266** | Drops patients missing any modality |

> Note: 266 > 153 because the data loading counts individual scan pairs, not unique patients. 153 refers to perfectly aligned 3-channel patients used in the actual training split.

**Train/Validation split**: 85% train / 15% validation (stratified, fixed random seed)

---

## Data Directory Structure

```
data/
├── raw/
│   ├── MSLesSeg/
│   │   └── {Subject}_{Time}_FLAIR.nii.gz
│   │   └── {Subject}_{Time}_MASK.nii.gz
│   ├── Mendeley/
│   │   └── {ID}-Flair.nii
│   │   └── {ID}-LesionSeg-Flair.nii
│   └── Long-MR-MS/
│       └── patient{N}/
│           └── {N}_study1_FLAIRreg.nii.gz
│           └── {N}_gt.nii.gz
└── processed/
    └── sub-000/
    │   └── image.nii.gz   ← stacked 3-channel volume
    │   └── label.nii.gz   ← binary lesion mask
    └── sub-001/ ...
```
