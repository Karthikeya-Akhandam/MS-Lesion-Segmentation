# Project Overview — MS Lesion Segmentation

## What is Multiple Sclerosis (MS)?

Multiple Sclerosis is a chronic disease of the central nervous system where the immune system attacks the protective myelin sheath around nerve fibres. This damage appears as **lesions (white spots)** in MRI brain scans. Neurologists track these lesions over time to:
- Diagnose MS
- Monitor disease progression
- Evaluate whether treatment is working

Manually counting and measuring lesions in 3D MRI volumes is extremely time-consuming for radiologists and subject to human error. This is why automated segmentation is clinically important.

---

## What This Project Does

This project builds a **deep learning pipeline** that takes a patient's MRI scan as input and automatically produces a 3D mask showing where the lesions are.

```
Input:  3D MRI scan (FLAIR + T1 + T2 sequences)
Output: 3D binary mask (1 = lesion voxel, 0 = healthy tissue)
```

The model is a **3D U-Net** — a convolutional neural network architecture that has been the standard for medical image segmentation since 2015. It processes the brain volume in 96×96×96 voxel patches and learns to distinguish lesion tissue from healthy tissue.

---

## Project Title

**Robust Multi-Modal Deep Learning Framework for Generalizable and Longitudinal Multiple Sclerosis Lesion Segmentation**

---

## Why This is Hard

MS lesions are:
- **Tiny**: Many lesions are < 10 voxels (smaller than a grain of rice in 3D)
- **Variable**: They appear in different locations, sizes and shapes per patient
- **Rare**: Lesions occupy < 1% of the total brain volume → the model sees mostly healthy tissue
- **Multi-dataset**: Models trained on one hospital's scanner often fail on another scanner

These challenges are addressed through careful loss function design, patch sampling strategy, and multi-modal input.

---

## Deliverables

| Deliverable | Description |
|---|---|
| Trained model weights | `best_model.pth` — BasicUNet achieving Dice 0.2326 |
| Ablation study | Comparison of 1-channel, 2-channel, 3-channel inputs |
| Training notebook | `Project_Training_v3.ipynb` — fully reproducible pipeline |
| Demo app | `Project_Demo.ipynb` — Gradio-based clinical demo |

---

## Datasets Used

| Dataset | Patients | Modalities | Purpose |
|---|---|---|---|
| MSLesSeg | 78 | FLAIR, T1, T2 | Primary training |
| Mendeley | 60 | FLAIR, T1, T2 | External validation |
| Long-MR-MS | 20 | FLAIR, T1, T2 (longitudinal) | Temporal tracking |
| **Total** | **158 raw → 153 aligned** | 3-channel | Combined training |

---

## Hardware

Training was performed on the **NVIDIA RTX 4500 Ada Generation GPU** at the SRMIST university system.

| Spec | Value |
|---|---|
| GPU | NVIDIA RTX 4500 Ada Generation |
| VRAM | 25.76 GB |
| CUDA Compute | 8.9 |
| Multiprocessors | 60 |
| Framework | PyTorch + MONAI |

---

## Final Results Summary

| Version | Config | Best Dice | Epochs |
|---|---|---|---|
| v1 | UNet, FLAIR only, BCE+Dice | 0.067 | 150 |
| v2 | BasicUNet, FLAIR only | ~0.076 | 150 |
| Ablation B | BasicUNet, FLAIR+T1, DiceCELoss | 0.166 | 80 |
| **Ablation C** | **BasicUNet, FLAIR+T1+T2, DiceCELoss→DiceFocal** | **0.2326** | **200** |

**This represents a 3.5× improvement over the v1 baseline (0.067 → 0.2326).**

For context, clinical-grade MS segmentation tools (e.g., LST-AI) achieve Dice ~0.65+. The gap reflects the scale of training data required (tens of thousands of patients vs our 153). For a university minor project, 0.2326 with a correct, reproducible methodology is a strong result.

---

## Technology Stack

- **Python 3.10**
- **PyTorch** — deep learning framework
- **MONAI** — medical imaging extensions for PyTorch
- **SimpleITK** — N4 bias correction, image registration
- **NiBabel** — reading NIfTI MRI files
- **Matplotlib** — visualisation
- **Telegram Bot API** — remote training monitoring
