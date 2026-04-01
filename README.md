# 🧠 MS Lesion Segmentation: A Multi-Modal 3D Deep Learning Approach

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-1.3+-ade048.svg)](https://monai.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Project Overview
Multiple Sclerosis (MS) is a chronic autoimmune disease where the immune system attacks the myelin sheath, resulting in lesions within the Central Nervous System. Manually segmenting these lesions in 3D MRI volumes is a labor-intensive, error-prone task for radiologists.

This project implements a **Robust Multi-Modal 3D Residual U-Net pipeline** designed to automate the detection and segmentation of MS lesions. By fusing information from **FLAIR, T1, and T2 MRI sequences**, the model achieves a **3.5x performance improvement** over single-modality baselines, providing a scalable solution for clinical diagnostic support.

---

## 🏗️ Technical Architecture

### 1. The Model: 3D Residual BasicUNet
We transitioned from a standard UNet to a **Residual BasicUNet** to better handle the complexities of 3D medical volumes.
- **Encoder-Decoder Depth:** 5 levels with feature maps ranging from 32 to 512 channels.
- **Normalization:** **Instance Normalization** (preferred over BatchNorm to maintain stability with a batch size of 1).
- **Activation:** **LeakyReLU** to mitigate the vanishing gradient problem in deep 3D layers.
- **Output:** Voxel-wise probability map for lesion vs. healthy tissue.

### 2. Multi-Modal Fusion (Ablation C)
The final architecture utilizes a **three-channel input strategy**:
- **FLAIR:** Primary sequence for detecting bright MS plaques.
- **T1-weighted:** Provides anatomical context for gray/white matter boundaries.
- **T2-weighted:** Captures edema and water content, providing complementary signals to FLAIR.

### 3. Advanced Training Strategy
- **Hybrid Loss Optimization:** 
    - **Phase 1:** `DiceCELoss` for stable, symmetric convergence.
    - **Phase 2:** `DiceFocalLoss` fine-tuning to amplify gradients for "hard" examples (small/ambiguous lesions).
- **Optimizer:** `AdamW` with **Cosine Annealing and Warm Restarts**.
- **Memory Efficiency:** **Gradient Accumulation** (Effective Batch Size = 8) and **AMP (Mixed Precision)** training to fit large 3D patches (96³) into 24GB VRAM.

---

## 🧪 Experimental Results & Ablation Study

We conducted a systematic ablation study to quantify the impact of multi-modality and architectural refinements.

| Configuration | Input Modalities | Loss Function | Best Dice |
| :--- | :--- | :--- | :--- |
| **Baseline (v1)** | FLAIR Only | BCE + Dice | 0.0670 |
| **Ablation A** | FLAIR Only | DiceCELoss | 0.0762 |
| **Ablation B** | FLAIR + T1 | DiceCELoss | 0.1660 |
| **Ablation C (Final)**| **FLAIR + T1 + T2** | **DiceFocal (Fine-tuned)** | **0.2326** |

**Key Metric:** The model achieved a final Dice score of **0.2326**, representing a research-grade segmentation capability given the high class imbalance (<0.6% lesion voxels).

---

## 🛠️ Engineering Challenges & Robustness

This project involved significant "in-the-trenches" engineering to ensure model stability and data integrity:

- **The Long-MR-MS Data Bug:** Identified a critical labeling mismatch where longitudinal follow-up scans were being paired with baseline masks. Fixed this by implementing a `study1` temporal filter, significantly raising the performance ceiling.
- **Bias Initialization Trick:** Initialized the final convolution bias to `-4.0` (`sigmoid ≈ 0.018`). This aligned the model's initial predictions with the true class prior (lesions occupy ~1-2% of volume), preventing the "False Positive Collapse" seen in early experiments.
- **Resilient Preprocessing:** Developed a custom N4 Bias Correction and 1mm Isotropic Resampling pipeline using `SimpleITK` and `MONAI` to ensure scanner-agnostic generalization.
- **Checkpoint Safety:** Implemented a **Top-3 Asynchronous Checkpointing** system to prevent data loss during hardware/driver crashes on high-utilization university clusters.

---

## 💻 Clinical Dashboard
The project includes a **Gradio-based Inference Dashboard** (`Project_Demo.ipynb`), allowing clinicians to:
1. Upload multi-modal NIfTI scans.
2. Generate 3D overlays of detected lesions.
3. View longitudinal progression reports and volumetric statistics.

---

## 🚀 Technical Stack
- **Frameworks:** PyTorch, MONAI (Medical Open Network for AI)
- **Imaging Libraries:** SimpleITK, NiBabel (NIfTI handling)
- **Visualization:** Matplotlib, Gradio
- **Compute:** Trained on **NVIDIA RTX 4500 Ada Generation** (SRMIST Cluster)
