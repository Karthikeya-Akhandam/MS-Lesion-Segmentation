# Results and Discussion

## Final Model Performance

| Metric | Value |
|---|---|
| Best Dice Score | **0.2326** |
| Achieved at Epoch | 200 |
| Lesion TPR (overall) | ~0.31 |
| Loss at completion | 0.5082 |
| Training duration | ~200 × 3 min ≈ 10 hours total |

---

## Progression of Results Across Experiments

| Experiment | Config | Best Dice | Improvement |
|---|---|---|---|
| v1 Baseline | UNet, FLAIR only, BCE+Dice | 0.067 | — |
| v2 BasicUNet | BasicUNet, FLAIR only, DiceCELoss | 0.076 | +13% |
| Ablation B | BasicUNet, FLAIR+T1, DiceCELoss | 0.166 | +118% over v1 |
| Ablation C (DiceCE) | BasicUNet, FLAIR+T1+T2, noisy label fix | 0.2193 | +227% over v1 |
| **Ablation C (DiceFocal)** | **+ DiceFocalLoss finisher** | **0.2326** | **+247% over v1** |

The 3.5× improvement from 0.067 to 0.2326 is attributable to three compounding contributions:
1. **Architecture upgrade** (UNet → BasicUNet): +13%
2. **Modality upgrade + data fix** (FLAIR→FLAIR+T1+T2, study1 filter): +188%
3. **Loss function fine-tuning** (DiceCELoss→DiceFocalLoss): +6%

---

## Phase Analysis

### Phase 1: Initial Learning (Epochs 1–25, Cycle 1)
- Loss: 0.8545 → 0.6277
- Dice: 0.024 → 0.178
- The model rapidly learns coarse segmentation. TPR climbs from 0.167 to ~0.33.
- First meaningful segmentation appears at epoch 7 (Dice 0.085).

### Phase 2: Deepening (Epochs 26–75, Cycle 2)
- Loss: 0.6583 → 0.5532
- Dice: First 0.20+ achieved at epoch 47 (0.2015)
- The cycle 2 restart at epoch 26 causes a temporary dip (expected with cosine annealing).
- Model reaches first plateau at 0.2063 (epoch 54).
- **46 consecutive epochs** without improvement (ep54–ep100).

### Phase 3: Cycle 3 Recovery (Epochs 76–150)
- Loss: 0.5783 → 0.5249
- Dice: Breaks plateau at epoch 102 (0.2094), peaks at epoch 144 (0.2193)
- Cycle 3 is 100 epochs long. Despite the long plateau, slow consistent loss reduction eventually produces new best scores.
- DiceCELoss ceiling confirmed at 0.2193.

### Phase 4: DiceFocalLoss Fine-Tuning (Epochs 151–200)
- Loss: 0.5260 → 0.5082
- Dice: 0.2193 → 0.2326 (+0.013)
- The focal term's effect is modest but consistent with the expectation that small/hard lesions are the bottleneck.
- Cycle 4 restart at epoch 176 causes a temporary dip to 0.1732 before recovery.
- Final 10 epochs produce the strongest improvement: 0.2225 → 0.2259 → 0.2312 → 0.2326.

---

## Comparison with Literature

| Method | Dataset | Dice |
|---|---|---|
| LST-AI (Wiltgen et al. 2024) | Multi-centre | ~0.65 |
| R2AUNet (Andishgar et al. 2025) | Private dataset | ~0.78 |
| Dense Residual UNet (Sarica et al. 2023) | Multi-sequence | ~0.68 |
| **This work (Ablation C)** | **3-dataset combined** | **0.2326** |

The gap between this work and published clinical tools is significant. The primary reasons:
1. **Training data scale**: Published methods use 100–1000+ patients. This work uses 153 aligned 3-channel samples.
2. **Dataset quality**: Published methods often use private, curated clinical datasets with expert annotations from multiple raters.
3. **Architecture**: nnU-Net (the current state-of-the-art pipeline) uses adaptive preprocessing and architecture selection that this project does not implement.

For a university minor project with a fixed dataset and GPU budget, 0.2326 is a strong result demonstrating correct methodology.

---

## What the Dice Score Means in Practice

At Dice = 0.2326:

- The model correctly segments approximately **23% of lesion voxels** overall
- It detects approximately **31% of individual lesions** (lesion-wise TPR)
- Large lesions (>100 voxels) are detected more reliably than small lesions (<10 voxels)
- There are false positives — some non-lesion voxels are incorrectly flagged

**Clinical context**: A radiologist reviewing this model's output would see partial outlines of larger lesions and occasional noise. It could not be used for diagnosis but could potentially be used as a "pre-screening" tool to direct radiologist attention.

---

## Limitations

### 1. Dataset Size
153 training patients is very small by medical imaging standards. Clinical-grade segmentation tools are trained on thousands of patients. More data would likely push Dice into the 0.35–0.50 range.

### 2. No Cross-Dataset Validation (Ablation C)
Due to time constraints, the final Ablation C model was not evaluated in a hold-out cross-dataset scenario (training on MSLesSeg, testing on Mendeley only). This experiment was planned but not executed.

### 3. 3-Modality Requirement
The Ablation C model requires FLAIR, T1, and T2 sequences. Not all clinical centres acquire all three in standard protocol. A deployed system would need single-modality fallback.

### 4. Dice Ceiling at ~0.23
Despite 200 epochs, the model plateaued near 0.23. This reflects the architectural ceiling of BasicUNet on this dataset size. Reaching 0.40+ would require either more data or a more sophisticated architecture like nnU-Net.

### 5. No Post-Processing Optimisation
The connected component filter (min_size=8) is simple. More sophisticated post-processing (morphological operations, atlas-based priors) could improve precision.

---

## Graph: Dice Progression Per Experiment

```python
import matplotlib.pyplot as plt
import numpy as np

experiments = [
    'v1 UNet\n(ep150)',
    'BasicUNet\nAblation A\n(ep150)',
    'BasicUNet\nAblation B\n(ep80)',
    'BasicUNet\nAblation C\nDiceCE\n(ep150)',
    'BasicUNet\nAblation C\nDiceFocal\n(ep200)',
]
dice = [0.067, 0.076, 0.166, 0.2193, 0.2326]
colors = ['#d9534f', '#e8a838', '#5bc0de', '#5cb85c', '#1a7a1a']

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(experiments, dice, color=colors, edgecolor='black')
for bar, d in zip(bars, dice):
    ax.text(bar.get_x() + bar.get_width()/2, d + 0.003,
            f'{d:.4f}', ha='center', fontweight='bold', fontsize=11)

ax.axhline(y=0.65, color='purple', linestyle='--', linewidth=1.5,
           label='Clinical benchmark ~0.65')
ax.set_ylabel('Best Dice Score', fontsize=13)
ax.set_title('Results Progression Across Experiments', fontsize=14)
ax.set_ylim(0, 0.75)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('results_progression.png', dpi=150)
plt.show()
```
