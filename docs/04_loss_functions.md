# Loss Functions

## Background: What is a Loss Function?

A loss function measures how wrong the model's predictions are. During training, the model adjusts its weights to minimise this number. The choice of loss function fundamentally determines what the model optimises for.

For MS lesion segmentation, choosing the wrong loss function causes the model to "cheat" by predicting all-background (no lesions), which gives low loss but is clinically useless.

---

## The Core Problem: Class Imbalance

In a 96³ patch (≈ 884,736 voxels):
- Lesion voxels: ~500–5000 (< 0.6%)
- Background voxels: > 99.4%

A naive loss function (plain Binary Cross Entropy) is trivially minimised by predicting "no lesion" everywhere. This is called **trivial solution collapse**.

---

## Mathematical Formulas

### Dice Loss

Measures overlap between prediction and ground truth:

```
Dice = (2 × |P ∩ G|) / (|P| + |G|)

DiceLoss = 1 - Dice
```

Where:
- P = predicted positive voxels
- G = ground truth positive voxels
- |·| = count (or sum of probabilities)

Dice Loss is **class-imbalance resistant** because it only considers positive voxels (ignores background). A score of 0 = no overlap, 1 = perfect overlap.

---

### Dice + Cross Entropy (DiceCELoss)

```
DiceCELoss = DiceLoss + CrossEntropyLoss
```

Cross Entropy adds a per-voxel classification signal. The combination:
- DiceLoss handles global overlap (lesion vs background balance)
- CE handles per-voxel probability calibration

This is symmetric — no directional bias toward false positives or false negatives.

---

### Focal Loss

Focal Loss down-weights "easy" examples and focuses learning on "hard" examples:

```
FocalLoss = -α × (1 - p_t)^γ × log(p_t)

Where:
  p_t  = predicted probability for correct class
  γ    = focusing parameter (we use γ=2.0)
  α    = class weight (balanced by default)
```

When γ=2: If a voxel is predicted with 90% confidence correctly, its loss contribution is reduced by (1-0.9)² = 0.01×. Hard misclassified voxels dominate learning.

**For MS lesions**: Small, ambiguous lesions are "hard examples". Focal Loss forces the model to prioritise learning them.

---

### DiceFocalLoss

```
DiceFocalLoss = DiceLoss + FocalLoss(γ=2.0)
```

Combines volumetric overlap optimisation (Dice) with hard-example focusing (Focal). Used as the fine-tuning finisher in this project.

---

## Loss Function History

### v1: DiceLoss + BCE(pos_weight=50)

```python
DiceLoss() + BCEWithLogitsLoss(pos_weight=torch.tensor([50.0]))
```

- pos_weight=50 means lesion voxels contribute 50× more to BCE loss
- **Collapsed at epoch 45**: The pos_weight was too aggressive. The model learned to predict everything as lesion (all-positive collapse), minimising the BCE component while Dice collapsed.
- Best Dice achieved: **0.067**

---

### v2: Focal Tversky Loss (α=0.3) — FAILED

```python
TverskyLoss(sigmoid=True, alpha=0.3, beta=0.7)
# plus focal: loss = tversky_loss ** 1.5
```

Tversky loss is a generalisation of Dice that penalises FN and FP differently:
```
Tversky = TP / (TP + α×FP + β×FN)
```
With α=0.3, β=0.7: false negatives penalised 2.3× more than false positives → supposed to recover small lesions.

**Collapsed at epoch 19 (precision collapse)**: With α=0.3, FP penalty was so low that the model learned to predict everything as lesion. PPV (precision) dropped to 1.5%.

---

### v2 Fixed: Focal Tversky (α=0.5) — STABLE

```python
TverskyLoss(sigmoid=True, alpha=0.5, beta=0.5)
# Symmetric — equal FP/FN penalty
# focal: loss = tversky_loss ** 1.5
```

α=0.5, β=0.5 makes Tversky equivalent to Dice Loss. Adding the `**1.5` focal modulation amplifies gradients at low Tversky index (when model is performing poorly).

**Result**: Stable training, no collapse. Best Dice ~0.19.

---

### v3: DiceCELoss — PRIMARY TRAINING PHASE

```python
from monai.losses import DiceCELoss

DiceCELoss(
    to_onehot_y=False,
    sigmoid=True,
    squared_pred=True,
    smooth_nr=1e-5,
    smooth_dr=1e-5,
)
```

Switched to DiceCELoss after confirming it is symmetric, stable, and well-suited for random-init networks. The `squared_pred=True` makes the denominator more numerically stable.

**Result**: Trained from epoch 1 to 150. Best Dice: **0.2193**. Plateau hit at ~epoch 54, slow improvement to epoch 150.

---

### v3.2: DiceFocalLoss — FINE-TUNING FINISHER

```python
from monai.losses import DiceFocalLoss

DiceFocalLoss(
    to_onehot_y=False,
    sigmoid=True,
    gamma=2.0,
    squared_pred=True,
    smooth_nr=1e-5,
    smooth_dr=1e-5,
)
```

Switched at epoch 151 after DiceCELoss plateau at 0.2193. The focal term (γ=2.0) amplifies gradient signal from hard examples — the small and partial lesions that DiceCELoss treated equally.

**Result**: Resumed from best DiceCELoss checkpoint (0.2193). Fine-tuned epochs 151–200. Best Dice: **0.2326**.

The improvement is modest (+0.013) but the loss function was operating at low LR (5e-5 vs original 1e-4) to avoid destabilisation. Higher LR with DiceFocalLoss caused instability at cycle restarts.

---

## Loss Function Summary Table

| Loss | Used In | Best Dice | Why Switched |
|---|---|---|---|
| Dice + BCE(pos_weight=50) | v1 | 0.067 | Collapsed ep45 (all-positive) |
| Focal Tversky α=0.3 | v2 attempt | ~0.05 | Precision collapse ep19 |
| Focal Tversky α=0.5 | v2 fixed | 0.190 | Plateau; switched to DiceCELoss |
| DiceCELoss | v3 (ep1–150) | 0.2193 | Plateau; fine-tune with DiceFocal |
| **DiceFocalLoss γ=2.0** | **v3.2 (ep151–200)** | **0.2326** | **Final** |

---

## Key Insight: Why Loss Switching Works

Training in two phases is a deliberate strategy:

1. **Phase 1 (DiceCELoss)**: Symmetric loss establishes stable feature representations. No directional bias — good for learning from scratch.

2. **Phase 2 (DiceFocalLoss)**: Once the model has learned coarse segmentation, switch to a loss that amplifies hard examples. The model already knows where lesions approximately are; now it's pushed to get the small/ambiguous ones right.

This is analogous to how a student first learns fundamentals (symmetrical feedback) and then focuses on their weak areas (targeted difficulty).
