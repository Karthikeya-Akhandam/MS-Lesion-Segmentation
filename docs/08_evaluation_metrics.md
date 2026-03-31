# Evaluation Metrics

## Why Standard Metrics Are Not Enough for MS Lesions

In most classification problems, accuracy is a fair metric. For MS lesion segmentation, standard accuracy is meaningless — if a model predicts "no lesion" for every voxel, it achieves ~99.5% accuracy (since only ~0.5% of voxels are actually lesions). The model would be completely useless clinically but look excellent on paper.

This project uses a hierarchy of metrics that go beyond simple overlap scores.

---

## Metric 1: Dice Score (Primary Metric)

### What it measures
Spatial overlap between predicted mask and ground truth mask.

### Formula
```
Dice = (2 × |Prediction ∩ Ground Truth|) / (|Prediction| + |Ground Truth|)

Where:
  |·| = count of voxels with value 1
  ∩   = voxels that are 1 in BOTH masks
```

### Range
- 0.0 = no overlap (model found nothing, or found only wrong voxels)
- 1.0 = perfect overlap (every lesion voxel found, no false positives)

### Intuition
Think of two circles. Dice measures how much they overlap relative to their combined area. Two identical circles → Dice=1.0. Two circles that don't touch → Dice=0.0.

### What our scores mean

| Dice | Interpretation |
|---|---|
| < 0.10 | Trivial/collapsed model — predicting near-zero |
| 0.10–0.20 | Weak but real segmentation |
| 0.20–0.40 | Research-grade, partial detection |
| 0.40–0.65 | Clinical research tool |
| > 0.65 | Clinical deployment grade |

**Our result: 0.2326** — research grade, correctly identifies lesion regions but misses many small ones and has false positives.

---

## Metric 2: Lesion-wise True Positive Rate (TPR)

### What it measures
Whether each individual lesion was **detected at all** (regardless of how precisely).

Standard Dice treats all lesion voxels equally. A large lesion with 1000 voxels contributes 1000× more to Dice than a small lesion with 1 voxel. Clinical neurologists care about *lesion count* — a model that perfectly segments 3 large lesions and misses 10 small ones looks good by Dice but is clinically poor.

### How it's computed

1. Label each connected component (separate lesion) in the ground truth mask
2. For each labelled lesion: check if ANY predicted voxel overlaps with it
3. TPR = (number of detected lesions) / (total number of lesions)

```python
from scipy.ndimage import label as cc_label

def _lesion_tpr(pred_bin, gt_bin):
    gt_cc, n_gt = cc_label(gt_bin)  # label each separate lesion
    if n_gt == 0:
        return None, {}

    detected = 0
    for comp_id in range(1, n_gt + 1):
        lesion_mask = (gt_cc == comp_id)
        if (pred_bin[lesion_mask] > 0).any():  # any overlap = detected
            detected += 1

    return detected / n_gt
```

### Size stratification

Lesions are further stratified by size:
- **Small**: < 10 voxels (newly forming or very small plaques — hardest to detect)
- **Medium**: 10–100 voxels (active lesions — clinically important)
- **Large**: > 100 voxels (chronic plaques — easier to detect)

**Our result: TPR ~0.31** — the model detects ~31% of individual lesions at threshold=0.3. Most missed lesions are in the "small" category.

---

## Metric 3: Multi-Threshold Sweep

### The problem
Converting a probability map (values 0–1) to a binary mask requires choosing a threshold. Different thresholds trade off sensitivity vs specificity:
- Low threshold (0.1): detects more lesions but more false positives
- High threshold (0.5): fewer false positives but misses ambiguous lesions

There is no universal "correct" threshold — it depends on the dataset and training phase.

### Our approach
Evaluate at four thresholds every epoch and report the best:

```python
THRESHOLDS = [0.1, 0.2, 0.3, 0.5]

thresh_scores = {}
for t in THRESHOLDS:
    binary_pred = (probability_map > t).float()
    thresh_scores[t] = dice_metric(binary_pred, ground_truth)

best_threshold = max(thresh_scores, key=thresh_scores.get)
best_dice = thresh_scores[best_threshold]
```

The validation Dice reported each epoch is the best across all thresholds. This prevents the metric from being artificially low due to a suboptimal fixed threshold.

---

## Metric 4: Connected Component Filtering

### The problem
Probability maps near 0.5 threshold often produce "salt and pepper" noise — tiny isolated 1-voxel predictions scattered throughout the brain that are almost certainly false positives (real MS lesions are at least 3–8 voxels in volume).

### Our approach
After binarisation, remove any connected component smaller than 8 voxels:

```python
from scipy.ndimage import label as cc_label
import numpy as np

def _cc_filter(binary_mask, min_size=8):
    labelled, n = cc_label(binary_mask)
    if n == 0:
        return binary_mask
    sizes = np.bincount(labelled.ravel())  # count voxels per component
    small_component_ids = np.where(sizes < min_size)[0]
    small_component_ids = small_component_ids[small_component_ids > 0]
    if len(small_component_ids) > 0:
        binary_mask[np.isin(labelled, small_component_ids)] = 0
    return binary_mask
```

This is applied to every threshold evaluation, improving Dice by ~0.01–0.02 by removing spurious predictions.

---

## Summary: Metric Hierarchy

```
Level 1: Dice Score
  → Did we find the right voxels overall?
  → Primary quantitative metric for comparison

Level 2: Lesion-wise TPR @ threshold=0.3
  → Did we find each individual lesion?
  → Clinically more meaningful than voxel-level Dice

Level 3: Size-stratified TPR (small / medium / large)
  → Which lesion sizes are we detecting?
  → Reveals model's specific weakness (small lesions)

Level 4: Best threshold from sweep [0.1, 0.2, 0.3, 0.5]
  → Are we measuring at the right operating point?
  → Prevents metric artefacts from poor threshold choice
```

---

## Graph: Multi-Threshold Dice Over Training

```python
import matplotlib.pyplot as plt

# Example data structure (from a single validation run)
# In practice these would be read from training logs

epochs = list(range(1, 201))

# Approximate best threshold selection per epoch
# (from Telegram summaries, best threshold was mostly 0.1-0.2 in early training
#  and 0.2-0.3 in later training)

# To reproduce exactly: re-run validation on saved checkpoints
# Here we show the reported best Dice per epoch
best_dice_per_epoch = [d[2] for d in training_data]  # from 07_training_curves.md

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(epochs, best_dice_per_epoch, 'b-', linewidth=1.5)
ax.fill_between(epochs, 0, best_dice_per_epoch, alpha=0.1, color='blue')
ax.axhline(y=0.2326, color='red', linestyle='--', label=f'Best: 0.2326')
ax.set_xlabel('Epoch')
ax.set_ylabel('Best Dice (across thresholds)')
ax.set_title('Best Validation Dice Per Epoch (Auto-threshold)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('best_dice_per_epoch.png', dpi=150)
plt.show()
```
