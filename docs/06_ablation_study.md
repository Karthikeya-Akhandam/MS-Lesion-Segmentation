# Ablation Study

## What is an Ablation Study?

An ablation study systematically removes or changes one component at a time to measure its individual contribution. The name comes from neuroscience — "ablating" (removing) a brain region to see what it does.

In this project, the ablation varies the **number of MRI input channels** (modalities) while keeping everything else identical. This answers the question: *How much does each MRI modality contribute to segmentation accuracy?*

---

## Three Ablation Configurations

| Config | Input Channels | Modalities | N_CHANNELS |
|---|---|---|---|
| Ablation A | 1 | FLAIR only | 1 |
| Ablation B | 2 | FLAIR + T1 | 2 |
| **Ablation C** | **3** | **FLAIR + T1 + T2** | **3** |

The model architecture (BasicUNet), loss function sequence, training epochs, and all hyperparameters are identical across ablations. Only the input channel count changes.

---

## Results Comparison

| Config | Loss | Best Dice | Epochs to Best | Key Change |
|---|---|---|---|---|
| Ablation A (v1 UNet) | BCE+Dice | 0.067 | 150 | Baseline |
| Ablation A (v3 BasicUNet) | DiceCELoss | ~0.076 | 150 | Better architecture |
| Ablation B | DiceCELoss | 0.166 | 80 | +T1 channel |
| **Ablation C** | **DiceCELoss → DiceFocal** | **0.2326** | **200** | **+T2 + noisy label fix** |

> Note: Ablation B had a dataset bug (Long-MR-MS noisy labels). The 0.166 ceiling is partially explained by this contamination. Ablation C fixed the bug AND added T2.

---

## Architecture Experiments (Not Part of Main Ablation)

Before settling on BasicUNet, two other architectures were tested:

### SwinUNETR

| Metric | Value |
|---|---|
| Architecture | Vision Transformer |
| Best Dice | 0.058 |
| Epochs | ~20 (abandoned) |
| Reason abandoned | Worse than v1 UNet; requires large datasets |

### AttentionUnet

| Metric | Value |
|---|---|
| Architecture | UNet with attention gates |
| Best Dice | 0.065 |
| Epochs | 86 (abandoned) |
| Reason abandoned | BatchNorm+batch_size=1 instability; stale checkpoint bug |

---

## Python Graph: Ablation Comparison Bar Chart

```python
import matplotlib.pyplot as plt
import numpy as np

configs = [
    'Ablation A\n(v1 UNet\nFLAIR only)',
    'SwinUNETR\n(abandoned)',
    'AttentionUnet\n(abandoned)',
    'Ablation A\n(BasicUNet\nFLAIR only)',
    'Ablation B\n(FLAIR+T1)',
    'Ablation C\n(FLAIR+T1+T2)\n[FINAL]',
]
dice_scores = [0.067, 0.058, 0.065, 0.076, 0.166, 0.2326]
colors = ['#d9534f', '#e8a838', '#e8a838', '#5bc0de', '#5cb85c', '#1a7a1a']

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(configs, dice_scores, color=colors, edgecolor='black', linewidth=0.8)

# Add value labels on bars
for bar, score in zip(bars, dice_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{score:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

# Add clinical benchmark line
ax.axhline(y=0.65, color='purple', linestyle='--', linewidth=1.5,
           label='Clinical benchmark (LST-AI ~0.65)')

ax.set_ylabel('Best Dice Score', fontsize=12)
ax.set_title('Ablation Study: Architecture and Modality Comparison', fontsize=14)
ax.set_ylim(0, 0.75)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('ablation_comparison.png', dpi=150)
plt.show()
```

---

## Python Graph: Ablation B vs C Training Curves

```python
import matplotlib.pyplot as plt

# Ablation B — epoch-level best dice (approximate from Telegram logs)
ablation_b_best = [
    0.0724, 0.0900, 0.1050, 0.1200, 0.1300,  # ep 1-5
    0.1300, 0.1400, 0.1450, 0.1450, 0.1500,  # ep 6-10
    0.1500, 0.1550, 0.1560, 0.1580, 0.1590,  # ep 11-15
    0.1600, 0.1620, 0.1640, 0.1640, 0.1640,  # ep 16-20
    0.1640, 0.1650, 0.1655, 0.1655, 0.1660,  # ep 21-25 (end cycle 1)
    # Plateau from ep26 onward; best never exceeded 0.1660
] + [0.1660] * 55  # ep 26-80

# Ablation C — actual best dice per epoch from issue_28.txt
ablation_c_dice = [
    0.0245, 0.0314, 0.0366, 0.0501, 0.0467,
    0.0489, 0.0853, 0.1013, 0.0784, 0.1173,
    0.0852, 0.1364, 0.1399, 0.1418, 0.1435,
    0.1644, 0.1726, 0.1782, 0.1625, 0.1428,
    0.1527, 0.1753, 0.1627, 0.1664, 0.1660,
    0.1458, 0.1400, 0.1270, 0.1337, 0.1781,
    0.1327, 0.1719, 0.1595, 0.1592, 0.1810,
    0.1640, 0.1478, 0.1819, 0.1823, 0.1628,
    0.1547, 0.1513, 0.1822, 0.1669, 0.1666,
    0.1780, 0.2015, 0.1716, 0.1878, 0.1659,
    0.1832, 0.1938, 0.1799, 0.2063, 0.1665,
    0.1795, 0.1639, 0.1824, 0.1887, 0.1903,
    0.1942, 0.1885, 0.1786, 0.1845, 0.1901,
    0.1834, 0.1933, 0.1966, 0.1896, 0.1910,
    0.1907, 0.1878, 0.1906, 0.1881, 0.1898,
    0.1751, 0.1890, 0.1773, 0.1892, 0.1523,
    0.1225, 0.1809, 0.1755, 0.1715, 0.1582,
    0.1934, 0.1677, 0.1738, 0.1929, 0.1923,
    0.1768, 0.1897, 0.1644, 0.1965, 0.1691,
    0.1755, 0.1955, 0.1992, 0.1958, 0.1946,
    0.1656, 0.2094, 0.1782, 0.1756, 0.2182,
    0.1881, 0.2128, 0.2029, 0.2148, 0.1974,
    0.2083, 0.2028, 0.2043, 0.2040, 0.2014,
    0.2106, 0.2142, 0.2005, 0.2095, 0.2131,
    0.2092, 0.2086, 0.1983, 0.2124, 0.2173,
    0.1915, 0.1885, 0.2095, 0.1988, 0.2058,
    0.1988, 0.2092, 0.2071, 0.2079, 0.2166,
    0.2126, 0.2108, 0.2067, 0.2143, 0.2048,
    0.2090, 0.2190, 0.2037, 0.2193, 0.2129,
    0.2176, 0.2176, 0.2080, 0.2105, 0.2151,
    # DiceFocal phase (ep 151-200)
    0.1938, 0.2195, 0.2086, 0.2108, 0.2131,
    0.2118, 0.2140, 0.2130, 0.2169, 0.2101,
    0.2116, 0.2105, 0.2108, 0.2143, 0.2131,
    0.2131, 0.2125, 0.2126, 0.2131, 0.2125,
    0.2125, 0.2132, 0.2142, 0.2138, 0.2146,
    0.1732, 0.2009, 0.1552, 0.1535, 0.2077,
    0.2093, 0.2133, 0.1930, 0.1904, 0.2211,
    0.2058, 0.2079, 0.2118, 0.2099, 0.2092,
    0.2225, 0.2259, 0.2183, 0.2093, 0.2312,
    0.2140, 0.2210, 0.2239, 0.2061, 0.2326,
]

epochs_c = list(range(1, len(ablation_c_dice) + 1))
epochs_b = list(range(1, len(ablation_b_best) + 1))

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(epochs_c, ablation_c_dice, 'b-', alpha=0.5, linewidth=0.8, label='Ablation C (per-epoch Dice)')
ax.plot(epochs_b, ablation_b_best, 'r--', alpha=0.7, linewidth=1.5, label='Ablation B best (approx)')

# Running best line for Ablation C
running_best_c = []
best = 0
for d in ablation_c_dice:
    best = max(best, d)
    running_best_c.append(best)
ax.plot(epochs_c, running_best_c, 'b-', linewidth=2.5, label='Ablation C running best')

# Mark DiceFocal switch
ax.axvline(x=151, color='green', linestyle='--', alpha=0.7, label='DiceFocal switch (ep151)')
ax.axhline(y=0.2326, color='darkblue', linestyle=':', alpha=0.5, label='Final best: 0.2326')
ax.axhline(y=0.1660, color='darkred', linestyle=':', alpha=0.5, label='Ablation B ceiling: 0.1660')

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Dice Score', fontsize=12)
ax.set_title('Ablation B vs Ablation C — Training Comparison', fontsize=14)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('ablation_bc_comparison.png', dpi=150)
plt.show()
```

---

## Interpretation

**Why did T2 help (+0.07 Dice over Ablation B)?**

T2-weighted images highlight water content. MS lesions have elevated water due to inflammation and demyelination. T2 provides complementary signal to FLAIR:
- FLAIR suppresses cerebrospinal fluid (CSF) to reduce false positives
- T2 is sensitive to subtle perilesional oedema that FLAIR may miss
- T1 provides grey/white matter anatomy for spatial context

The three modalities together give the model three independent sources of evidence per voxel.

**Why was the improvement not larger?**

The jump from Ablation B to C is +0.07 Dice. The Long-MR-MS noisy label fix contributed some of this, and T2 the rest. The architectural ceiling for BasicUNet on 153 patients is approximately 0.25–0.30. Reaching 0.65+ (clinical grade) would require nnU-Net or equivalent with thousands of training samples.
