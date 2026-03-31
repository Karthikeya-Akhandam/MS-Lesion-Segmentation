# Training Curves — Full 200 Epoch Data

All training data from the Ablation C final run (FLAIR+T1+T2, DiceCELoss ep1-150, DiceFocalLoss ep151-200).

---

## Raw Data

```python
# Complete epoch-by-epoch data from Ablation C (200 epochs)
# Format: (epoch, loss, dice, tpr)

training_data = [
    (1,   0.8545, 0.0245, 0.167),
    (2,   0.8007, 0.0314, 0.132),
    (3,   0.7708, 0.0366, 0.171),
    (4,   0.7599, 0.0501, 0.213),
    (5,   0.7443, 0.0467, 0.252),
    (6,   0.7359, 0.0489, 0.293),
    (7,   0.7313, 0.0853, 0.344),
    (8,   0.7159, 0.1013, 0.277),
    (9,   0.7029, 0.0784, 0.332),
    (10,  0.6924, 0.1173, 0.347),
    (11,  0.6849, 0.0852, 0.287),
    (12,  0.6768, 0.1364, 0.313),
    (13,  0.6697, 0.1399, 0.373),
    (14,  0.6673, 0.1418, 0.357),
    (15,  0.6636, 0.1435, 0.342),
    (16,  0.6549, 0.1644, 0.326),
    (17,  0.6459, 0.1726, 0.327),
    (18,  0.6423, 0.1782, 0.298),
    (19,  0.6452, 0.1625, 0.335),
    (20,  0.6391, 0.1428, 0.337),
    (21,  0.6347, 0.1527, 0.351),
    (22,  0.6367, 0.1753, 0.315),
    (23,  0.6383, 0.1627, 0.330),
    (24,  0.6267, 0.1664, 0.333),
    (25,  0.6277, 0.1660, 0.332),
    (26,  0.6583, 0.1458, 0.336),  # Cycle 2 restart — LR spike
    (27,  0.6709, 0.1400, 0.327),
    (28,  0.6560, 0.1270, 0.326),
    (29,  0.6450, 0.1337, 0.330),
    (30,  0.6427, 0.1781, 0.307),
    (31,  0.6333, 0.1327, 0.270),
    (32,  0.6335, 0.1719, 0.249),
    (33,  0.6440, 0.1595, 0.300),
    (34,  0.6235, 0.1592, 0.347),
    (35,  0.6299, 0.1810, 0.317),
    (36,  0.6225, 0.1640, 0.307),
    (37,  0.6208, 0.1478, 0.357),
    (38,  0.6098, 0.1819, 0.321),
    (39,  0.6172, 0.1823, 0.259),
    (40,  0.6087, 0.1628, 0.358),
    (41,  0.5978, 0.1547, 0.331),
    (42,  0.5990, 0.1513, 0.342),
    (43,  0.6036, 0.1822, 0.330),
    (44,  0.6100, 0.1669, 0.374),
    (45,  0.6019, 0.1666, 0.313),
    (46,  0.5908, 0.1780, 0.291),
    (47,  0.5939, 0.2015, 0.269),  # First 0.20+ milestone
    (48,  0.5994, 0.1716, 0.338),
    (49,  0.5829, 0.1878, 0.330),
    (50,  0.5882, 0.1659, 0.343),
    (51,  0.5847, 0.1832, 0.306),
    (52,  0.5867, 0.1938, 0.349),
    (53,  0.5759, 0.1799, 0.347),
    (54,  0.5690, 0.2063, 0.298),  # New best
    (55,  0.5760, 0.1665, 0.326),
    (56,  0.5741, 0.1795, 0.350),
    (57,  0.5717, 0.1639, 0.342),
    (58,  0.5641, 0.1824, 0.322),
    (59,  0.5679, 0.1887, 0.311),
    (60,  0.5583, 0.1903, 0.322),
    (61,  0.5652, 0.1942, 0.322),
    (62,  0.5610, 0.1885, 0.324),
    (63,  0.5561, 0.1786, 0.327),
    (64,  0.5647, 0.1845, 0.313),
    (65,  0.5565, 0.1901, 0.290),
    (66,  0.5536, 0.1834, 0.327),
    (67,  0.5546, 0.1933, 0.322),
    (68,  0.5534, 0.1966, 0.302),
    (69,  0.5539, 0.1896, 0.314),
    (70,  0.5505, 0.1910, 0.306),
    (71,  0.5544, 0.1907, 0.318),
    (72,  0.5525, 0.1878, 0.312),
    (73,  0.5572, 0.1906, 0.313),
    (74,  0.5510, 0.1881, 0.319),
    (75,  0.5532, 0.1898, 0.316),
    (76,  0.5783, 0.1751, 0.271),  # Cycle 3 restart — LR spike
    (77,  0.5862, 0.1890, 0.335),
    (78,  0.5802, 0.1773, 0.281),
    (79,  0.5900, 0.1892, 0.356),
    (80,  0.5852, 0.1523, 0.371),
    (81,  0.5819, 0.1225, 0.269),
    (82,  0.5851, 0.1809, 0.322),
    (83,  0.5785, 0.1755, 0.290),
    (84,  0.5755, 0.1715, 0.340),
    (85,  0.5867, 0.1582, 0.421),
    (86,  0.5773, 0.1934, 0.329),
    (87,  0.5808, 0.1677, 0.328),
    (88,  0.5782, 0.1738, 0.406),
    (89,  0.5860, 0.1929, 0.376),
    (90,  0.5879, 0.1923, 0.350),
    (91,  0.5833, 0.1768, 0.383),
    (92,  0.5875, 0.1897, 0.349),
    (93,  0.5887, 0.1644, 0.379),
    (94,  0.5829, 0.1965, 0.261),
    (95,  0.5846, 0.1691, 0.429),
    (96,  0.5916, 0.1755, 0.396),
    (97,  0.5874, 0.1955, 0.319),
    (98,  0.5856, 0.1992, 0.246),
    (99,  0.5879, 0.1958, 0.335),
    (100, 0.5796, 0.1946, 0.320),
    (101, 0.5763, 0.1656, 0.355),
    (102, 0.5843, 0.2094, 0.373),  # New best — cycle 3 recovery
    (103, 0.5813, 0.1782, 0.372),
    (104, 0.5811, 0.1756, 0.345),
    (105, 0.5824, 0.2182, 0.339),  # New best
    (106, 0.5742, 0.1881, 0.347),
    (107, 0.5783, 0.2128, 0.291),
    (108, 0.5795, 0.2029, 0.346),
    (109, 0.5706, 0.2148, 0.278),
    (110, 0.5703, 0.1974, 0.343),
    (111, 0.5678, 0.2083, 0.285),
    (112, 0.5626, 0.2028, 0.296),
    (113, 0.5724, 0.2043, 0.361),
    (114, 0.5749, 0.2040, 0.309),
    (115, 0.5615, 0.2014, 0.298),
    (116, 0.5600, 0.2106, 0.300),
    (117, 0.5658, 0.2142, 0.327),
    (118, 0.5621, 0.2005, 0.314),
    (119, 0.5645, 0.2095, 0.356),
    (120, 0.5656, 0.2131, 0.316),
    (121, 0.5610, 0.2092, 0.311),
    (122, 0.5560, 0.2086, 0.304),
    (123, 0.5519, 0.1983, 0.328),
    (124, 0.5501, 0.2124, 0.307),
    (125, 0.5567, 0.2173, 0.302),
    (126, 0.5534, 0.1915, 0.247),
    (127, 0.5507, 0.1885, 0.331),
    (128, 0.5566, 0.2095, 0.310),
    (129, 0.5465, 0.1988, 0.297),
    (130, 0.5541, 0.2058, 0.307),
    (131, 0.5490, 0.1988, 0.320),
    (132, 0.5478, 0.2092, 0.316),
    (133, 0.5385, 0.2071, 0.325),
    (134, 0.5416, 0.2079, 0.321),
    (135, 0.5402, 0.2166, 0.308),
    (136, 0.5423, 0.2126, 0.310),
    (137, 0.5447, 0.2108, 0.321),
    (138, 0.5411, 0.2067, 0.267),
    (139, 0.5372, 0.2143, 0.305),
    (140, 0.5320, 0.2048, 0.320),
    (141, 0.5347, 0.2090, 0.337),
    (142, 0.5348, 0.2190, 0.289),
    (143, 0.5340, 0.2037, 0.317),
    (144, 0.5312, 0.2193, 0.323),  # DiceCELoss best
    (145, 0.5292, 0.2129, 0.274),
    (146, 0.5332, 0.2176, 0.309),
    (147, 0.5315, 0.2176, 0.295),
    (148, 0.5289, 0.2080, 0.313),
    (149, 0.5321, 0.2105, 0.310),
    (150, 0.5249, 0.2151, 0.308),
    # --- DiceFocalLoss phase starts ---
    (151, 0.5260, 0.1938, 0.322),
    (152, 0.5262, 0.2195, 0.274),  # New best (DiceFocal)
    (153, 0.5140, 0.2086, 0.281),
    (154, 0.5117, 0.2108, 0.305),
    (155, 0.5110, 0.2131, 0.268),
    (156, 0.5097, 0.2118, 0.295),
    (157, 0.5089, 0.2140, 0.274),
    (158, 0.5095, 0.2130, 0.272),
    (159, 0.5050, 0.2169, 0.304),
    (160, 0.5038, 0.2101, 0.269),
    (161, 0.5058, 0.2116, 0.269),
    (162, 0.5027, 0.2105, 0.271),
    (163, 0.5033, 0.2108, 0.271),
    (164, 0.4994, 0.2143, 0.291),
    (165, 0.5024, 0.2131, 0.271),
    (166, 0.5076, 0.2131, 0.268),
    (167, 0.5032, 0.2125, 0.269),
    (168, 0.4965, 0.2126, 0.268),
    (169, 0.5035, 0.2131, 0.270),
    (170, 0.5027, 0.2125, 0.271),
    (171, 0.5005, 0.2125, 0.272),
    (172, 0.4959, 0.2132, 0.271),
    (173, 0.4957, 0.2142, 0.271),
    (174, 0.4939, 0.2138, 0.268),
    (175, 0.4953, 0.2146, 0.268),
    (176, 0.5449, 0.1732, 0.283),  # Cycle 4 restart — LR spike
    (177, 0.5618, 0.2009, 0.272),
    (178, 0.5654, 0.1552, 0.286),
    (179, 0.5574, 0.1535, 0.358),
    (180, 0.5483, 0.2077, 0.303),
    (181, 0.5382, 0.2093, 0.267),
    (182, 0.5407, 0.2133, 0.324),
    (183, 0.5312, 0.1930, 0.324),
    (184, 0.5447, 0.1904, 0.328),
    (185, 0.5376, 0.2211, 0.280),  # New best
    (186, 0.5206, 0.2058, 0.322),
    (187, 0.5316, 0.2079, 0.325),
    (188, 0.5239, 0.2118, 0.260),
    (189, 0.5248, 0.2099, 0.266),
    (190, 0.5283, 0.2092, 0.344),
    (191, 0.5330, 0.2225, 0.283),  # New best
    (192, 0.5188, 0.2259, 0.330),  # New best
    (193, 0.5097, 0.2183, 0.306),
    (194, 0.5178, 0.2093, 0.308),
    (195, 0.5128, 0.2312, 0.326),  # New best
    (196, 0.5199, 0.2140, 0.286),
    (197, 0.5110, 0.2210, 0.276),
    (198, 0.5132, 0.2239, 0.337),
    (199, 0.5302, 0.2061, 0.279),
    (200, 0.5082, 0.2326, 0.312),  # FINAL BEST
]
```

---

## Graph 1: Loss Curve

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

epochs = [d[0] for d in training_data]
losses = [d[1] for d in training_data]

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(epochs, losses, 'r-', linewidth=1.5, label='Training Loss')

# Shade DiceFocal phase
ax.axvspan(151, 200, alpha=0.1, color='green', label='DiceFocalLoss phase')

# Mark cycle restarts
for ep, label in [(26, 'C2'), (76, 'C3'), (151, 'DiceFocal\n+LR=5e-5'), (176, 'C4')]:
    ax.axvline(x=ep, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(ep+1, max(losses)*0.97, label, fontsize=7, color='gray')

ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Loss — Ablation C (200 Epochs)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=150)
plt.show()
```

---

## Graph 2: Dice Score + Running Best

```python
import matplotlib.pyplot as plt
import numpy as np

epochs = [d[0] for d in training_data]
dices  = [d[2] for d in training_data]

# Compute running best
running_best = []
best = 0
for d in dices:
    best = max(best, d)
    running_best.append(best)

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(epochs, dices, 'b-', alpha=0.3, linewidth=0.8, label='Per-epoch Dice')
ax.plot(epochs, running_best, 'b-', linewidth=2.5, label='Running Best Dice')

# Shade DiceFocal phase
ax.axvspan(151, 200, alpha=0.1, color='green', label='DiceFocalLoss phase')

# Mark key milestones
milestones = [
    (47,  0.2015, 'First 0.20+'),
    (54,  0.2063, 'Plateau begins'),
    (105, 0.2182, 'Cycle 3\nrecovery'),
    (144, 0.2193, 'DiceCELoss\nbest'),
    (200, 0.2326, 'Final best\n0.2326'),
]
for ep, dice, label in milestones:
    ax.annotate(label, xy=(ep, dice), xytext=(ep+3, dice+0.01),
                fontsize=8, arrowprops=dict(arrowstyle='->', color='black'))

# Cycle restarts
for ep in [26, 76, 151, 176]:
    ax.axvline(x=ep, color='gray', linestyle='--', alpha=0.4, linewidth=1)

ax.set_xlabel('Epoch')
ax.set_ylabel('Dice Score')
ax.set_title('Validation Dice — Ablation C (200 Epochs)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('dice_curve.png', dpi=150)
plt.show()
```

---

## Graph 3: TPR Over Epochs

```python
import matplotlib.pyplot as plt

epochs = [d[0] for d in training_data]
tprs   = [d[3] for d in training_data]

# Smooth with 10-epoch rolling average
import numpy as np
tpr_smooth = np.convolve(tprs, np.ones(10)/10, mode='valid')
ep_smooth  = epochs[9:]

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(epochs, tprs, 'g-', alpha=0.3, linewidth=0.8, label='Per-epoch TPR@0.3')
ax.plot(ep_smooth, tpr_smooth, 'g-', linewidth=2, label='10-epoch rolling average')
ax.axvspan(151, 200, alpha=0.1, color='green')
ax.set_xlabel('Epoch')
ax.set_ylabel('TPR (Lesion Detection Rate)')
ax.set_title('Lesion True Positive Rate — Ablation C')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('tpr_curve.png', dpi=150)
plt.show()
```

---

## Graph 4: Loss vs Dice Correlation

```python
import matplotlib.pyplot as plt

losses = [d[1] for d in training_data]
dices  = [d[2] for d in training_data]
epochs = [d[0] for d in training_data]

# Colour by phase
colors = ['blue' if e <= 150 else 'green' for e in epochs]

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(losses, dices, c=colors, alpha=0.4, s=20)

import matplotlib.patches as mpatches
blue_patch  = mpatches.Patch(color='blue',  label='DiceCELoss phase (ep1-150)')
green_patch = mpatches.Patch(color='green', label='DiceFocalLoss phase (ep151-200)')
ax.legend(handles=[blue_patch, green_patch])

ax.set_xlabel('Training Loss')
ax.set_ylabel('Validation Dice')
ax.set_title('Loss vs Dice Correlation')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('loss_vs_dice.png', dpi=150)
plt.show()
```

---

## Key Milestones Table

| Epoch | Dice | Event |
|---|---|---|
| 1 | 0.0245 | Training start (bias=-4.0 init) |
| 7 | 0.0853 | First meaningful segmentation |
| 17 | 0.1726 | Exceeded Ablation B's best (0.1660) |
| 47 | 0.2015 | First time crossing 0.20 |
| 54 | 0.2063 | Cycle 2 peak — 46-epoch plateau begins |
| 76 | — | Cycle 3 restart (LR spike, Dice dips) |
| 105 | 0.2182 | Cycle 3 recovery — new best |
| 144 | 0.2193 | DiceCELoss all-time best |
| 151 | — | DiceFocalLoss + LR=5e-5 activated |
| 176 | — | Cycle 4 restart (LR spike again) |
| 185 | 0.2211 | DiceFocal new best |
| 195 | 0.2312 | Rapid improvement resumes |
| **200** | **0.2326** | **FINAL BEST** |
