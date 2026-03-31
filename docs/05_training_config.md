# Training Configuration

## Final Hyperparameters

| Parameter | Value | Justification |
|---|---|---|
| EPOCHS | 200 (150 DiceCELoss + 50 DiceFocalLoss) | Full cosine cycle 3 + partial cycle 4 |
| BATCH_SIZE | 1 | VRAM constraint with 96³ 3D patches |
| PATCH_SIZE | (96, 96, 96) | Standard for 3D medical segmentation; fits in VRAM |
| LR (phase 1) | 1e-4 | AdamW standard for medical imaging |
| LR (phase 2) | 5e-5 | Reduced for fine-tuning to prevent DiceFocal instability |
| GRAD_ACCUM | 8 | Effective batch = 8; simulates larger batch without OOM |
| WEIGHT_DECAY | 1e-5 | L2 regularisation |
| THRESHOLDS | [0.1, 0.2, 0.3, 0.5] | Multi-threshold sweep to find best binarisation per epoch |

---

## Optimizer: AdamW

```python
from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
```

**AdamW** = Adam with decoupled weight decay. It adapts learning rates per parameter (good for sparse gradients like lesion signals) and applies weight decay correctly (unlike Adam which conflates it with gradient scaling).

---

## Learning Rate Scheduler: Cosine Annealing with Warm Restarts

```python
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=2, eta_min=1e-6)
```

### What this does

The learning rate follows a cosine curve: starts at max (1e-4), smoothly decays to min (1e-6), then restarts to max. Each restart the cycle length doubles (T_mult=2):

```
Cycle 1: Epochs   1– 25  (25 epochs)   LR: 1e-4 → 1e-6
Cycle 2: Epochs  26– 75  (50 epochs)   LR: 1e-4 → 1e-6
Cycle 3: Epochs  76–175  (100 epochs)  LR: 1e-4 → 1e-6
Cycle 4: Epochs 176–375  (200 epochs)  LR: 5e-5 → 1e-6  ← fine-tuning phase
```

### Why warm restarts help

At each restart, the LR jumps back to the maximum. This "shakes" the model out of a local minimum and allows it to explore new weight configurations. The longer cycles allow progressively deeper convergence.

In practice, new best Dice scores were often achieved just before each restart (when LR is lowest and updates are most precise):
- Cycle 2 best: Dice 0.2063 (epoch 54)
- Cycle 3 best: Dice 0.2193 (epoch 144)
- Cycle 4 best: Dice 0.2326 (epoch 200)

### Visualisation of LR schedule

```python
import matplotlib.pyplot as plt
import numpy as np

def cosine_annealing_lr(epoch, T_0=25, T_mult=2, lr_max=1e-4, lr_min=1e-6):
    """Compute LR for given epoch using CosineAnnealingWarmRestarts."""
    t = epoch
    T_cur = T_0
    while t >= T_cur:
        t -= T_cur
        T_cur = int(T_cur * T_mult)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * t / T_cur))

epochs = np.arange(1, 201)
lr_phase1 = [cosine_annealing_lr(e, lr_max=1e-4) for e in epochs[:150]]
lr_phase2 = [cosine_annealing_lr(e, lr_max=5e-5) for e in epochs[150:]]
lrs = lr_phase1 + lr_phase2

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(epochs, lrs, 'b-', linewidth=1.5)
ax.axvline(x=26, color='gray', linestyle='--', alpha=0.5, label='Cycle restart')
ax.axvline(x=76, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=151, color='red', linestyle='--', alpha=0.7, label='DiceFocal switch (LR→5e-5)')
ax.axvline(x=176, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Learning Rate')
ax.set_title('Learning Rate Schedule — CosineAnnealingWarmRestarts')
ax.legend()
ax.set_yscale('log')
plt.tight_layout()
plt.savefig('lr_schedule.png', dpi=150)
plt.show()
```

---

## Patch Sampling Strategy

```python
RandCropByPosNegLabeld(
    keys=["image", "label"],
    label_key="label",
    spatial_size=(96, 96, 96),
    pos=3,
    neg=1,
    num_samples=4,
)
```

**What this does**: For each training volume, extracts 4 random 96³ patches. The `pos=3, neg=1` ratio means:
- 3 patches are centred on a lesion voxel (guaranteed to contain at least one lesion)
- 1 patch is randomly placed (may or may not contain lesion)

**Why**: Without this, random patches would be ~99% background. The model would never see enough lesion examples to learn. The 75% lesion-patch guarantee ensures the model sees pathology every step.

---

## Gradient Accumulation

```python
GRAD_ACCUM = 8

# In training loop:
loss = loss_fn(model(images), labels) / GRAD_ACCUM
loss.backward()
if (step + 1) % GRAD_ACCUM == 0:
    optimizer.step()
    optimizer.zero_grad()
```

**What this does**: Instead of updating weights every step, accumulate gradients for 8 steps before updating. This simulates a batch size of 8 (1 sample × 8 accumulation steps) while only keeping 1 sample in VRAM at once.

**Why**: 3D volumes at 96³ are large. Batch size > 1 causes out-of-memory errors. Gradient accumulation gives the benefits of larger batches without the memory cost.

---

## Mixed Precision Training (AMP)

```python
scaler = torch.amp.GradScaler('cuda')

with torch.amp.autocast('cuda'):
    loss = loss_fn(model(images), labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

Uses FP16 (16-bit floating point) for forward pass and FP32 for gradient accumulation. This:
- Halves VRAM usage for activations
- Speeds up GPU operations (~2× throughput on RTX 4500)
- GradScaler prevents FP16 underflow (very small gradients becoming zero)

---

## Data Augmentation

Applied only during training (not validation):

| Transform | Parameters | Purpose |
|---|---|---|
| RandFlipd | prob=0.5, axis=0 | Left-right brain symmetry |
| RandRotate90d | prob=0.5 | Orientation invariance |
| RandGaussianNoised | prob=0.2, std=0.1 | Scanner noise simulation |
| RandScaleIntensityd | factors=0.2, prob=0.3 | Scanner intensity variation |
| RandShiftIntensityd | offsets=0.1, prob=0.3 | Brightness variation |

---

## Checkpoint System (Top-3)

Three checkpoints are maintained at all times:
- `best_model.pth` — all-time best Dice
- `top2_{ep}_{dice}.pth` — second best
- `top3_{ep}_{dice}.pth` — third best

On each new best Dice:
1. Delete top3
2. Rename top2 → top3
3. Copy current best → top2
4. Save new best → best_model.pth

This prevents model loss if a single checkpoint file is corrupted or overwritten.

---

## Resume from Checkpoint

```python
RESUME_CHECKPOINT = True  # set to True to resume

if RESUME_CHECKPOINT:
    ckpt = torch.load('models/checkpoint_latest.pth')
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    for pg in optimizer.param_groups:
        pg["lr"] = LR  # override to new LR if fine-tuning
    best_dice   = ckpt["best_dice"]
    start_epoch = ckpt["epoch"] + 1
    scheduler.load_state_dict(ckpt["scheduler"])
```

`checkpoint_latest.pth` is saved every epoch (asynchronously in a background thread to avoid slowing training).
