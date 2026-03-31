# Model Architecture

## What is a U-Net?

A U-Net is a neural network shaped like the letter "U". It has two halves:

- **Encoder (left side)**: Progressively shrinks the image while extracting features. Like zooming out to understand context.
- **Decoder (right side)**: Progressively expands back to the original size to produce a pixel-level prediction.
- **Skip connections**: Horizontal bridges that pass fine detail from encoder to decoder, preventing loss of spatial precision.

For 3D medical images, a 3D U-Net applies 3D convolutions (instead of 2D), processing the full volumetric brain scan.

---

## Architecture Progression

### v1 — Standard UNet (FAILED, Dice 0.067)

```python
monai.networks.nets.UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)
```

**Why it failed:**
- Narrow capacity (max 256 channels) — insufficient for complex 3D lesion features
- Loss function collapsed at epoch 45 (not architecture issue, but compounded failure)
- Single channel (FLAIR only) — limited input information

---

### SwinUNETR (TRIED AND ABANDONED, Dice 0.058)

A transformer-based architecture that treats 3D image patches as tokens, similar to how BERT processes words.

**Why it failed for this project:**
- Requires `einops` library not pre-installed
- MONAI 1.5.2 has a breaking API change (`img_size` parameter removed)
- Even after fixes: Dice 0.058 — worse than v1
- Transformers need large datasets (tens of thousands of patients) to outperform CNNs; our 153-patient dataset is too small

---

### AttentionUnet (TRIED AND ABANDONED, Dice 0.065)

AttentionUnet adds attention gates to skip connections, allowing the model to selectively focus on relevant regions.

**Why it failed:**
1. **BatchNorm with batch_size=1**: AttentionUnet uses BatchNorm internally. With batch_size=1, BatchNorm statistics are computed over a single sample — meaningless, and produces unstable training.
2. **Stale checkpoint contamination**: A bug in the resume logic (`elif os.path.exists(best_path)`) was loading weights from the previous architecture (different number of channels/layers) into the new model. This silently corrupted every experiment until the bug was found and removed.

---

### v3 — BasicUNet (CURRENT, Best Dice 0.2326)

```python
from monai.networks.nets import BasicUNet

BasicUNet(
    spatial_dims=3,
    in_channels=3,          # FLAIR + T1 + T2
    out_channels=1,         # binary lesion mask
    features=(32, 64, 128, 256, 512, 32),  # encoder→decoder channel widths
    act="LEAKYRELU",        # activation function
    norm="INSTANCE",        # normalisation type
    dropout=0.1,            # 10% dropout for regularisation
)
```

**Why BasicUNet works:**
- **Instance Normalization** (vs BatchNorm): Computed per-sample, works correctly with batch_size=1
- **4× more capacity** than v1 (max 512 channels vs 256)
- **LeakyReLU**: Prevents dead neurons by allowing small negative gradients
- **Proven for small datasets**: Pure CNN generalises well with < 200 training samples

---

## Architecture Diagram (ASCII)

```
Input: (B, 3, 96, 96, 96)
         │
    [Conv 32]──────────────────────────────────────────┐ skip
         │                                             │
    [Conv 64]──────────────────────────────────────┐  │ skip
         │                                         │  │
   [Conv 128]──────────────────────────────────┐   │  │ skip
         │                                     │   │  │
   [Conv 256]──────────────────────────────┐   │   │  │ skip
         │                                 │   │   │  │
   [Conv 512]  ← bottleneck               │   │   │  │
         │                                 │   │   │  │
  [UpConv 256] ←──────────────────────────┘   │   │  │
         │                                     │   │  │
  [UpConv 128] ←──────────────────────────────┘   │  │
         │                                         │  │
   [UpConv 64] ←──────────────────────────────────┘  │
         │                                            │
   [UpConv 32] ←─────────────────────────────────────┘
         │
   [Conv 1 → Sigmoid]
         │
Output: (B, 1, 96, 96, 96)  ← lesion probability map
```

---

## Bias Initialisation Trick

The final convolutional layer's bias is initialised to **-4.0**:

```python
# Applied after model creation
for m in reversed(list(model.modules())):
    if isinstance(m, nn.Conv3d) and m.bias is not None:
        nn.init.constant_(m.bias, -4.0)
        break
```

**Why this matters:**

`sigmoid(-4.0) ≈ 0.018`

This means at epoch 1, the model predicts ~1.8% of voxels as lesion. Since real lesion voxels are ~0.5–2% of total brain volume, this matches the true prior exactly. Without this:
- `bias=0` → sigmoid(0) = 0.5 → model starts predicting 50% positive → massive false positive rate → loss dominated by FP correction from epoch 1

This single trick prevented the "false positive collapse" that plagued v1.

---

## Parameter Count

| Architecture | Parameters | Notes |
|---|---|---|
| v1 UNet (16→256) | ~4.8M | Too narrow |
| BasicUNet (32→512) | ~19M | Current |
| AttentionUnet | ~27M | Too heavy for batch_size=1+BatchNorm |
| SwinUNETR | ~62M | Too large for 153 patients |

---

## Inference

At test time, the full brain volume (larger than 96³) is processed using **sliding window inference**:

```python
from monai.inferers import sliding_window_inference

pred = sliding_window_inference(
    inputs=val_img,
    roi_size=(96, 96, 96),
    sw_batch_size=4,
    predictor=model,
)
```

The volume is divided into overlapping 96³ patches, each patch is predicted independently, and predictions are stitched together with Gaussian blending to avoid boundary artefacts.
