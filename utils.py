"""
utils.py — Shared MS Lesion Segmentation Engine
Imported by Project_Demo.ipynb (Gradio) and app.py (Streamlit).
"""
import uuid, os
import torch
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from monai.networks.nets import BasicUNet
from monai.inferers import sliding_window_inference
from scipy.ndimage import label as cc_label

# ── Device (auto-detected; notebook/app can override) ───────────────────────
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_NAME = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"

# ── Constants ────────────────────────────────────────────────────────────────
PATCH_SIZE  = (96, 96, 96)
OVERLAY_DIR = "./demo_outputs"
os.makedirs(OVERLAY_DIR, exist_ok=True)

# ── Model placeholder — set by notebook/app after loading weights ────────────
model = None


# ── Model definition ─────────────────────────────────────────────────────────
def get_model():
    return BasicUNet(
        spatial_dims=3, in_channels=1, out_channels=1,
        features=(32, 64, 128, 256, 512, 32),
        act="LEAKYRELU", norm="INSTANCE", dropout=0.1,
    )


# ── Preprocessing ─────────────────────────────────────────────────────────────
def apply_skull_strip(sitk_img):
    otsu        = sitk.OtsuThreshold(sitk_img, 0, 1)
    filled      = sitk.BinaryFillhole(otsu)
    closed      = sitk.BinaryMorphologicalClosing(filled, [3, 3, 3])
    labeled     = sitk.ConnectedComponent(closed)
    sorted_comp = sitk.RelabelComponent(labeled, sortByObjectSize=True)
    brain_mask  = sitk.Equal(sorted_comp, 1)
    mask_float  = sitk.Cast(brain_mask, sitk.sitkFloat32)
    return sitk.Multiply(sitk.Cast(sitk_img, sitk.sitkFloat32), mask_float)


def preprocess_for_demo(img_path):
    raw  = sitk.ReadImage(img_path)
    mask = sitk.OtsuThreshold(raw, 0, 1)
    cor  = sitk.N4BiasFieldCorrectionImageFilter()
    cor.SetMaximumNumberOfIterations([20, 20, 20])
    corrected = cor.Execute(raw, mask)
    stripped  = apply_skull_strip(corrected)
    img       = sitk.DICOMOrient(stripped, "RAS")
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing((1.0, 1.0, 1.0))
    orig_size, orig_spc = img.GetSize(), img.GetSpacing()
    new_size = [int(round(osz * ospc)) for osz, ospc in zip(orig_size, orig_spc)]
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    img       = resampler.Execute(img)
    arr       = sitk.GetArrayFromImage(img)
    brain_vox = arr[arr > 0]
    if brain_vox.size > 0:
        arr = (arr - brain_vox.mean()) / (brain_vox.std() + 1e-8)
    return arr


# ── Inference ─────────────────────────────────────────────────────────────────
def remove_fp_noise(mask, min_size=5):
    labeled, n = cc_label(mask)
    cleaned = np.zeros_like(mask, dtype=bool)
    for comp_id in range(1, n + 1):
        if (labeled == comp_id).sum() >= min_size:
            cleaned[labeled == comp_id] = True
    return cleaned


def segment_volume(arr):
    t = torch.tensor(arr).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        if device.type == "cuda":
            with torch.amp.autocast('cuda'):
                prob = sliding_window_inference(t, PATCH_SIZE, 4, model)
        else:
            prob = sliding_window_inference(t, PATCH_SIZE, 4, model)
    prob_np = prob.sigmoid().cpu().numpy().squeeze()
    mask    = (prob_np > 0.5).astype(bool)
    return remove_fp_noise(mask, min_size=5), prob_np


# ── Metrics ───────────────────────────────────────────────────────────────────
def run_lesion_stats(mask):
    labeled, n = cc_label(mask)
    vol    = int(mask.sum())
    sizes  = {"small": 0, "medium": 0, "large": 0}
    for comp_id in range(1, n + 1):
        sz = int((labeled == comp_id).sum())
        if sz < 10:       sizes["small"]  += 1
        elif sz <= 100:   sizes["medium"] += 1
        else:             sizes["large"]  += 1
    return vol, n, sizes


def compute_confidence_metrics(mask, prob_np):
    if mask.sum() == 0:
        return {"mean_confidence": 0.0, "brain_coverage_pct": 0.0, "high_conf_pct": 0.0}
    lesion_probs = prob_np[mask]
    brain_voxels = (prob_np > 0.01).sum()
    return {
        "mean_confidence":    float(lesion_probs.mean()),
        "brain_coverage_pct": float(mask.sum() / max(brain_voxels, 1) * 100),
        "high_conf_pct":      float((lesion_probs > 0.8).mean() * 100),
    }


def compute_accuracy_metrics(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt   = gt_mask.astype(bool)
    tp   = (pred & gt).sum()
    fp   = (pred & ~gt).sum()
    fn   = (~pred & gt).sum()
    tn   = (~pred & ~gt).sum()
    return {
        "dice":        float(2*tp / (2*tp + fp + fn + 1e-8)),
        "sensitivity": float(tp / (tp + fn + 1e-8)),
        "precision":   float(tp / (tp + fp + 1e-8)),
        "specificity": float(tn / (tn + fp + 1e-8)),
    }


# ── Visualisation ─────────────────────────────────────────────────────────────
def create_mosaic(img_vol, mask_vol):
    D, H, W = img_vol.shape
    views = [
        (img_vol[D//2, :, :], mask_vol[D//2, :, :], "Axial"),
        (img_vol[:, H//2, :], mask_vol[:, H//2, :], "Coronal"),
        (img_vol[:, :, W//2], mask_vol[:, :, W//2], "Sagittal"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='#0d1117')
    for ax, (img_sl, mask_sl, title) in zip(axes, views):
        ax.imshow(img_sl, cmap='gray', origin='lower')
        ax.imshow(np.ma.masked_where(mask_sl == 0, mask_sl), cmap='Reds', alpha=0.6, origin='lower')
        ax.set_title(title, color='white', fontsize=13, fontweight='bold')
        ax.axis('off')
    fig.tight_layout(pad=0.5)
    out = os.path.join(OVERLAY_DIR, f"mosaic_{uuid.uuid4().hex[:8]}.png")
    plt.savefig(out, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    return out


def create_change_mosaic(img_vol, new_mask, resolved_mask):
    D, H, W = img_vol.shape
    views = [
        (img_vol[D//2, :, :], new_mask[D//2, :, :], resolved_mask[D//2, :, :], "Axial"),
        (img_vol[:, H//2, :], new_mask[:, H//2, :], resolved_mask[:, H//2, :], "Coronal"),
        (img_vol[:, :, W//2], new_mask[:, :, W//2], resolved_mask[:, :, W//2], "Sagittal"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='#0d1117')
    for ax, (img_sl, new_sl, res_sl, title) in zip(axes, views):
        ax.imshow(img_sl, cmap='gray', origin='lower')
        ax.imshow(np.ma.masked_where(new_sl == 0, new_sl),  cmap='Greens', alpha=0.7, origin='lower')
        ax.imshow(np.ma.masked_where(res_sl == 0, res_sl),  cmap='Blues',  alpha=0.7, origin='lower')
        ax.set_title(title, color='white', fontsize=13, fontweight='bold')
        ax.axis('off')
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color='lime', label='New lesions'),
                        Patch(color='cornflowerblue', label='Resolved lesions')],
               loc='lower center', ncol=2, framealpha=0, labelcolor='white', fontsize=11)
    fig.tight_layout(pad=0.5)
    out = os.path.join(OVERLAY_DIR, f"change_{uuid.uuid4().hex[:8]}.png")
    plt.savefig(out, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    return out


# ── High-level analysis functions ─────────────────────────────────────────────
def run_ai_analysis(file_obj, gt_file_obj=None):
    if file_obj is None:
        return None, "Please upload a NIfTI file (.nii or .nii.gz)."
    arr            = preprocess_for_demo(file_obj.name)
    mask, prob_np  = segment_volume(arr)
    vol, count, sizes = run_lesion_stats(mask)
    conf           = compute_confidence_metrics(mask, prob_np)

    acc_section = ""
    if gt_file_obj is not None:
        import nibabel as nib
        from skimage.transform import resize as sk_resize
        gt_arr = (nib.load(gt_file_obj.name).get_fdata() > 0.5).astype(bool)
        if gt_arr.shape != mask.shape:
            gt_arr = sk_resize(gt_arr, mask.shape, order=0, anti_aliasing=False).astype(bool)
        acc = compute_accuracy_metrics(mask, gt_arr)
        acc_section = f"""
### Segmentation Accuracy (vs Ground Truth)
| Metric | Score |
|--------|-------|
| **Dice Score** | **{acc['dice']:.3f}** |
| Sensitivity (Recall) | {acc['sensitivity']:.3f} |
| Precision | {acc['precision']:.3f} |
| Specificity | {acc['specificity']:.3f} |
"""

    report = f"""
## AI Analysis Report
| Metric | Value |
|--------|-------|
| Total Lesion Volume | **{vol:.0f} mm³** |
| Total Lesion Count  | **{count}** |
| Small (<10 vox)     | {sizes['small']} |
| Medium (10-100 vox) | {sizes['medium']} |
| Large (>100 vox)    | {sizes['large']} |

### Model Confidence
| Metric | Value |
|--------|-------|
| Mean Lesion Confidence | {conf['mean_confidence']:.1%} |
| High-Confidence Voxels (>80%) | {conf['high_conf_pct']:.1f}% of lesion |
| Brain Coverage | {conf['brain_coverage_pct']:.3f}% |
{acc_section}
*Processed on {DEVICE_NAME}*
"""
    return create_mosaic(arr, mask), report


def run_longitudinal_analysis(t0_file, t1_file):
    if t0_file is None or t1_file is None:
        return None, None, "Please upload both Baseline (T0) and Follow-up (T1) scans."

    arr_t0          = preprocess_for_demo(t0_file.name)
    arr_t1          = preprocess_for_demo(t1_file.name)
    mask_t0, _      = segment_volume(arr_t0)
    mask_t1, _      = segment_volume(arr_t1)

    fixed  = sitk.GetImageFromArray(arr_t0.astype(np.float32))
    moving = sitk.GetImageFromArray(arr_t1.astype(np.float32))
    reg    = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
    reg.SetInitialTransform(sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform()))
    reg.SetInterpolator(sitk.sitkLinear)
    tx = reg.Execute(fixed, moving)

    mask_t1_sitk     = sitk.GetImageFromArray(mask_t1.astype(np.float32))
    mask_t1_sitk.CopyInformation(moving)
    mask_t1_reg      = sitk.GetArrayFromImage(
        sitk.Resample(mask_t1_sitk, fixed, tx, sitk.sitkNearestNeighbor, 0.0)
    ).astype(bool)

    new_lesions      = mask_t1_reg & ~mask_t0
    resolved_lesions = mask_t0 & ~mask_t1_reg

    vol_t0, cnt_t0, _ = run_lesion_stats(mask_t0)
    vol_t1, cnt_t1, _ = run_lesion_stats(mask_t1_reg)
    _, new_cnt, _      = run_lesion_stats(new_lesions)
    _, res_cnt, _      = run_lesion_stats(resolved_lesions)

    report  = "## Longitudinal Progression Report\n\n"
    report += "| Metric | Baseline (T0) | Follow-up (T1) | Change |\n|---|---|---|---|\n"
    report += f"| Lesion Volume | {vol_t0} mm³ | {vol_t1} mm³ | {vol_t1 - vol_t0:+d} mm³ |\n"
    report += f"| Lesion Count  | {cnt_t0} | {cnt_t1} | {cnt_t1 - cnt_t0:+d} |\n"
    report += f"\n**New lesions detected:** {new_cnt} &nbsp;&nbsp; **Resolved lesions:** {res_cnt}\n"
    report += f"\n*Registration + segmentation on {DEVICE_NAME}*"

    return create_mosaic(arr_t0, mask_t0), create_change_mosaic(arr_t0, new_lesions, resolved_lesions), report
