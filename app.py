"""
app.py — Streamlit UI for MS Lesion Segmentation
All inference/preprocessing logic lives in utils.py.
Run with: streamlit run app.py
"""
import os
import torch
import streamlit as st
import utils
from utils import (
    get_model, run_ai_analysis, run_longitudinal_analysis,
    DEVICE_NAME,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MS Lesion AI",
    page_icon="🧠",
    layout="wide",
    menu_items={
        "About": """
        **🧠 MS Lesion Segmentation Dashboard**

        Developed by **Karthikeya Akhandam**

        🔗 [GitHub](https://github.com/Karthikeya-Akhandam)  
        💼 [LinkedIn](https://linkedin.com/in/karthikeyaakhandam)
        """
    }
)

# ── Model loading (cached so it only runs once) ───────────────────────────────
@st.cache_resource
def load_model():
    try:
        from google.colab import drive
        model_path = "/content/drive/MyDrive/MS_Project/models/best_model.pth"
    except ImportError:
        model_path = "./models/best_model.pth"

    m = get_model().to(utils.device)
    if os.path.exists(model_path):
        m.load_state_dict(torch.load(model_path, map_location=utils.device))
        m.eval()
        st.sidebar.success(f"Model loaded from `{model_path}`")
    else:
        st.sidebar.error(f"Model not found at `{model_path}`. Run training first.")
    utils.model = m
    return m

load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🧠 MS Lesion AI")
st.sidebar.markdown("**3D U-Net** trained on MSLesSeg + Mendeley + Long-MR-MS datasets.")
st.sidebar.markdown(f"**Device:** `{DEVICE_NAME}`")
st.sidebar.markdown("---")
mode = st.sidebar.radio("Mode", ["Single-Scan Analysis", "Longitudinal Tracking"])

# ── Helper: save uploaded file to disk (Streamlit gives BytesIO) ──────────────
def save_upload(uploaded_file):
    tmp_path = os.path.join("demo_outputs", uploaded_file.name)
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.read())
    return tmp_path


class _FileObj:
    """Thin wrapper so utils functions (which call .name) work with Streamlit uploads."""
    def __init__(self, path):
        self.name = path


# ── Single-Scan Analysis ──────────────────────────────────────────────────────
if mode == "Single-Scan Analysis":
    st.title("Single-Scan MS Lesion Analysis")
    st.markdown("Upload a **FLAIR MRI** (`.nii` or `.nii.gz`) to detect MS lesions.")

    col1, col2 = st.columns([1, 2])
    with col1:
        flair_file = st.file_uploader("FLAIR MRI (NIfTI)", type=["nii", "gz"], key="flair")
        gt_file    = st.file_uploader("Ground Truth Mask — optional", type=["nii", "gz"], key="gt")
        analyze    = st.button("Analyze Lesions", type="primary", disabled=flair_file is None)

    with col2:
        if analyze and flair_file:
            with st.spinner("Running N4 correction, skull strip, inference..."):
                flair_path = save_upload(flair_file)
                gt_path    = save_upload(gt_file) if gt_file else None
                mosaic_path, report = run_ai_analysis(
                    _FileObj(flair_path),
                    _FileObj(gt_path) if gt_path else None,
                )
            if mosaic_path:
                st.image(mosaic_path, caption="Axial | Coronal | Sagittal Overlay", use_container_width=True)
            st.markdown(report)

# ── Longitudinal Tracking ─────────────────────────────────────────────────────
elif mode == "Longitudinal Tracking":
    st.title("Longitudinal MS Lesion Tracking")
    st.markdown("Upload **Baseline (T0)** and **Follow-up (T1)** FLAIR scans to detect disease progression.")
    st.info("Scans are rigidly registered before comparison — no manual alignment needed.")

    col1, col2 = st.columns(2)
    with col1:
        t0_file = st.file_uploader("Baseline Scan — T0 (.nii / .nii.gz)", type=["nii", "gz"], key="t0")
    with col2:
        t1_file = st.file_uploader("Follow-up Scan — T1 (.nii / .nii.gz)", type=["nii", "gz"], key="t1")

    run_btn = st.button("Detect Progression", type="primary", disabled=not (t0_file and t1_file))

    if run_btn and t0_file and t1_file:
        with st.spinner("Preprocessing, segmenting, registering..."):
            t0_path = save_upload(t0_file)
            t1_path = save_upload(t1_file)
            img_t0, change_img, report = run_longitudinal_analysis(
                _FileObj(t0_path), _FileObj(t1_path)
            )

        col_a, col_b = st.columns(2)
        with col_a:
            if img_t0:
                st.image(img_t0, caption="Baseline Segmentation (T0)", use_container_width=True)
        with col_b:
            if change_img:
                st.image(change_img, caption="Change Map  [Green = New   Blue = Resolved]", use_container_width=True)

        st.markdown(report)
