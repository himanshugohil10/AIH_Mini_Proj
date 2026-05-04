"""
app.py — Streamlit Web App for TB Detection via Vision Transformer

Pipeline mirrors heatmap_output.ipynb exactly:
  - ViT (google/vit-base-patch16-224) → prediction + confidence
  - MobileNetV2 + Grad-CAM             → heatmap overlay
  - Healthy images                     → clean original (no overlay)
  - TB images                          → Grad-CAM overlay on original

Both models cached via @st.cache_resource (loaded once per session).
"""

import os
import tempfile

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from transformers import ViTForImageClassification
import json

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Path constants ─────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "vit_tb_model.pth")
MAPPING_PATH = os.path.join(BASE_DIR, "class_mapping.json")

# ── Preprocessing transform (ImageNet standard) ────────────────────────────────
_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ==============================================================================
# Cached model loader — runs exactly once per Streamlit session
# ==============================================================================
@st.cache_resource(show_spinner="Loading models… (first run only)")
def _load_models():
    # ── Class mapping ──────────────────────────────────────────────────────────
    with open(MAPPING_PATH, "r") as f:
        raw = json.load(f)

    if all(isinstance(v, int) for v in raw.values()):
        idx_to_class = {v: k for k, v in raw.items()}
    else:
        idx_to_class = {int(k): v for k, v in raw.items()}

    num_classes = len(idx_to_class)

    # ── ViT model (classification only) ───────────────────────────────────────
    vit = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
        output_attentions=True 
    )
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint.get("state_dict", checkpoint)
    vit.load_state_dict(state_dict, strict=False)
    vit.to(DEVICE).eval()

    return vit, idx_to_class


# ==============================================================================
# Preprocessing  (Cell 5)
# ==============================================================================
def _preprocess(image_path):
    pil  = Image.open(image_path).convert("RGB")
    np_  = np.array(pil)
    tens = _TRANSFORM(pil).unsqueeze(0).to(DEVICE)
    return tens, np_

def extract_vit_attention(outputs):
    attentions = outputs.attentions[-1]

    attn = torch.mean(attentions, dim=1)
    cls_attn = attn[0, 0, 1:]

    grid_size = int(cls_attn.shape[0] ** 0.5)
    attn_map = cls_attn.reshape(grid_size, grid_size)

    attn_map = (attn_map - attn_map.min()) / (attn_map.max() + 1e-8)

    return attn_map.cpu().numpy()

def apply_vit_overlay(original_np, attention_map, alpha=0.5):
    H, W = original_np.shape[:2]

    heatmap = cv2.resize(attention_map, (W, H))
    heatmap_u8 = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(original_np, 1 - alpha, heatmap_color, alpha, 0)

    return overlay, heatmap
# ==============================================================================
# Region detection  (Cell 9 — 4-quadrant version from notebook)
# ==============================================================================
def _get_region(heatmap_resized, is_healthy):
    if is_healthy:
        return "No abnormal region detected"
    H, W    = heatmap_resized.shape
    mid_h, mid_w = H // 2, W // 2
    quads = {
        "upper left":  heatmap_resized[0:mid_h,  0:mid_w].sum(),
        "upper right": heatmap_resized[0:mid_h,  mid_w:W].sum(),
        "lower left":  heatmap_resized[mid_h:H,  0:mid_w].sum(),
        "lower right": heatmap_resized[mid_h:H,  mid_w:W].sum(),
    }
    dominant = max(quads, key=lambda k: quads[k])
    return f"{dominant} lung"


def run_inference(image_path: str) -> dict:
    vit, idx_to_class = _load_models()

    input_tensor, original_np = _preprocess(image_path)

    with torch.no_grad():
        outputs = vit(input_tensor)
        probs = F.softmax(outputs.logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    pred_idx = pred_idx.item()
    conf = conf.item()

    label = idx_to_class.get(pred_idx, f"class_{pred_idx}")
    is_tb = pred_idx == 1

    # 🔥 ViT Attention
    attention_map = extract_vit_attention(outputs)
    overlay, resized_map = apply_vit_overlay(original_np, attention_map)

    if not is_tb:
        display_image = original_np
        region = "No abnormal region detected"
    else:
        display_image = overlay
        region = _get_region(resized_map, False)

    return {
        "prediction": label,
        "confidence": conf,
        "region": region,
        "heatmap": display_image,
        "original": original_np,
    }


# ==============================================================================
# Page configuration
# ==============================================================================
st.set_page_config(
    page_title="TB Detection · ViT",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .hero-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        border-radius: 16px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 1.8rem;
        color: #fff;
    }
    .hero-header h1 { font-size: 2.2rem; font-weight: 700; margin: 0; }
    .hero-header p  { font-size: 1rem; opacity: 0.75; margin-top: 0.4rem; }

    .badge-tb      { background:#ff4b4b22; border:1.5px solid #ff4b4b; border-radius:8px;
                     padding:.45rem 1.1rem; color:#ff4b4b; font-weight:600; font-size:1.1rem; display:inline-block; }
    .badge-healthy { background:#21c97822; border:1.5px solid #21c978; border-radius:8px;
                     padding:.45rem 1.1rem; color:#21c978; font-weight:600; font-size:1.1rem; display:inline-block; }

    .info-card {
        background: #1e2a38;
        border-radius: 12px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 1rem;
        border-left: 4px solid #4fa3e0;
        color: #cdd8e5;
    }
    .info-card h4 { margin: 0 0 .3rem; font-size: .8rem; text-transform: uppercase;
                    letter-spacing: .08em; color: #7fa8c9; }
    .info-card p  { margin: 0; font-size: 1rem; font-weight: 500; color: #eaf2ff; }

    section[data-testid="stSidebar"] { background: #0d1b2a; }
    div[data-testid="stFileUploader"] { border-radius: 12px; }
    hr { border-color: #1e2a38; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/Chest_Xray_PA_3-8-2010.png/266px-Chest_Xray_PA_3-8-2010.png",
        caption="Sample Chest X-Ray",
        use_container_width=True,
    )
    st.markdown("---")
    st.markdown("### 🫁 About")
    st.markdown(
        "This tool uses a **fine-tuned Vision Transformer (ViT)** to classify "
        "chest X-rays as **Tuberculosis (TB)** or **Healthy**.\n\n"
        "**Grad-CAM** heatmaps (via MobileNetV2) highlight the lung region "
        "the model focused on. Healthy images show the clean original."
    )
    st.markdown("---")
    st.markdown("### 📋 Instructions")
    st.markdown(
        "1. Upload a chest X-ray (JPG / PNG)\n"
        "2. Wait for inference (~5 s on CPU)\n"
        "3. Review prediction & Grad-CAM heatmap"
    )
    st.markdown("---")
    st.caption("ViT: `google/vit-base-patch16-224` · Heatmap: MobileNetV2 Grad-CAM")

# ── Hero header ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-header">
        <h1>🫁 TB Detection via Vision Transformer</h1>
        <p>Upload a chest X-ray to get an AI-powered prediction with Grad-CAM attention heatmap.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── File uploader ──────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "**Upload Chest X-Ray Image**",
    type=["jpg", "jpeg", "png"],
    help="Accepts standard PA / AP chest X-rays in JPG or PNG format.",
)

if uploaded_file is not None:
    suffix = os.path.splitext(uploaded_file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("🔬 Running inference (ViT classification + Grad-CAM heatmap)…"):
        try:
            result = run_inference(tmp_path)
        except Exception as exc:
            st.error(f"Inference failed: {exc}")
            st.stop()
        finally:
            os.unlink(tmp_path)

    prediction = result["prediction"]
    confidence = result["confidence"]
    region     = result["region"]
    heatmap    = result["heatmap"]
    original   = result["original"]

    # ── Image panels ────────────────────────────────────────────────────────────
    st.markdown("### 📸 Visual Analysis")
    col_orig, col_heat = st.columns(2, gap="large")

    with col_orig:
        st.markdown("**Original X-Ray**")
        st.image(original, use_container_width=True, clamp=True)

    with col_heat:
        is_tb = prediction.lower() == "tb"
        panel_title = "**Grad-CAM Heatmap Overlay**" if is_tb else "**Original X-Ray (Healthy — No Overlay)**"
        st.markdown(panel_title)
        st.image(heatmap, use_container_width=True, clamp=True)

    st.markdown("---")

    # ── Result metrics ───────────────────────────────────────────────────────────
    st.markdown("### 🧠 Inference Results")
    m1, m2, m3 = st.columns(3, gap="medium")

    badge_cls  = "badge-tb" if is_tb else "badge-healthy"
    badge_icon = "🔴"       if is_tb else "🟢"

    with m1:
        st.markdown(
            f"""
            <div class="info-card">
                <h4>Prediction</h4>
                <span class="{badge_cls}">{badge_icon} {prediction.upper()}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown(
            f"""
            <div class="info-card">
                <h4>Confidence Score</h4>
                <p>{confidence:.2%}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(confidence)

    with m3:
        st.markdown(
            f"""
            <div class="info-card">
                <h4>Most Attended Region</h4>
                <p>📍 {region.title()}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.info("🧠 Heatmap shows where the Vision Transformer focused.")
    st.markdown("---")

    with st.expander("🤖 LLM-Based Explanation *(coming soon)*", expanded=False):
        st.info(
            f"⚠️  LLM explanation not yet implemented.\n\n"
            f"*(Inputs received — Prediction: {prediction}, "
            f"Confidence: {confidence:.2%}, Region: {region})*"
        )

else:
    # ── Empty state ──────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center; padding:3rem 1rem; color:#4a6380;">
            <div style="font-size:4rem;">🫁</div>
            <h3 style="color:#6e9dc0; margin-top:.5rem;">No image uploaded yet</h3>
            <p>Use the file uploader above to get started.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
