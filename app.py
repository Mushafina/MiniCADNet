"""Streamlit demo prototype for MiniCADNet v1."""

from __future__ import annotations

from pathlib import Path
from PIL import Image, UnidentifiedImageError
import streamlit as st

from config import CFG
from inference.predictor import MiniCADPredictor
from models.minicadnet import MiniCADNet

LOW_CONFIDENCE_THRESHOLD = 0.50


@st.cache_resource
def load_predictor(ckpt_path: str) -> MiniCADPredictor:
    return MiniCADPredictor(checkpoint_path=Path(ckpt_path))


def main() -> None:
    st.set_page_config(
        page_title="MiniCADNet — Automotive Damage Classifier",
        page_icon="🚗",
        layout="wide",
    )

    st.title("MiniCADNet v1 — Automotive Damage Classification")
    st.caption(
        "Miniature Compact Automotive Damage Classification Network | "
        "Custom CNN trained from scratch (4,397 trainable parameters)"
    )

    st.warning(
        "**Critical Input Limitation:** This model classifies automotive damage images/crops. "
        "It is not a full vehicle damage detector. Full vehicle images may produce unreliable predictions."
    )

    ckpt_path = CFG.checkpoint_dir / "best_model.pth"
    model_param_count = sum(
        p.numel() for p in MiniCADNet(num_classes=CFG.num_classes).parameters() if p.requires_grad
    )

    predictor: MiniCADPredictor | None = None
    ckpt_error: str | None = None
    if not ckpt_path.exists():
        ckpt_error = (
            "Trained model checkpoint (`best_model.pth`) was not found. "
            "Place the trained checkpoint in `outputs/checkpoints/best_model.pth` and reload."
        )
    else:
        try:
            predictor = load_predictor(str(ckpt_path))
        except Exception as exc:
            ckpt_error = f"Failed to load trained checkpoint: {exc}"

    with st.sidebar:
        st.header("Model Information")
        st.markdown(
            f"- **Model:** MiniCADNet (v1)\n"
            f"- **Architecture:** Custom CNN trained from scratch\n"
            f"- **Trainable Parameters:** `{model_param_count:,}`\n"
            f"- **Input Size:** `{CFG.image_size[0]} × {CFG.image_size[1]}` RGB\n"
            f"- **Categories ({CFG.num_classes}):** {', '.join(CFG.target_classes)}"
        )
        if predictor is not None:
            st.success(f"Checkpoint loaded (Epoch {predictor.epoch})")
        else:
            st.error("Checkpoint unavailable")

        with st.expander("Input Guidance"):
            st.markdown(
                "- Upload a **close-up crop** centered on a single damage region.\n"
                "- Supported formats: **JPG, JPEG, PNG**.\n"
                "- Images are resized to **96 × 96 RGB** before inference.\n"
                "- MiniCADNet does **not** localize damage or draw bounding boxes."
            )

    if ckpt_error is not None or predictor is None:
        st.error(ckpt_error or "Model checkpoint could not be loaded.")
        return

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("1. Upload Damage Crop")
        uploaded_file = st.file_uploader(
            "Choose a cropped automotive damage image (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded_file is None:
            st.info("Upload a close-up damage crop to run classification.")
            return

        try:
            image = Image.open(uploaded_file)
            image.load()
        except (UnidentifiedImageError, OSError):
            st.error("Invalid or corrupted image file. Please upload a valid JPG, JPEG, or PNG image.")
            return

        st.image(image, caption=f"Uploaded Image ({image.size[0]} × {image.size[1]} px)", use_container_width=True)
        run_clicked = st.button("Predict Damage Class", type="primary", use_container_width=True)

    with col_right:
        st.subheader("2. Classification Results")
        if not run_clicked:
            st.info("Click **Predict Damage Class** to inspect model predictions.")
            return

        result = predictor.predict_pil(image)
        pred_cls = result["predicted_class"]
        conf = result["confidence"]
        probs = result["probabilities"]

        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Predicted Category", pred_cls)
        m_col2.metric("Confidence", f"{conf * 100:.2f}%")

        if conf < LOW_CONFIDENCE_THRESHOLD:
            st.warning(
                f"Low confidence prediction ({conf * 100:.1f}% < {LOW_CONFIDENCE_THRESHOLD * 100:.0f}%). "
                "The image may contain multiple damage types, background clutter, or a full-vehicle view."
            )

        st.markdown("#### Class Probabilities")
        sorted_items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        for cls_name, p in sorted_items:
            c_lbl, c_bar, c_pct = st.columns([2.2, 5.0, 1.5])
            c_lbl.write(f"**{cls_name}**" if cls_name == pred_cls else cls_name)
            c_bar.progress(min(max(p, 0.0), 1.0))
            c_pct.write(f"{p * 100:.2f}%")


if __name__ == "__main__":
    main()
