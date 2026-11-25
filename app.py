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
    st.set_page_config(page_title="MiniCADNet v1 Prototype", page_icon="🚗", layout="centered")

    st.title("MiniCADNet v1")
    st.caption("Miniature Compact Automotive Damage Classification Network (trained from scratch)")

    st.warning(
        "**Important Limitation:** This model classifies automotive damage images/crops. "
        "It is not a full vehicle damage detector. Full vehicle images may produce unreliable predictions."
    )

    with st.expander("Input Guidance (Recommended)"):
        st.markdown(
            "- Upload a **close-up crop** centered on a single damage region "
            "(`Scratch`, `Dent`, `Broken part`, `Paint chip`, or `Missing part`).\n"
            "- The model resizes the image to **96 × 96 RGB** and outputs a single-label classification.\n"
            "- It does **not** localize damage or draw bounding boxes."
        )

    ckpt_path = CFG.checkpoint_dir / "best_model.pth"
    model_param_count = sum(p.numel() for p in MiniCADNet(num_classes=CFG.num_classes).parameters() if p.requires_grad)

    with st.sidebar:
        st.header("Model Info")
        st.write(f"**Architecture:** Custom CNN (v1)")
        st.write(f"**Trainable Parameters:** {model_param_count:,}")
        st.write(f"**Input Size:** {CFG.image_size[0]} × {CFG.image_size[1]} RGB")
        st.write(f"**Classes ({CFG.num_classes}):** {', '.join(CFG.target_classes)}")
        st.write(f"**Checkpoint:** {'Found' if ckpt_path.exists() else 'Missing'}")

    if not ckpt_path.exists():
        st.error(
            f"Trained checkpoint not found at `{ckpt_path}`.\n\n"
            "Please copy `best_model.pth` from your Colab/Google Drive backup "
            "(`MyDrive/MiniCADNet_v1/outputs/checkpoints/best_model.pth`) into `outputs/checkpoints/best_model.pth` "
            "and reload this page."
        )
        return

    try:
        predictor = load_predictor(str(ckpt_path))
    except Exception as exc:
        st.error(f"Failed to load model checkpoint: {exc}")
        return

    uploaded_file = st.file_uploader(
        "Upload a cropped automotive damage image (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        return

    try:
        image = Image.open(uploaded_file)
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        st.error(f"Invalid image file: {exc}")
        return

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict", type="primary"):
        result = predictor.predict_pil(image)
        pred_cls = result["predicted_class"]
        conf = result["confidence"]
        probs = result["probabilities"]

        st.subheader(f"Prediction: {pred_cls}")
        st.metric("Confidence", f"{conf * 100:.2f}%")

        if conf < LOW_CONFIDENCE_THRESHOLD:
            st.warning(
                f"Low confidence prediction ({conf * 100:.1f}% < {LOW_CONFIDENCE_THRESHOLD * 100:.0f}%). "
                "Try uploading a tighter close-up crop of the damaged region."
            )

        st.subheader("Class Probabilities")
        # Horizontal bar chart sorted by probability descending
        sorted_items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        for cls_name, p in sorted_items:
            col_label, col_bar, col_val = st.columns([2, 5, 1.5])
            col_label.write(cls_name)
            col_bar.progress(min(max(p, 0.0), 1.0))
            col_val.write(f"{p * 100:.2f}%")


if __name__ == "__main__":
    main()
