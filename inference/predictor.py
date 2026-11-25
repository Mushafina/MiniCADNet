"""Inference helper for MiniCADNet v1."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import CFG  # noqa: E402
from models.minicadnet import MiniCADNet  # noqa: E402


class MiniCADPredictor:
    """Loads a trained MiniCADNet checkpoint and predicts 5-class damage probabilities."""

    def __init__(self, checkpoint_path: Path | str | None = None, device: str | None = None) -> None:
        self.checkpoint_path = Path(checkpoint_path or (CFG.checkpoint_dir / "best_model.pth"))
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                "Trained model checkpoint (best_model.pth) was not found. "
                "Ensure outputs/checkpoints/best_model.pth is present."
            )

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        if "model_state_dict" not in ckpt:
            raise ValueError("Invalid checkpoint format: missing 'model_state_dict'.")

        self.class_names: tuple[str, ...] = tuple(ckpt.get("class_names", CFG.target_classes))
        self.epoch: int = int(ckpt.get("epoch", -1))
        if self.epoch <= 0:
            raise ValueError(f"Checkpoint does not appear to be trained (epoch={self.epoch}).")

        self.model = MiniCADNet(num_classes=len(self.class_names)).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.trainable_params: int = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    @staticmethod
    def preprocess_pil(image: Image.Image, target_size: tuple[int, int] = CFG.image_size) -> torch.Tensor:
        """Convert PIL Image to (1, 3, 96, 96) RGB float32 tensor in [0, 1]."""
        img_rgb = np.array(image.convert("RGB"))
        target_w, target_h = target_size
        h, w = img_rgb.shape[:2]
        interp = cv2.INTER_AREA if (w > target_w or h > target_h) else cv2.INTER_LINEAR
        resized = cv2.resize(img_rgb, (target_w, target_h), interpolation=interp)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0)

    def predict_pil(self, image: Image.Image) -> dict[str, Any]:
        """Run inference on a PIL image and return class prediction and probabilities."""
        x = self.preprocess_pil(image).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        prob_dict = {name: float(probs[i]) for i, name in enumerate(self.class_names)}
        return {
            "predicted_class": self.class_names[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": prob_dict,
            "logits_shape": tuple(logits.shape),
        }
