"""PyTorch Dataset for MiniCADNet cropped damage images.

Applies lightweight data augmentation ONLY when split == 'train'.
Validation and test splits remain strictly unaugmented.
"""

from __future__ import annotations

import csv
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import CFG


class MiniCADDataset(Dataset):
    def __init__(
        self,
        split: str,
        processed_dir: Path | None = None,
        seed: int = CFG.seed,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")

        self.split = split
        self.is_train = split == "train"
        self.processed_dir = Path(processed_dir or CFG.processed_data_dir)
        self.rng = random.Random(seed)
        self.samples: list[tuple[Path, int]] = []

        csv_path = self.processed_dir / "metadata.csv"
        with csv_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["split"] == split:
                    self.samples.append((self.processed_dir / row["crop_path"], int(row["class_id"])))

    def __len__(self) -> int:
        return len(self.samples)

    def _augment_train(self, img_rgb: np.ndarray) -> np.ndarray:
        # Random horizontal flip
        if self.rng.random() < 0.5:
            img_rgb = np.fliplr(img_rgb).copy()

        # Slight brightness/contrast jitter
        if self.rng.random() < 0.5:
            alpha = self.rng.uniform(0.85, 1.15)  # contrast
            beta = self.rng.uniform(-15.0, 15.0)  # brightness
            img_rgb = np.clip(img_rgb.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # Small random rotation (-12 to +12 deg)
        if self.rng.random() < 0.5:
            h, w = img_rgb.shape[:2]
            angle = self.rng.uniform(-12.0, 12.0)
            mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
            img_rgb = cv2.warpAffine(img_rgb, mat, (w, h), borderMode=cv2.BORDER_REFLECT_101)

        return img_rgb

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read crop image: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if self.is_train:
            img_rgb = self._augment_train(img_rgb)

        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        return tensor, label
