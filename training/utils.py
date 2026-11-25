"""Utility functions for MiniCADNet training and evaluation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import torch


def set_seed(seed: int) -> None:
    """Set reproducible random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_train_class_weights(labels: list[int], num_classes: int) -> tuple[torch.Tensor, dict[int, int]]:
    """Compute inverse-frequency balanced class weights from training labels only."""
    counts = Counter(labels)
    total = len(labels)
    weights = []
    for cls_id in range(num_classes):
        cls_count = counts.get(cls_id, 0)
        if cls_count == 0:
            raise ValueError(f"Training split has 0 samples for class_id={cls_id}")
        weights.append(total / (num_classes * cls_count))
    return torch.tensor(weights, dtype=torch.float32), dict(counts)


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    class_names: tuple[str, ...],
) -> dict:
    """Compute accuracy, macro metrics, per-class metrics, and confusion matrix."""
    labels = list(range(len(class_names)))
    acc = float(accuracy_score(y_true, y_pred))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    per_p, per_r, per_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class = {}
    for idx, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(per_p[idx]),
            "recall": float(per_r[idx]),
            "f1": float(per_f1[idx]),
            "support": int(support[idx]),
        }

    return {
        "accuracy": acc,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def _draw_line_panel(
    canvas: np.ndarray,
    x0: int,
    y0: int,
    w: int,
    h: int,
    title: str,
    series: list[tuple[str, list[float], tuple[int, int, int]]],
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    """Draw a clean line chart panel onto an OpenCV BGR canvas."""
    pad_l, pad_r, pad_t, pad_b = 55, 20, 35, 35
    px0, py0 = x0 + pad_l, y0 + pad_t
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b

    cv2.rectangle(canvas, (px0, py0), (px0 + pw, py0 + ph), (180, 180, 180), 1)
    cv2.putText(canvas, title, (px0, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)

    all_vals = [v for _, vals, _ in series for v in vals]
    if not all_vals:
        return
    lo = min(all_vals) if y_min is None else y_min
    hi = max(all_vals) if y_max is None else y_max
    if abs(hi - lo) < 1e-8:
        hi = lo + 1.0

    # Horizontal gridlines & y-axis labels
    for frac in (0.0, 0.5, 1.0):
        gy = int(py0 + ph - frac * ph)
        val = lo + frac * (hi - lo)
        cv2.line(canvas, (px0, gy), (px0 + pw, gy), (225, 225, 225), 1)
        cv2.putText(canvas, f"{val:.2f}", (x0 + 6, gy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (70, 70, 70), 1, cv2.LINE_AA)

    n_pts = max(len(vals) for _, vals, _ in series)
    for idx, (label, vals, color) in enumerate(series):
        if len(vals) == 1:
            pt = (px0 + pw // 2, int(py0 + ph - ((vals[0] - lo) / (hi - lo)) * ph))
            cv2.circle(canvas, pt, 4, color, -1)
        else:
            pts = []
            for i, v in enumerate(vals):
                px = int(px0 + (i / max(1, n_pts - 1)) * pw)
                py = int(py0 + ph - ((v - lo) / (hi - lo)) * ph)
                pts.append((px, py))
            for i in range(len(pts) - 1):
                cv2.line(canvas, pts[i], pts[i + 1], color, 2, cv2.LINE_AA)

        # Legend
        lx = px0 + 10 + idx * 125
        ly = py0 + 16
        cv2.line(canvas, (lx, ly - 4), (lx + 18, ly - 4), color, 2)
        cv2.putText(canvas, label, (lx + 24, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1, cv2.LINE_AA)

    cv2.putText(canvas, f"Epochs (1..{n_pts})", (px0 + pw // 2 - 40, y0 + h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70, 70, 70), 1, cv2.LINE_AA)


def save_training_curves(history: list[dict], out_path: Path) -> None:
    """Save training/validation loss, accuracy, and macro F1 curves as a PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel_w, panel_h = 460, 320
    canvas = np.full((panel_h, panel_w * 3, 3), 252, dtype=np.uint8)

    _draw_line_panel(
        canvas,
        0,
        0,
        panel_w,
        panel_h,
        "Loss (Weighted CrossEntropy)",
        [
            ("Train", [r["train_loss"] for r in history], (200, 90, 30)),
            ("Val", [r["val_loss"] for r in history], (30, 120, 220)),
        ],
    )
    _draw_line_panel(
        canvas,
        panel_w,
        0,
        panel_w,
        panel_h,
        "Accuracy",
        [
            ("Train", [r["train_acc"] for r in history], (200, 90, 30)),
            ("Val", [r["val_acc"] for r in history], (30, 120, 220)),
        ],
        y_min=0.0,
        y_max=1.0,
    )
    _draw_line_panel(
        canvas,
        panel_w * 2,
        0,
        panel_w,
        panel_h,
        "Validation Macro F1",
        [
            ("Val Macro F1", [r["val_macro_f1"] for r in history], (40, 160, 60)),
        ],
        y_min=0.0,
        y_max=1.0,
    )

    cv2.imwrite(str(out_path), canvas)


def save_confusion_matrix_plot(
    cm: np.ndarray,
    class_names: tuple[str, ...],
    out_path: Path,
) -> None:
    """Save a readable confusion matrix heatmap PNG using OpenCV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(class_names)
    cell = 92
    left_margin = 140
    top_margin = 90
    bottom_margin = 95
    right_margin = 30

    w = left_margin + n * cell + right_margin
    h = top_margin + n * cell + bottom_margin
    canvas = np.full((h, w, 3), 252, dtype=np.uint8)

    cv2.putText(canvas, "MiniCADNet Test Confusion Matrix", (left_margin, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Predicted Class", (left_margin + (n * cell) // 2 - 55, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1, cv2.LINE_AA)

    row_sums = cm.sum(axis=1, keepdims=True)
    for r in range(n):
        cy = top_margin + r * cell
        cv2.putText(canvas, class_names[r], (12, cy + cell // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
        for c in range(n):
            cx = left_margin + c * cell
            if r == 0:
                cv2.putText(canvas, class_names[c], (cx + 6, top_margin - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)

            frac = float(cm[r, c] / row_sums[r, 0]) if row_sums[r, 0] > 0 else 0.0
            # Blue intensity proportional to row-normalized recall
            b = int(245 - frac * 45)
            g = int(245 - frac * 135)
            red = int(245 - frac * 195)
            cv2.rectangle(canvas, (cx, cy), (cx + cell, cy + cell), (b, g, red), -1)
            cv2.rectangle(canvas, (cx, cy), (cx + cell, cy + cell), (180, 180, 180), 1)

            txt_color = (255, 255, 255) if frac > 0.55 else (20, 20, 20)
            cv2.putText(canvas, str(int(cm[r, c])), (cx + 26, cy + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, txt_color, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{frac * 100:.1f}%", (cx + 20, cy + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.38, txt_color, 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), canvas)
