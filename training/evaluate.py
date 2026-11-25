"""Evaluate the best trained MiniCADNet checkpoint once on the test split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import CFG  # noqa: E402
from dataset import MiniCADDataset  # noqa: E402
from models.minicadnet import MiniCADNet  # noqa: E402
from training.utils import (  # noqa: E402
    compute_metrics,
    save_confusion_matrix_plot,
    set_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate best MiniCADNet checkpoint on test split.")
    parser.add_argument("--checkpoint", type=Path, default=CFG.checkpoint_dir / "best_model.pth")
    parser.add_argument("--batch-size", type=int, default=CFG.batch_size)
    parser.add_argument("--num-workers", type=int, default=CFG.num_workers)
    parser.add_argument("--output-dir", type=Path, default=CFG.output_dir)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    seed = int(ckpt.get("seed", CFG.seed))
    set_seed(seed)

    class_names = tuple(ckpt.get("class_names", CFG.target_classes))
    model = MiniCADNet(num_classes=len(class_names)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_ds = MiniCADDataset("test", processed_dir=CFG.processed_data_dir, seed=seed)
    assert not test_ds.is_train, "Test dataset must remain unaugmented."

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.cpu().tolist())

    metrics = compute_metrics(y_true, y_pred, class_names)
    metrics["checkpoint_epoch"] = int(ckpt.get("epoch", -1))
    metrics["num_test_samples"] = len(test_ds)

    metrics_dir = args.output_dir / "metrics"
    plots_dir = args.output_dir / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON metrics
    test_metrics_path = metrics_dir / "test_metrics.json"
    test_metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Save per-class classification report CSV
    report_csv_path = metrics_dir / "classification_report.csv"
    with report_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["class_name", "precision", "recall", "f1", "support"]
        )
        writer.writeheader()
        for cls_name in class_names:
            row = metrics["per_class"][cls_name]
            writer.writerow(
                {
                    "class_name": cls_name,
                    "precision": round(row["precision"], 6),
                    "recall": round(row["recall"], 6),
                    "f1": round(row["f1"], 6),
                    "support": row["support"],
                }
            )
        writer.writerow(
            {
                "class_name": "macro_avg",
                "precision": round(metrics["macro_precision"], 6),
                "recall": round(metrics["macro_recall"], 6),
                "f1": round(metrics["macro_f1"], 6),
                "support": len(test_ds),
            }
        )

    cm = np.array(metrics["confusion_matrix"], dtype=np.int64)
    cm_plot_path = plots_dir / "confusion_matrix.png"
    save_confusion_matrix_plot(cm, class_names, cm_plot_path)

    print("=== MiniCADNet Test Evaluation ===")
    print(f"Checkpoint: {args.checkpoint} (epoch {metrics['checkpoint_epoch']})")
    print(f"Test samples: {len(test_ds)}")
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall:    {metrics['macro_recall']:.4f}")
    print(f"Macro F1:        {metrics['macro_f1']:.4f}")
    print("\nPer-class metrics:")
    print(f"  {'Class':<15s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>9s}")
    print("  " + "-" * 58)
    for cls_name in class_names:
        r = metrics["per_class"][cls_name]
        print(
            f"  {cls_name:<15s} {r['precision']:>10.4f} {r['recall']:>10.4f} "
            f"{r['f1']:>10.4f} {r['support']:>9d}"
        )
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)
    print(f"\nSaved test metrics to: {test_metrics_path}")
    print(f"Saved classification report to: {report_csv_path}")
    print(f"Saved confusion matrix plot to: {cm_plot_path}")


if __name__ == "__main__":
    main()
