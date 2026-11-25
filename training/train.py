"""Train MiniCADNet from scratch on 5-class automotive damage crops."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import CFG  # noqa: E402
from dataset import MiniCADDataset  # noqa: E402
from models.minicadnet import MiniCADNet  # noqa: E402
from training.utils import (  # noqa: E402
    compute_metrics,
    compute_train_class_weights,
    save_training_curves,
    set_seed,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
    desc: str = "",
) -> tuple[float, list[int], list[int]]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    iterator = tqdm(loader, desc=desc, leave=False) if tqdm is not None else loader

    with torch.set_grad_enabled(is_train):
        for batch_idx, (images, labels) in enumerate(iterator):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            bs = labels.size(0)
            total_loss += float(loss.item()) * bs
            total_samples += bs

            preds = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    avg_loss = total_loss / max(1, total_samples)
    return avg_loss, y_true, y_pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MiniCADNet from scratch.")
    parser.add_argument("--epochs", type=int, default=CFG.num_epochs)
    parser.add_argument("--batch-size", type=int, default=CFG.batch_size)
    parser.add_argument("--lr", type=float, default=CFG.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=CFG.weight_decay)
    parser.add_argument("--patience", type=int, default=CFG.early_stopping_patience)
    parser.add_argument("--seed", type=int, default=CFG.seed)
    parser.add_argument("--num-workers", type=int, default=CFG.num_workers)
    parser.add_argument("--output-dir", type=Path, default=CFG.output_dir)
    parser.add_argument("--resume", type=Path, default=None, help="Optional checkpoint path to resume from.")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit batches per epoch (for sanity checks).")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = args.output_dir / "checkpoints"
    metrics_dir = args.output_dir / "metrics"
    plots_dir = args.output_dir / "plots"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load train and validation splits only (keep test split untouched)
    train_ds = MiniCADDataset("train", processed_dir=CFG.processed_data_dir, seed=args.seed)
    val_ds = MiniCADDataset("val", processed_dir=CFG.processed_data_dir, seed=args.seed)

    # Verify tensor format and normalization
    sample_x, sample_y = val_ds[0]
    assert sample_x.shape == (3, CFG.image_size[1], CFG.image_size[0]), f"Unexpected shape {sample_x.shape}"
    assert sample_x.dtype == torch.float32, f"Unexpected dtype {sample_x.dtype}"
    assert 0.0 <= float(sample_x.min()) and float(sample_x.max()) <= 1.0, "Input tensor not in [0, 1]"

    # Compute class weights strictly from the training split
    train_labels = [label for _, label in train_ds.samples]
    class_weights, train_counts = compute_train_class_weights(train_labels, CFG.num_classes)
    class_weights = class_weights.to(device)

    pin_mem = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )

    model = MiniCADNet(num_classes=CFG.num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 1
    best_f1 = -1.0
    best_val_loss = float("inf")
    best_val_metrics: dict = {}
    epochs_without_improvement = 0
    history: list[dict] = []

    if args.resume is not None and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_f1 = float(ckpt.get("best_val_f1", -1.0))
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        best_val_metrics = ckpt.get("best_val_metrics", {})
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    print(f"Device: {device} | Train crops: {len(train_ds)} | Val crops: {len(val_ds)}")
    print("Training class counts:", {CFG.target_classes[k]: v for k, v in sorted(train_counts.items())})
    print("Training class weights:", [round(w, 4) for w in class_weights.cpu().tolist()])

    config_snapshot = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in asdict(CFG).items()
    }
    config_snapshot.update(
        {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "early_stopping_patience": args.patience,
            "seed": args.seed,
        }
    )

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.perf_counter()

        train_loss, tr_true, tr_pred = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer,
            max_batches=args.max_batches, desc=f"Epoch {epoch}/{args.epochs} [Train]"
        )
        train_metrics = compute_metrics(tr_true, tr_pred, CFG.target_classes)

        with torch.no_grad():
            val_loss, va_true, va_pred = run_epoch(
                model, val_loader, criterion, device, optimizer=None,
                max_batches=args.max_batches, desc=f"Epoch {epoch}/{args.epochs} [Val]"
            )
        val_metrics = compute_metrics(va_true, va_pred, CFG.target_classes)

        duration = time.perf_counter() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "duration_sec": round(duration, 3),
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_metrics["accuracy"], 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_metrics["accuracy"], 6),
            "val_macro_precision": round(val_metrics["macro_precision"], 6),
            "val_macro_recall": round(val_metrics["macro_recall"], 6),
            "val_macro_f1": round(val_metrics["macro_f1"], 6),
        }
        history.append(row)

        val_f1 = val_metrics["macro_f1"]
        improved = (val_f1 > best_f1) or (abs(val_f1 - best_f1) < 1e-12 and val_loss < best_val_loss)

        if improved:
            best_f1 = val_f1
            best_val_loss = val_loss
            best_val_metrics = {
                "epoch": epoch,
                "val_loss": val_loss,
                **val_metrics,
            }
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "class_names": CFG.target_classes,
                    "config": config_snapshot,
                    "best_val_metrics": best_val_metrics,
                    "best_val_f1": best_f1,
                    "best_val_loss": best_val_loss,
                    "seed": args.seed,
                },
                ckpt_dir / "best_model.pth",
            )
            tag = " *BEST*"
        else:
            epochs_without_improvement += 1
            tag = ""

        # Save recovery checkpoint every epoch
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "class_names": CFG.target_classes,
                "config": config_snapshot,
                "best_val_metrics": best_val_metrics,
                "best_val_f1": best_f1,
                "best_val_loss": best_val_loss,
                "seed": args.seed,
            },
            ckpt_dir / "last_checkpoint.pth",
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} ({duration:.1f}s) | "
            f"Train Loss: {train_loss:.4f} Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_metrics['accuracy']:.4f} "
            f"P: {val_metrics['macro_precision']:.4f} R: {val_metrics['macro_recall']:.4f} "
            f"F1: {val_f1:.4f}{tag}"
        )

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping triggered after {args.patience} epochs without validation macro F1 improvement.")
            break

    # Save training history CSV, best validation metrics JSON, and curves plot
    history_csv = metrics_dir / "training_history.csv"
    if history:
        with history_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        save_training_curves(history, plots_dir / "training_curves.png")

    (metrics_dir / "best_validation_metrics.json").write_text(
        json.dumps(best_val_metrics, indent=2), encoding="utf-8"
    )
    print(f"\nSaved best checkpoint to: {ckpt_dir / 'best_model.pth'}")
    print(f"Saved training history to: {history_csv}")
    print(f"Saved best validation metrics to: {metrics_dir / 'best_validation_metrics.json'}")


if __name__ == "__main__":
    main()
