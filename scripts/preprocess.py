"""Preprocess DrBimmer/car-parts-and-damage-dataset into 96x96 damage crops.

Reads Supervisely polygon annotations from `Car parts dataset/File1/` (which holds
the 814 damage-annotated images), filters to the 5 approved damage classes,
splits source images 70/15/15 BEFORE cropping, extracts 15%-padded bounding-box
crops (capped at 5 per class per source image), resizes to 96x96, writes
metadata.csv, and generates a sample visualization grid.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import random
import shutil
import sys
import urllib.parse
import urllib.request

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import CFG  # noqa: E402

HF_API_BASE = "https://huggingface.co/api/datasets"
HF_RESOLVE_BASE = "https://huggingface.co/datasets"


@dataclass
class CandidateCrop:
    source_image: str
    img_w: int
    img_h: int
    class_name: str
    class_id: int
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    box_w: int
    box_h: int
    num_polygon_points: int


def fetch_json(url: str, timeout: int = 30) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": "MiniCADNet-Preprocess/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def list_hf_tree(dataset_id: str, subpath: str) -> list[dict]:
    encoded_path = urllib.parse.quote(subpath, safe="/")
    url = f"{HF_API_BASE}/{dataset_id}/tree/main/{encoded_path}"
    data = fetch_json(url)
    return data if isinstance(data, list) else []


def download_file(dataset_id: str, remote_path: str, local_path: Path) -> Path:
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    encoded_path = urllib.parse.quote(remote_path, safe="/")
    url = f"{HF_RESOLVE_BASE}/{dataset_id}/resolve/main/{encoded_path}"
    req = urllib.request.Request(url, headers={"User-Agent": "MiniCADNet-Preprocess/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        local_path.write_bytes(resp.read())
    return local_path


def ensure_annotations(dataset_id: str, subpath: str, raw_dir: Path, workers: int) -> list[Path]:
    ann_dir = raw_dir / subpath / "ann"
    existing = sorted(ann_dir.glob("*.json")) if ann_dir.exists() else []
    if existing:
        return existing

    remote_ann_dir = f"{subpath}/ann"
    entries = [e for e in list_hf_tree(dataset_id, remote_ann_dir) if e["type"] == "file"]
    print(f"Downloading {len(entries)} annotation JSONs to {ann_dir}...")
    paths: list[Path] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [
            pool.submit(download_file, dataset_id, e["path"], raw_dir / e["path"])
            for e in entries
        ]
        for fut in as_completed(futs):
            paths.append(fut.result())
    return sorted(paths)


def parse_annotations(
    ann_files: list[Path],
    target_classes: tuple[str, ...],
    min_box_size: int,
    max_per_class_per_image: int,
) -> tuple[dict[str, list[CandidateCrop]], Counter[str]]:
    class_to_id = {name: idx for idx, name in enumerate(target_classes)}
    rejection_counts: Counter[str] = Counter()
    crops_by_image: dict[str, list[CandidateCrop]] = {}

    for ann_path in ann_files:
        # Supervisely naming: "<image_filename>.json"
        source_image = ann_path.name[:-5] if ann_path.name.endswith(".json") else ann_path.stem
        data = json.loads(ann_path.read_text(encoding="utf-8"))

        size_info = data.get("size", {})
        img_w = int(size_info.get("width", 0))
        img_h = int(size_info.get("height", 0))
        if img_w <= 0 or img_h <= 0:
            rejection_counts["invalid_image_size_metadata"] += 1
            continue

        per_class_candidates: dict[str, list[CandidateCrop]] = defaultdict(list)

        for obj in data.get("objects", []):
            if obj.get("geometryType") != "polygon":
                rejection_counts["non_polygon_geometry"] += 1
                continue

            cls_name = obj.get("classTitle", "")
            if cls_name not in class_to_id:
                rejection_counts["non_target_class"] += 1
                continue

            exterior = obj.get("points", {}).get("exterior", [])
            if not isinstance(exterior, list) or len(exterior) < 3:
                rejection_counts["invalid_polygon_fewer_than_3_points"] += 1
                continue

            try:
                xs = [float(pt[0]) for pt in exterior]
                ys = [float(pt[1]) for pt in exterior]
            except (TypeError, ValueError, IndexError):
                rejection_counts["invalid_polygon_coordinates"] += 1
                continue

            x1 = max(0, min(img_w, int(np.floor(min(xs)))))
            y1 = max(0, min(img_h, int(np.floor(min(ys)))))
            x2 = max(0, min(img_w, int(np.ceil(max(xs)))))
            y2 = max(0, min(img_h, int(np.ceil(max(ys)))))

            box_w = x2 - x1
            box_h = y2 - y1
            if box_w <= 0 or box_h <= 0:
                rejection_counts["empty_bounding_box"] += 1
                continue

            if box_w < min_box_size or box_h < min_box_size:
                rejection_counts[f"too_small_lt_{min_box_size}px"] += 1
                continue

            per_class_candidates[cls_name].append(
                CandidateCrop(
                    source_image=source_image,
                    img_w=img_w,
                    img_h=img_h,
                    class_name=cls_name,
                    class_id=class_to_id[cls_name],
                    bbox_x1=x1,
                    bbox_y1=y1,
                    bbox_x2=x2,
                    bbox_y2=y2,
                    box_w=box_w,
                    box_h=box_h,
                    num_polygon_points=len(exterior),
                )
            )

        kept_for_image: list[CandidateCrop] = []
        for cls_name, candidates in per_class_candidates.items():
            # Keep largest-area instances first when capping per-image bias
            candidates.sort(key=lambda c: (c.box_w * c.box_h, c.box_w, c.box_h), reverse=True)
            if len(candidates) > max_per_class_per_image:
                rejection_counts["capped_per_class_per_image"] += len(candidates) - max_per_class_per_image
                candidates = candidates[:max_per_class_per_image]
            kept_for_image.extend(candidates)

        if kept_for_image:
            crops_by_image[source_image] = kept_for_image
        else:
            rejection_counts["source_image_no_valid_target_crops"] += 1

    return crops_by_image, rejection_counts


def split_source_images(
    crops_by_image: dict[str, list[CandidateCrop]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    # Order classes by image-level rarity so multi-label images stratify by their rarest class
    img_class_freq: Counter[str] = Counter()
    for crops in crops_by_image.values():
        for cls_name in {c.class_name for c in crops}:
            img_class_freq[cls_name] += 1

    images = sorted(crops_by_image.keys())
    strat_labels = [
        min({c.class_name for c in crops_by_image[img]}, key=lambda cls: img_class_freq[cls])
        for img in images
    ]

    val_test_ratio = val_ratio + test_ratio
    train_imgs, temp_imgs, _, temp_labels = train_test_split(
        images,
        strat_labels,
        test_size=val_test_ratio,
        random_state=seed,
        stratify=strat_labels,
    )

    test_share_of_temp = test_ratio / val_test_ratio
    val_imgs, test_imgs = train_test_split(
        temp_imgs,
        test_size=test_share_of_temp,
        random_state=seed,
        stratify=temp_labels,
    )

    assert set(train_imgs).isdisjoint(val_imgs)
    assert set(train_imgs).isdisjoint(test_imgs)
    assert set(val_imgs).isdisjoint(test_imgs)

    return {"train": sorted(train_imgs), "val": sorted(val_imgs), "test": sorted(test_imgs)}


def ensure_needed_images(
    dataset_id: str,
    subpath: str,
    raw_dir: Path,
    needed_filenames: list[str],
    workers: int,
) -> dict[str, Path]:
    img_dir = raw_dir / subpath / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    missing = [name for name in needed_filenames if not (img_dir / name).exists() or (img_dir / name).stat().st_size == 0]

    if missing:
        print(f"Downloading {len(missing)} source images from Hugging Face to {img_dir}...")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(
                    download_file,
                    dataset_id,
                    f"{subpath}/img/{name}",
                    img_dir / name,
                ): name
                for name in missing
            }
            done = 0
            for fut in as_completed(futs):
                fut.result()
                done += 1
                if done % 100 == 0 or done == len(missing):
                    print(f"  Downloaded {done}/{len(missing)} images")
    else:
        print(f"All {len(needed_filenames)} needed source images are already cached.")

    return {name: img_dir / name for name in needed_filenames}


def build_visualization_grid(
    metadata_rows: list[dict],
    processed_dir: Path,
    target_classes: tuple[str, ...],
    output_path: Path,
    samples_per_class: int = 5,
    tile_size: int = 96,
    seed: int = 42,
) -> None:
    rng = random.Random(seed)
    by_class: dict[str, list[dict]] = defaultdict(list)
    for row in metadata_rows:
        by_class[row["class_name"]].append(row)

    label_col_w = 135
    pad = 8
    header_h = 30
    rows_count = len(target_classes)
    canvas_h = header_h + rows_count * (tile_size + pad) + pad
    canvas_w = label_col_w + samples_per_class * (tile_size + pad) + pad
    canvas = np.full((canvas_h, canvas_w, 3), 245, dtype=np.uint8)

    cv2.putText(
        canvas,
        "MiniCADNet Sample Crops (96x96, 15% Context Padding)",
        (pad, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )

    for r_idx, cls_name in enumerate(target_classes):
        y0 = header_h + pad + r_idx * (tile_size + pad)
        cv2.putText(
            canvas,
            cls_name,
            (pad, y0 + tile_size // 2 + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

        candidates = list(by_class[cls_name])
        rng.shuffle(candidates)
        chosen = candidates[:samples_per_class]

        for c_idx, row in enumerate(chosen):
            x0 = label_col_w + c_idx * (tile_size + pad)
            crop_img = cv2.imread(str(processed_dir / row["crop_path"]))
            if crop_img is None:
                continue
            if crop_img.shape[:2] != (tile_size, tile_size):
                crop_img = cv2.resize(crop_img, (tile_size, tile_size))
            canvas[y0 : y0 + tile_size, x0 : x0 + tile_size] = crop_img
            cv2.putText(
                canvas,
                row["split"],
                (x0 + 4, y0 + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    cv2.imwrite(str(output_path), canvas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess MiniCADNet 5-class damage crops.")
    parser.add_argument("--clean", action="store_true", help="Remove existing processed folder before running.")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    processed_dir = CFG.processed_data_dir
    if args.clean and processed_dir.exists():
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    ann_files = ensure_annotations(CFG.hf_dataset_id, CFG.hf_damage_subpath, CFG.raw_data_dir, args.workers)
    print(f"Found {len(ann_files)} annotation files in {CFG.hf_damage_subpath}/ann.")

    crops_by_image, rejection_counts = parse_annotations(
        ann_files=ann_files,
        target_classes=CFG.target_classes,
        min_box_size=CFG.min_box_size,
        max_per_class_per_image=CFG.max_crops_per_class_per_image,
    )

    splits = split_source_images(
        crops_by_image=crops_by_image,
        train_ratio=CFG.train_split,
        val_ratio=CFG.val_split,
        test_ratio=CFG.test_split,
        seed=CFG.seed,
    )

    needed_images = sorted(crops_by_image.keys())
    image_paths = ensure_needed_images(
        dataset_id=CFG.hf_dataset_id,
        subpath=CFG.hf_damage_subpath,
        raw_dir=CFG.raw_data_dir,
        needed_filenames=needed_images,
        workers=args.workers,
    )

    target_w, target_h = CFG.image_size
    metadata_rows: list[dict] = []
    split_class_counts: dict[str, Counter[str]] = {s: Counter() for s in ("train", "val", "test")}
    total_class_counts: Counter[str] = Counter()
    used_source_images: set[str] = set()

    for split_name, img_list in splits.items():
        for cls_name in CFG.target_classes:
            (processed_dir / split_name / cls_name).mkdir(parents=True, exist_ok=True)

        for src_name in img_list:
            img = cv2.imread(str(image_paths[src_name]))
            if img is None:
                rejection_counts["unreadable_source_image"] += 1
                rejection_counts["dropped_crops_from_unreadable_image"] += len(crops_by_image[src_name])
                continue

            actual_h, actual_w = img.shape[:2]
            src_stem = Path(src_name).stem.replace(" ", "_")
            per_cls_idx: Counter[str] = Counter()

            for cand in crops_by_image[src_name]:
                x1 = min(cand.bbox_x1, actual_w)
                y1 = min(cand.bbox_y1, actual_h)
                x2 = min(cand.bbox_x2, actual_w)
                y2 = min(cand.bbox_y2, actual_h)

                bw = x2 - x1
                bh = y2 - y1
                pad_w = CFG.context_padding * bw
                pad_h = CFG.context_padding * bh

                px1 = max(0, int(round(x1 - pad_w)))
                py1 = max(0, int(round(y1 - pad_h)))
                px2 = min(actual_w, int(round(x2 + pad_w)))
                py2 = min(actual_h, int(round(y2 + pad_h)))

                crop = img[py1:py2, px1:px2]
                if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                    rejection_counts["empty_crop_after_padding"] += 1
                    continue

                interp = cv2.INTER_AREA if (crop.shape[1] > target_w or crop.shape[0] > target_h) else cv2.INTER_LINEAR
                resized = cv2.resize(crop, (target_w, target_h), interpolation=interp)

                per_cls_idx[cand.class_name] += 1
                cls_slug = cand.class_name.lower().replace(" ", "_")
                out_rel = Path(split_name) / cand.class_name / f"{src_stem}__{cls_slug}__{per_cls_idx[cand.class_name]:02d}.png"
                cv2.imwrite(str(processed_dir / out_rel), resized)

                metadata_rows.append(
                    {
                        "crop_path": out_rel.as_posix(),
                        "split": split_name,
                        "class_name": cand.class_name,
                        "class_id": cand.class_id,
                        "source_image": src_name,
                        "source_width": actual_w,
                        "source_height": actual_h,
                        "bbox_x1": x1,
                        "bbox_y1": y1,
                        "bbox_x2": x2,
                        "bbox_y2": y2,
                        "padded_x1": px1,
                        "padded_y1": py1,
                        "padded_x2": px2,
                        "padded_y2": py2,
                        "box_w": bw,
                        "box_h": bh,
                        "num_polygon_points": cand.num_polygon_points,
                    }
                )
                split_class_counts[split_name][cand.class_name] += 1
                total_class_counts[cand.class_name] += 1
                used_source_images.add(src_name)

    csv_path = processed_dir / "metadata.csv"
    fieldnames = list(metadata_rows[0].keys()) if metadata_rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)

    grid_path = processed_dir / "sample_grid.png"
    build_visualization_grid(
        metadata_rows=metadata_rows,
        processed_dir=processed_dir,
        target_classes=CFG.target_classes,
        output_path=grid_path,
        seed=CFG.seed,
    )

    summary = {
        "total_annotation_files": len(ann_files),
        "source_images_with_valid_crops": len(used_source_images),
        "split_source_image_counts": {k: len(v) for k, v in splits.items()},
        "total_crops": len(metadata_rows),
        "crops_per_class_total": {c: total_class_counts[c] for c in CFG.target_classes},
        "crops_per_class_by_split": {
            s: {c: split_class_counts[s][c] for c in CFG.target_classes}
            for s in ("train", "val", "test")
        },
        "rejections": dict(rejection_counts),
        "metadata_csv": str(csv_path),
        "sample_grid": str(grid_path),
    }
    (processed_dir / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n=== Preprocessing Summary ===")
    print(f"Total annotation files: {len(ann_files)}")
    print(f"Source images used: {len(used_source_images)}")
    print(f"Source image splits: { {k: len(v) for k, v in splits.items()} }")
    print(f"Total 96x96 crops generated: {len(metadata_rows)}")
    print("\nCrops per class by split:")
    print(f"  {'Class':<15s} {'Train':>7s} {'Val':>7s} {'Test':>7s} {'Total':>7s}")
    print("  " + "-" * 47)
    for c in CFG.target_classes:
        tr = split_class_counts["train"][c]
        va = split_class_counts["val"][c]
        te = split_class_counts["test"][c]
        tot = total_class_counts[c]
        print(f"  {c:<15s} {tr:>7d} {va:>7d} {te:>7d} {tot:>7d}")

    print("\nRejected annotations / skipped items:")
    for reason, cnt in sorted(rejection_counts.items()):
        print(f"  - {reason}: {cnt}")
    print(f"\nSaved metadata CSV: {csv_path}")
    print(f"Saved sample grid: {grid_path}")


if __name__ == "__main__":
    main()
