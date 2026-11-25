"""Inspect the DrBimmer/car-parts-and-damage-dataset on Hugging Face.

Lists repository structure, counts images/annotations, downloads annotation JSONs
(and optional sample images) into data/raw for fast inspection without downloading
the entire ~3.3 GB image archive prematurely.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import urllib.parse
import urllib.request

# Allow running directly as `python scripts/inspect_dataset.py`
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import CFG  # noqa: E402

HF_API_BASE = "https://huggingface.co/api/datasets"
HF_RESOLVE_BASE = "https://huggingface.co/datasets"


def fetch_json(url: str, timeout: int = 30) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": "MiniCADNet-Inspector/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def list_hf_tree(dataset_id: str, subpath: str = "") -> list[dict]:
    encoded_path = urllib.parse.quote(subpath, safe="/") if subpath else ""
    url = f"{HF_API_BASE}/{dataset_id}/tree/main"
    if encoded_path:
        url = f"{url}/{encoded_path}"
    data = fetch_json(url)
    return data if isinstance(data, list) else []


def download_file(dataset_id: str, remote_path: str, local_path: Path) -> Path:
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    encoded_path = urllib.parse.quote(remote_path, safe="/")
    url = f"{HF_RESOLVE_BASE}/{dataset_id}/resolve/main/{encoded_path}"
    req = urllib.request.Request(url, headers={"User-Agent": "MiniCADNet-Inspector/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        local_path.write_bytes(resp.read())
    return local_path


def inspect_image_bytes(img_path: Path) -> dict:
    info: dict[str, object] = {"file": img_path.name, "bytes": img_path.stat().st_size}
    try:
        import cv2  # type: ignore

        img = cv2.imread(str(img_path))
        if img is not None:
            h, w, c = img.shape
            info["shape_hwc"] = (h, w, c)
            return info
    except ImportError:
        pass

    try:
        from PIL import Image  # type: ignore

        with Image.open(img_path) as im:
            info["size_wh"] = im.size
            info["mode"] = im.mode
            return info
    except ImportError:
        pass

    # Minimal PNG/JPEG header fallback
    raw = img_path.read_bytes()[:32]
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        w = int.from_bytes(raw[16:20], "big")
        h = int.from_bytes(raw[20:24], "big")
        info["size_wh"] = (w, h)
        info["format"] = "PNG"
    elif raw[:2] == b"\xff\xd8":
        info["format"] = "JPEG"
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect HF automotive damage dataset.")
    parser.add_argument("--dataset-id", default=CFG.hf_dataset_id)
    parser.add_argument("--raw-dir", type=Path, default=CFG.raw_data_dir)
    parser.add_argument("--sample-images", type=int, default=3)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    print(f"Inspecting Hugging Face dataset: {args.dataset_id}")
    root_entries = list_hf_tree(args.dataset_id)
    print("\n[1] Top-level entries:")
    for entry in root_entries:
        print(f"  - {entry['type']:9s} {entry.get('size', 0):>10d}  {entry['path']}")

    # Inspect both subsets at top level
    subsets = [e["path"] for e in root_entries if e["type"] == "directory"]
    for subset in subsets:
        sub_entries = list_hf_tree(args.dataset_id, subset)
        print(f"\n[2] Contents of '{subset}':")
        for entry in sub_entries:
            print(f"  - {entry['type']:9s} {entry.get('size', 0):>10d}  {entry['path']}")

        # Check File1 subdirectories if present
        file1_dirs = [e["path"] for e in sub_entries if e["type"] == "directory"]
        for fdir in file1_dirs:
            fdir_entries = list_hf_tree(args.dataset_id, fdir)
            print(f"    Subfolders in '{fdir}':")
            for sub in fdir_entries:
                if sub["type"] == "directory":
                    files = list_hf_tree(args.dataset_id, sub["path"])
                    total_bytes = sum(f.get("size", 0) for f in files if f["type"] == "file")
                    print(
                        f"      - {sub['path']} -> {len(files)} items "
                        f"({total_bytes / (1024 * 1024):.2f} MB)"
                    )
                else:
                    print(f"      - {sub['path']} ({sub.get('size', 0)} bytes)")

    # Inspect annotation JSONs for both subsets (in case folder names are swapped)
    for subset in ("Car damages dataset", "Car parts dataset"):
        subset_entries = list_hf_tree(args.dataset_id, subset)
        for entry in subset_entries:
            if entry["type"] == "file" and entry["path"].endswith(".json"):
                local_meta = args.raw_dir / entry["path"]
                download_file(args.dataset_id, entry["path"], local_meta)
                print(f"\nDownloaded metadata file: {local_meta}")

        ann_remote_dir = f"{subset}/File1/ann"
        ann_entries = [e for e in list_hf_tree(args.dataset_id, ann_remote_dir) if e["type"] == "file"]
        print(f"\n[3] Downloading {len(ann_entries)} annotation JSON files for '{subset}'...")

        downloaded_anns: list[Path] = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    download_file,
                    args.dataset_id,
                    entry["path"],
                    args.raw_dir / entry["path"],
                ): entry["path"]
                for entry in ann_entries
            }
            for fut in as_completed(futures):
                downloaded_anns.append(fut.result())

        print(f"Downloaded/cached {len(downloaded_anns)} annotation files for '{subset}'.")

        # Parse annotation JSONs
        object_class_counts: Counter[str] = Counter()
        image_class_presence: Counter[str] = Counter()
        geometry_types: Counter[str] = Counter()
        objects_per_image: list[int] = []
        images_without_objects = 0
        single_label_images = 0
        multi_label_images = 0
        sample_ann_structure: dict | None = None

        for ann_path in sorted(downloaded_anns):
            data = json.loads(ann_path.read_text(encoding="utf-8"))
            if sample_ann_structure is None:
                sample_ann_structure = data

            objs = data.get("objects", [])
            objects_per_image.append(len(objs))
            if not objs:
                images_without_objects += 1

            classes_in_image: set[str] = set()
            for obj in objs:
                cls_title = obj.get("classTitle", "<missing>")
                geom_type = obj.get("geometryType", "<missing>")
                object_class_counts[cls_title] += 1
                geometry_types[geom_type] += 1
                classes_in_image.add(cls_title)

            for cls_title in classes_in_image:
                image_class_presence[cls_title] += 1

            if len(classes_in_image) == 1:
                single_label_images += 1
            elif len(classes_in_image) > 1:
                multi_label_images += 1

        print(f"\n[4] Annotation format summary ({subset}):")
        if sample_ann_structure is not None:
            print(f"  Top-level JSON keys: {list(sample_ann_structure.keys())}")
            print(f"  Sample image size field: {sample_ann_structure.get('size')}")
            if sample_ann_structure.get("objects"):
                first_obj = sample_ann_structure["objects"][0]
                print(f"  Sample object keys: {list(first_obj.keys())}")
        print(f"  Geometry types: {dict(geometry_types)}")
        print(f"  Total annotated images: {len(downloaded_anns)}")
        print(f"  Images with 0 objects: {images_without_objects}")
        print(f"  Images with exactly 1 class: {single_label_images}")
        print(f"  Images with >1 class (multi-label): {multi_label_images}")
        if objects_per_image:
            print(
                f"  Objects per image: min={min(objects_per_image)}, "
                f"max={max(objects_per_image)}, "
                f"avg={sum(objects_per_image) / len(objects_per_image):.2f}"
            )

        print(f"\n[5] Class distribution ({subset}):")
        print(f"  {'Class':<20s} {'Polygon Count':>14s} {'Images Containing Class':>25s}")
        print("  " + "-" * 61)
        for cls_name, poly_cnt in object_class_counts.most_common():
            img_cnt = image_class_presence[cls_name]
            print(f"  {cls_name:<20s} {poly_cnt:>14d} {img_cnt:>25d}")

        # Download & inspect a few sample images
        img_remote_dir = f"{subset}/File1/img"
        img_entries = [e for e in list_hf_tree(args.dataset_id, img_remote_dir) if e["type"] == "file"]
        ext_counts = Counter(Path(e["path"]).suffix.lower() for e in img_entries)
        print(f"\n[6] Images summary ({subset}):")
        print(f"  Total files in {img_remote_dir}: {len(img_entries)}")
        print(f"  Image extensions: {dict(ext_counts)}")

        for entry in img_entries[: args.sample_images]:
            local_img = download_file(args.dataset_id, entry["path"], args.raw_dir / entry["path"])
            print(f"  Sample image: {inspect_image_bytes(local_img)}")


if __name__ == "__main__":
    main()
