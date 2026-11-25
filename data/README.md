# MiniCADNet Dataset Notes

Raw and processed images are excluded from this repository.

## Source Dataset
- **Hugging Face:** `DrBimmer/car-parts-and-damage-dataset`
- **Subfolder used:** `Car parts dataset/File1/` (in the upstream repo, folder names are swapped; `Car parts dataset` holds the 814 damage-annotated images and Supervisely polygon JSONs).

## Processed Classification Dataset (v1)
- **Classes (5):** `Scratch`, `Dent`, `Broken part`, `Paint chip`, `Missing part`
- **Crop extraction:** Bounding boxes derived from Supervisely polygons with 15% context padding, clamped to image bounds, minimum box size 16 px, capped at 5 crops per class per source image, resized to `96 x 96` RGB.
- **Leakage-free split (at source-image level before cropping):**
  - `train`: 566 source images -> 3,140 crops
  - `val`: 121 source images -> 724 crops
  - `test`: 122 source images -> 717 crops
  - **Total:** 809 source images -> 4,581 crops

## Reproduce Locally
```bash
python scripts/inspect_dataset.py
python scripts/preprocess.py
```
