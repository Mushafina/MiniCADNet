"""Configuration parameters for MiniCADNet."""

from dataclasses import dataclass, field
from pathlib import Path
import torch


@dataclass
class Config:
    # Project & paths
    project_name: str = "MiniCADNet"
    base_dir: Path = Path(__file__).resolve().parent
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "data")
    raw_data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "data" / "raw")
    processed_data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "data" / "processed")
    output_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "outputs")
    checkpoint_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "outputs" / "checkpoints")
    metrics_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "outputs" / "metrics")
    plots_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "outputs" / "plots")

    # Dataset source (Note: 'Car parts dataset/File1' holds the damage annotations on HF)
    hf_dataset_id: str = "DrBimmer/car-parts-and-damage-dataset"
    hf_damage_subpath: str = "Car parts dataset/File1"

    # Approved 5-class damage taxonomy
    target_classes: tuple[str, ...] = (
        "Scratch",
        "Dent",
        "Broken part",
        "Paint chip",
        "Missing part",
    )
    num_classes: int = 5

    # Preprocessing parameters
    image_size: tuple[int, int] = (96, 96)
    context_padding: float = 0.15
    min_box_size: int = 16
    max_crops_per_class_per_image: int = 5
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15

    # Training parameters
    seed: int = 42
    batch_size: int = 32
    num_epochs: int = 25
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 5
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = Config()
