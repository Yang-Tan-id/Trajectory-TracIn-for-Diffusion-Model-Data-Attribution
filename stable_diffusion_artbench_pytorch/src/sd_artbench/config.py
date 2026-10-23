from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ArtBenchConfig:
    """Paper-faithful defaults for the D-TRAK ArtBench-2 experiment."""

    metadata_csv: Path = PACKAGE_ROOT / "dataset" / "metadata" / "ArtBench-10.csv"
    image_root: Path = PACKAGE_ROOT / "dataset" / "images" / "artbench-10-imagefolder"
    index_root: Path = PACKAGE_ROOT / "dataset" / "indices"
    classes: Tuple[str, str] = ("post_impressionism", "ukiyo_e")
    split_seed: int = 42
    train_size: int = 5000
    validation_size: int = 1000
    lds_subset_fraction: float = 0.5
    lds_subset_count: int = 64


def label_to_prompt(label: str) -> str:
    return f"a {label.replace('_', ' ').lower()} painting"

