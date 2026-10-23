import json
import pickle
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ArtBenchConfig, label_to_prompt


REQUIRED_COLUMNS = {"name", "label", "split"}


def load_metadata(config: ArtBenchConfig) -> pd.DataFrame:
    if not config.metadata_csv.is_file():
        raise FileNotFoundError(
            f"Missing metadata CSV: {config.metadata_csv}. "
            "See dataset/README.md for the expected layout."
        )
    frame = pd.read_csv(config.metadata_csv)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Metadata CSV is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["image_path"] = frame.apply(
        lambda row: str(config.image_root / row["label"] / row["name"]), axis=1
    )
    frame["prompt"] = frame["label"].map(label_to_prompt)
    return frame


def make_artbench2_splits(
    frame: pd.DataFrame, config: ArtBenchConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    selected = frame[frame["label"].isin(config.classes)]
    train_pool = selected[selected["split"] == "train"]
    validation_pool = selected[selected["split"] == "test"]

    if len(train_pool) < config.train_size:
        raise ValueError(
            f"ArtBench-2 train pool has {len(train_pool)} rows; "
            f"need at least {config.train_size}."
        )
    if len(validation_pool) < config.validation_size:
        raise ValueError(
            f"ArtBench-2 test pool has {len(validation_pool)} rows; "
            f"need at least {config.validation_size}."
        )

    train, _ = train_test_split(
        train_pool,
        train_size=config.train_size,
        random_state=config.split_seed,
        stratify=train_pool["label"],
    )
    validation, _ = train_test_split(
        validation_pool,
        train_size=config.validation_size,
        random_state=config.split_seed,
        stratify=validation_pool["label"],
    )
    return train, validation


def official_lds_seed(base_seed: int, subset_index: int) -> int:
    """Match the first LDS loop in the official ArtBench2/00_EDA.ipynb.

    The notebook increments ``count`` while also adding ``k``, producing
    seeds 42, 44, 46, ... rather than 42, 43, 44, ....
    """

    return base_seed + 2 * subset_index


def make_lds_subsets(
    train: pd.DataFrame, config: ArtBenchConfig
) -> Iterable[Tuple[int, int, pd.DataFrame]]:
    for subset_index in range(config.lds_subset_count):
        seed = official_lds_seed(config.split_seed, subset_index)
        subset, _ = train_test_split(
            train,
            train_size=config.lds_subset_fraction,
            random_state=seed,
            stratify=train["label"],
        )
        yield subset_index, seed, subset


def save_pickle(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def validate_image_paths(frame: pd.DataFrame) -> None:
    missing = [path for path in frame["image_path"] if not Path(path).is_file()]
    if missing:
        preview = "\n".join(missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} selected images are missing. First missing paths:\n{preview}"
        )


def write_prepared_data(
    frame: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    config: ArtBenchConfig,
) -> None:
    config.index_root.mkdir(parents=True, exist_ok=True)
    manifest_root = config.index_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)

    save_pickle(config.index_root / "idx-train.pkl", train.index.tolist())
    save_pickle(config.index_root / "idx-val.pkl", validation.index.tolist())
    train.to_csv(manifest_root / "train.csv", index_label="source_index")
    validation.to_csv(manifest_root / "validation.csv", index_label="source_index")

    lds_seeds = {}
    for subset_index, seed, subset in make_lds_subsets(train, config):
        save_pickle(
            config.index_root / "lds-val" / f"sub-idx-{subset_index}.pkl",
            subset.index.tolist(),
        )
        lds_seeds[str(subset_index)] = seed

    summary = {
        "classes": list(config.classes),
        "candidate_train_rows": int(
            len(frame[(frame["label"].isin(config.classes)) & (frame["split"] == "train")])
        ),
        "selected_train_rows": len(train),
        "selected_validation_rows": len(validation),
        "train_class_counts": train["label"].value_counts().sort_index().to_dict(),
        "validation_class_counts": validation["label"].value_counts().sort_index().to_dict(),
        "split_seed": config.split_seed,
        "lds_subset_fraction": config.lds_subset_fraction,
        "lds_subset_count": config.lds_subset_count,
        "lds_seeds": lds_seeds,
    }
    (config.index_root / "split_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

