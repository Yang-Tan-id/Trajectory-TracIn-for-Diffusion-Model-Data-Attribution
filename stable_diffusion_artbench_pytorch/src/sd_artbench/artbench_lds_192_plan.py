"""Prepare, train, audit, and report 192 ArtBench LDS subset models."""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ArtBenchConfig


SUBSET_BANKS = (0, 1, 2)
SUBSETS_PER_BANK = 64
SUBSET_SIZE = 2500
TRAIN_SIZE = 5000
TASK_COUNT = len(SUBSET_BANKS) * SUBSETS_PER_BANK
TRAIN_SEED = 42


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_output_root() -> Path:
    return package_root() / "outputs" / "artbench2-lds"


def build_tasks() -> list[tuple[int, int]]:
    tasks = [
        (subset_bank, subset_id)
        for subset_bank in SUBSET_BANKS
        for subset_id in range(SUBSETS_PER_BANK)
    ]
    if len(tasks) != TASK_COUNT or len(set(tasks)) != TASK_COUNT:
        raise AssertionError("LDS plan must contain exactly 192 unique tasks")
    return tasks


def bank_subset_seeds(subset_bank: int) -> np.ndarray:
    if subset_bank not in SUBSET_BANKS:
        raise ValueError(f"subset bank must be one of {SUBSET_BANKS}")
    rng = np.random.default_rng(subset_bank)
    return rng.integers(
        0, np.iinfo(np.int32).max, size=SUBSETS_PER_BANK, dtype=np.int64
    )


def generate_subset_indices(
    base_indices: np.ndarray,
    labels: np.ndarray,
    *,
    subset_bank: int,
    subset_id: int,
) -> tuple[int, np.ndarray]:
    base_indices = np.asarray(base_indices, dtype=np.int64)
    labels = np.asarray(labels)
    if base_indices.shape != (TRAIN_SIZE,) or len(np.unique(base_indices)) != TRAIN_SIZE:
        raise ValueError("base indices must contain exactly 5000 unique values")
    if labels.shape != (TRAIN_SIZE,):
        raise ValueError("labels must align with the 5000 base indices")
    if subset_id < 0 or subset_id >= SUBSETS_PER_BANK:
        raise ValueError(f"subset id must be in [0,{SUBSETS_PER_BANK - 1}]")
    random_seed = int(bank_subset_seeds(subset_bank)[subset_id])
    kept, _ = train_test_split(
        base_indices,
        train_size=SUBSET_SIZE,
        random_state=random_seed,
        stratify=labels,
    )
    kept = np.sort(np.asarray(kept, dtype=np.int64))
    if kept.shape != (SUBSET_SIZE,) or len(np.unique(kept)) != SUBSET_SIZE:
        raise AssertionError("generated LDS subset is not exactly 2500 unique images")
    return random_seed, kept


def run_root(output_root: Path, subset_bank: int) -> Path:
    return (
        output_root
        / f"train_seed_{TRAIN_SEED}"
        / f"m_{SUBSETS_PER_BANK}_k_{SUBSET_SIZE}_subset_seed_{subset_bank}"
    )


def index_path(output_root: Path, subset_bank: int, subset_id: int) -> Path:
    return run_root(output_root, subset_bank) / "indices" / f"subset_{subset_id:04d}.pkl"


def model_dir(output_root: Path, subset_bank: int, subset_id: int) -> Path:
    return run_root(output_root, subset_bank) / "models" / f"subset_{subset_id:04d}"


def load_base_data() -> tuple[np.ndarray, np.ndarray]:
    config = ArtBenchConfig()
    with (config.index_root / "idx-train.pkl").open("rb") as handle:
        base_indices = np.asarray(pickle.load(handle), dtype=np.int64)
    frame = pd.read_csv(config.metadata_csv)
    if base_indices.shape != (TRAIN_SIZE,) or len(np.unique(base_indices)) != TRAIN_SIZE:
        raise ValueError("idx-train.pkl must contain exactly 5000 unique indices")
    if base_indices.min() < 0 or base_indices.max() >= len(frame):
        raise ValueError("idx-train.pkl contains indices outside ArtBench metadata")
    labels = frame.loc[base_indices, "label"].to_numpy()
    counts = pd.Series(labels).value_counts().to_dict()
    if counts != {"post_impressionism": 2500, "ukiyo_e": 2500}:
        raise ValueError(f"unexpected ArtBench-2 class balance: {counts}")
    return base_indices, labels


def validate_plan() -> None:
    base_indices, labels = load_base_data()
    label_by_index = dict(zip(base_indices.tolist(), labels.tolist(), strict=True))
    signatures: set[bytes] = set()
    for subset_bank, subset_id in build_tasks():
        _, kept = generate_subset_indices(
            base_indices,
            labels,
            subset_bank=subset_bank,
            subset_id=subset_id,
        )
        signature = kept.tobytes()
        if signature in signatures:
            raise ValueError(f"duplicate membership at bank={subset_bank} subset={subset_id}")
        signatures.add(signature)
        selected_labels = np.asarray([label_by_index[int(index)] for index in kept])
        counts = pd.Series(selected_labels).value_counts().to_dict()
        if counts != {"post_impressionism": 1250, "ukiyo_e": 1250}:
            raise ValueError(
                f"bank={subset_bank} subset={subset_id} is not stratified: {counts}"
            )
    print(
        "[validated] 3 subset banks x 64 subsets; 2500/5000 images per model; "
        "1250 images per class; 192 unique memberships",
        flush=True,
    )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def prepare_indices(output_root: Path) -> None:
    base_indices, labels = load_base_data()
    for subset_bank in SUBSET_BANKS:
        subsets = []
        for subset_id in range(SUBSETS_PER_BANK):
            random_seed, kept = generate_subset_indices(
                base_indices,
                labels,
                subset_bank=subset_bank,
                subset_id=subset_id,
            )
            path = index_path(output_root, subset_bank, subset_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".pkl.tmp")
            with temporary.open("wb") as handle:
                pickle.dump(kept.tolist(), handle)
            temporary.replace(path)
            subsets.append(
                {
                    "subset_id": subset_id,
                    "membership_seed": random_seed,
                    "index_path": str(path),
                    "size": len(kept),
                }
            )
        write_json(
            run_root(output_root, subset_bank) / "lds_model_config.json",
            {
                "dataset": "ArtBench-2",
                "candidate_count": TRAIN_SIZE,
                "m": SUBSETS_PER_BANK,
                "k": SUBSET_SIZE,
                "fraction": 0.5,
                "subset_seed": subset_bank,
                "model_train_seed": TRAIN_SEED,
                "stratified": True,
                "subsets": subsets,
            },
        )
    print(f"[prepared] 192 subset index files under {output_root}", flush=True)


def final_weights_exist(path: Path) -> bool:
    final_dir = path / "final"
    return final_dir.is_dir() and any(
        candidate.is_file() and candidate.stat().st_size > 0
        for candidate in final_dir.iterdir()
    )


def train_task(output_root: Path, subset_bank: int, subset_id: int) -> None:
    subset_path = index_path(output_root, subset_bank, subset_id)
    if not subset_path.is_file():
        raise FileNotFoundError(f"prepared subset missing: {subset_path}")
    output_dir = model_dir(output_root, subset_bank, subset_id)
    marker = output_dir / "TRAINING_COMPLETE"
    if marker.is_file() and final_weights_exist(output_dir):
        print(f"[skip] bank={subset_bank} subset={subset_id}: complete", flush=True)
        return
    command = [
        sys.executable,
        "-m",
        "sd_artbench.train_lora",
        "--index-path",
        str(subset_path),
        "--output-dir",
        str(output_dir),
        "--epochs",
        "100",
        "--batch-size",
        "8",
        "--gradient-accumulation-steps",
        "8",
        "--num-workers",
        "8",
        "--seed",
        str(TRAIN_SEED),
        "--mixed-precision",
        "fp16",
        "--num-checkpoints",
        "0",
    ]
    print("[train] " + " ".join(command), flush=True)
    subprocess.run(command, check=True)
    if not final_weights_exist(output_dir):
        raise FileNotFoundError(f"training did not create final LoRA weights: {output_dir}")
    write_json(
        output_dir / "training_manifest.json",
        {
            "dataset": "ArtBench-2",
            "subset_seed": subset_bank,
            "subset_id": subset_id,
            "subset_size": SUBSET_SIZE,
            "candidate_count": TRAIN_SIZE,
            "fraction": 0.5,
            "model_train_seed": TRAIN_SEED,
            "index_path": str(subset_path),
            "status": "trained",
        },
    )
    marker.write_text("complete\n", encoding="utf-8")
    print(f"[done] bank={subset_bank} subset={subset_id} output={output_dir}", flush=True)


def completed_tasks(output_root: Path) -> list[tuple[int, int]]:
    return [
        (subset_bank, subset_id)
        for subset_bank, subset_id in build_tasks()
        if (model_dir(output_root, subset_bank, subset_id) / "TRAINING_COMPLETE").is_file()
        and final_weights_exist(model_dir(output_root, subset_bank, subset_id))
    ]


def report_status(output_root: Path) -> None:
    complete = completed_tasks(output_root)
    print(f"COMPLETED={len(complete)}/{TASK_COUNT}")
    for subset_bank in SUBSET_BANKS:
        count = sum(bank == subset_bank for bank, _ in complete)
        print(f"subset_seed_{subset_bank}={count}/{SUBSETS_PER_BANK}")


def audit(output_root: Path) -> None:
    report_status(output_root)
    complete = set(completed_tasks(output_root))
    missing = [task for task in build_tasks() if task not in complete]
    if missing:
        raise RuntimeError(f"{len(missing)} LDS models incomplete; first={missing[:8]}")
    print("[audit complete] 192/192 ArtBench LDS models", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("tasks", "validate", "prepare", "train", "status", "audit"),
        required=True,
    )
    parser.add_argument("--output-root", type=Path, default=default_output_root())
    parser.add_argument("--subset-bank", type=int)
    parser.add_argument("--subset-id", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "tasks":
        for subset_bank, subset_id in build_tasks():
            print(f"{subset_bank}\t{subset_id}")
    elif args.mode == "validate":
        validate_plan()
    elif args.mode == "prepare":
        prepare_indices(args.output_root)
    elif args.mode == "train":
        if args.subset_bank is None or args.subset_id is None:
            raise ValueError("--subset-bank and --subset-id are required for train mode")
        train_task(args.output_root, args.subset_bank, args.subset_id)
    elif args.mode == "status":
        report_status(args.output_root)
    else:
        audit(args.output_root)


if __name__ == "__main__":
    main()
