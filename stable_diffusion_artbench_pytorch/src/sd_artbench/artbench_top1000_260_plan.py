"""Plan, stage, validate, and audit the ArtBench 260-model H100 run."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .config import ArtBenchConfig
from .topk_removal import (
    METHODS,
    RANDOM_METHOD,
    load_score_file,
    random_scores,
    select_topk,
)


ATTRIBUTION_QUERY_IDS = tuple(range(10, 60))
RANDOM_QUERY_IDS = tuple(range(60))
TASK_COUNT = len(METHODS) * len(ATTRIBUTION_QUERY_IDS) + len(RANDOM_QUERY_IDS)
SCORE_RELATIVE_PATHS = (
    "outputs/attribution/endpoint-q60-t50-full-linear-hooks/endpoint_pollute_timestamp_sum_squared.npz",
    "outputs/attribution/endpoint-q60-t50-full-linear-hooks/endpoint_tracin.npz",
    "outputs/attribution/retrac-q60-t50-full-linear-hooks/retrac.npz",
)


def build_tasks() -> list[tuple[str, int]]:
    tasks = [
        (method, query_index)
        for method in METHODS
        for query_index in ATTRIBUTION_QUERY_IDS
    ]
    tasks.extend((RANDOM_METHOD, query_index) for query_index in RANDOM_QUERY_IDS)
    if len(tasks) != TASK_COUNT or len(set(tasks)) != TASK_COUNT:
        raise AssertionError("task plan is not exactly 260 unique models")
    return tasks


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    return package_root().parent


def relative_to_repo(path: Path) -> Path:
    return path.resolve().relative_to(repo_root().resolve())


def transfer_manifest() -> list[Path]:
    config = ArtBenchConfig()
    with config.index_root.joinpath("idx-train.pkl").open("rb") as handle:
        indices = [int(value) for value in pickle.load(handle)]
    if len(indices) != 5000 or len(indices) != len(set(indices)):
        raise ValueError("idx-train.pkl must contain 5000 unique metadata indices")
    with config.metadata_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    files = [config.metadata_csv, config.index_root / "idx-train.pkl"]
    for metadata_index in indices:
        try:
            row = rows[metadata_index]
        except IndexError as error:
            raise ValueError(
                f"metadata index {metadata_index} is outside {config.metadata_csv}"
            ) from error
        files.append(config.image_root / row["label"] / row["name"])
    files.extend(package_root() / value for value in SCORE_RELATIVE_PATHS)
    relative = [relative_to_repo(path) for path in files]
    if len(relative) != 5005 or len(set(relative)) != 5005:
        raise AssertionError("transfer manifest must contain 5005 unique files")
    missing = [path for path in files if not path.is_file() or path.stat().st_size == 0]
    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} required transfer files are missing/empty; first files:\n{preview}"
        )
    return relative


def _extract_bank(payload: dict[str, Any], variant: str | None) -> np.ndarray:
    try:
        transform_slot = payload["transforms"].index("adamw_full")
    except ValueError as error:
        raise ValueError("score artifact lacks adamw_full") from error
    if variant is None:
        bank = payload["scores"][transform_slot]
    else:
        variants = payload["variants"]
        if variants is None or variant not in variants:
            raise ValueError(f"score artifact lacks variant {variant}")
        bank = payload["scores"][transform_slot, variants.index(variant)]
    bank = np.asarray(bank, dtype=np.float64)
    if bank.shape != (60, 5000):
        raise ValueError(f"expected score bank [60,5000], found {bank.shape}")
    if not np.all(np.isfinite(bank)):
        raise ValueError("score bank contains non-finite values")
    return bank


def validate_inputs(top_k: int = 1000, random_seed: int = 2026) -> None:
    config = ArtBenchConfig()
    base_path = config.index_root / "idx-train.pkl"
    with base_path.open("rb") as handle:
        base_indices = np.asarray(pickle.load(handle), dtype=np.int64)
    if base_indices.shape != (5000,) or len(np.unique(base_indices)) != 5000:
        raise ValueError("base split must contain exactly 5000 unique indices")

    endpoint_root = package_root() / "outputs/attribution/endpoint-q60-t50-full-linear-hooks"
    pollute = load_score_file(endpoint_root / "endpoint_pollute_timestamp_sum_squared.npz")
    tracin = load_score_file(endpoint_root / "endpoint_tracin.npz")
    retrac = load_score_file(
        package_root()
        / "outputs/attribution/retrac-q60-t50-full-linear-hooks/retrac.npz"
    )
    payloads = (pollute, tracin, retrac)
    for payload in payloads:
        indices = np.asarray(payload["indices"], dtype=np.int64)
        if indices.shape != (5000,) or set(indices.tolist()) != set(base_indices.tolist()):
            raise ValueError("score candidates do not exactly match idx-train.pkl")

    banks = {
        "endpoint_pollute_query_train_l2": (_extract_bank(pollute, "query_train_l2"), 1),
        "endpoint_tracin_raw": (_extract_bank(tracin, "raw"), -1),
        "endpoint_tracin_query_train_l2": (_extract_bank(tracin, "query_train_l2"), -1),
        "retrac_query_train_l2": (_extract_bank(retrac, None), -1),
    }
    for method, (bank, sign) in banks.items():
        for query_index in (0, 10, 59):
            removed, _, _ = select_topk(
                bank[query_index], base_indices, top_k=top_k, ranking_sign=sign
            )
            if removed.shape != (top_k,):
                raise AssertionError(f"invalid top-k selection for {method} Q{query_index}")

    for query_index in RANDOM_QUERY_IDS:
        scores, indices, sign, _ = random_scores(
            query_index=query_index,
            base_indices=base_indices,
            random_seed=random_seed,
        )
        removed, _, _ = select_topk(scores, indices, top_k=top_k, ranking_sign=sign)
        if removed.shape != (top_k,):
            raise AssertionError(f"invalid random selection for Q{query_index}")
    print(
        "[validated] three score files; four AdamW banks=[60,5000]; "
        "candidate universe=idx-train.pkl; random Q0-Q59 deterministic",
        flush=True,
    )


def audit(output_root: Path, top_k: int = 1000) -> None:
    failures = []
    for method, query_index in build_tasks():
        run_dir = output_root / method / f"query-{query_index:03d}" / f"topk-{top_k:04d}"
        required = (
            run_dir / "TRAINING_COMPLETE",
            run_dir / "removal-manifest.json",
            run_dir / "retained-indices.pkl",
        )
        missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
        weight_candidates = (
            run_dir / "model/final/pytorch_lora_weights.bin",
            run_dir / "model/final/pytorch_lora_weights.safetensors",
        )
        if not any(path.is_file() and path.stat().st_size > 0 for path in weight_candidates):
            missing.append(" or ".join(str(path) for path in weight_candidates))
        if missing:
            failures.append({"method": method, "query_index": query_index, "missing": missing})
            continue
        payload = json.loads((run_dir / "removal-manifest.json").read_text())
        if payload.get("status") != "trained" or payload.get("top_k") != top_k:
            failures.append(
                {"method": method, "query_index": query_index, "invalid_manifest": payload}
            )
    if failures:
        preview = json.dumps(failures[:3], indent=2)
        raise RuntimeError(f"{len(failures)} incomplete/invalid tasks; first failures:\n{preview}")
    print(f"[audit complete] {TASK_COUNT}/{TASK_COUNT} trained models", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("tasks", "transfer", "validate", "audit"), required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "tasks":
        for method, query_index in build_tasks():
            print(f"{method}\t{query_index}")
    elif args.mode == "transfer":
        for path in transfer_manifest():
            print(path.as_posix())
    elif args.mode == "validate":
        validate_inputs(top_k=args.top_k, random_seed=args.random_seed)
    else:
        if args.output_root is None:
            raise ValueError("--output-root is required for audit mode")
        audit(args.output_root, top_k=args.top_k)


if __name__ == "__main__":
    main()
