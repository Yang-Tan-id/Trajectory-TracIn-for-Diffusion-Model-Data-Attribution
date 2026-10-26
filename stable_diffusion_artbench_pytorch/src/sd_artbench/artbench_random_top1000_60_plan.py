"""Plan, validate, and audit ArtBench random top-1000 removal Q0-Q59."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np

from .config import ArtBenchConfig
from .topk_removal import RANDOM_METHOD, random_scores, select_topk


QUERY_IDS = tuple(range(60))
TASK_COUNT = len(QUERY_IDS)


def build_tasks(
    *, shard_index: int | None = None, shard_count: int | None = None
) -> list[tuple[str, int]]:
    if (shard_index is None) != (shard_count is None):
        raise ValueError("shard_index and shard_count must be provided together")
    if shard_count is not None:
        if shard_count < 1:
            raise ValueError("shard_count must be positive")
        if shard_index is None or shard_index < 0 or shard_index >= shard_count:
            raise ValueError("shard_index must be in [0, shard_count)")
    tasks = [(RANDOM_METHOD, query_index) for query_index in QUERY_IDS]
    if shard_count is not None:
        tasks = [
            task
            for position, task in enumerate(tasks)
            if position % shard_count == shard_index
        ]
    return tasks


def validate_inputs(top_k: int = 1000, random_seed: int = 2026) -> None:
    config = ArtBenchConfig()
    base_path = config.index_root / "idx-train.pkl"
    with base_path.open("rb") as handle:
        base_indices = np.asarray(pickle.load(handle), dtype=np.int64)
    if base_indices.shape != (5000,) or len(np.unique(base_indices)) != 5000:
        raise ValueError("idx-train.pkl must contain exactly 5000 unique indices")
    with config.metadata_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    missing_images: list[Path] = []
    for metadata_index in base_indices:
        try:
            row = rows[int(metadata_index)]
        except IndexError as error:
            raise ValueError(
                f"metadata index {metadata_index} is outside {config.metadata_csv}"
            ) from error
        image_path = config.image_root / row["label"] / row["name"]
        if not image_path.is_file() or image_path.stat().st_size == 0:
            missing_images.append(image_path)
    if missing_images:
        preview = "\n".join(map(str, missing_images[:10]))
        raise FileNotFoundError(
            f"{len(missing_images)} staged training images missing; first files:\n{preview}"
        )
    for query_index in QUERY_IDS:
        scores, indices, sign, _ = random_scores(
            query_index=query_index,
            base_indices=base_indices,
            random_seed=random_seed,
        )
        removed, _, _ = select_topk(
            scores, indices, top_k=top_k, ranking_sign=sign
        )
        if removed.shape != (top_k,) or len(np.unique(removed)) != top_k:
            raise AssertionError(f"invalid random selection for Q{query_index}")
    print(
        "[validated] random Q0-Q59; metadata + 5000 images; top-1000; "
        f"deterministic SeedSequence([{random_seed}, query])",
        flush=True,
    )


def audit(
    output_root: Path,
    *,
    top_k: int = 1000,
    shard_index: int | None = None,
    shard_count: int | None = None,
) -> None:
    failures: list[dict[str, object]] = []
    tasks = build_tasks(shard_index=shard_index, shard_count=shard_count)
    for method, query_index in tasks:
        run_dir = output_root / method / f"query-{query_index:03d}" / f"topk-{top_k:04d}"
        required = (
            run_dir / "TRAINING_COMPLETE",
            run_dir / "removal-manifest.json",
            run_dir / "retained-indices.pkl",
        )
        missing = [
            str(path)
            for path in required
            if not path.is_file() or path.stat().st_size == 0
        ]
        weights = (
            run_dir / "model/final/pytorch_lora_weights.bin",
            run_dir / "model/final/pytorch_lora_weights.safetensors",
        )
        if not any(path.is_file() and path.stat().st_size > 0 for path in weights):
            missing.append("final LoRA weights")
        if missing:
            failures.append(
                {"method": method, "query_index": query_index, "missing": missing}
            )
            continue
        payload = json.loads((run_dir / "removal-manifest.json").read_text())
        expected_seed = [2026, query_index]
        if (
            payload.get("status") != "trained"
            or payload.get("top_k") != top_k
            or payload.get("method") != RANDOM_METHOD
            or payload.get("random_seed") != expected_seed[0]
        ):
            failures.append(
                {
                    "method": method,
                    "query_index": query_index,
                    "invalid_manifest": payload,
                }
            )
    if failures:
        preview = json.dumps(failures[:3], indent=2)
        raise RuntimeError(
            f"{len(failures)} incomplete/invalid random tasks; first failures:\n{preview}"
        )
    label = "all" if shard_index is None else f"shard {shard_index}/{shard_count}"
    print(f"[audit complete] {len(tasks)}/{len(tasks)} random models ({label})", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("tasks", "validate", "audit"), required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "tasks":
        for method, query_index in build_tasks(
            shard_index=args.shard_index, shard_count=args.shard_count
        ):
            print(f"{method}\t{query_index}")
    elif args.mode == "validate":
        validate_inputs(top_k=args.top_k, random_seed=args.random_seed)
    else:
        if args.output_root is None:
            raise ValueError("--output-root is required for audit mode")
        audit(
            args.output_root,
            top_k=args.top_k,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )


if __name__ == "__main__":
    main()
