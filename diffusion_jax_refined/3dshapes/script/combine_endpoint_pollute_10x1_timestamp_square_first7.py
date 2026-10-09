#!/usr/bin/env python3
"""Sum the first seven cached per-timestamp square score vectors."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SHAPES_ROOT.parents[1]
for root in (SHAPES_ROOT, REPO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from dataset_config import _prompt_tag  # noqa: E402
from diffusion_jax_refined.common.stage_artifact_runner import (  # noqa: E402
    _write_score_outputs,
)


TIMESTAMPS = (0, 111, 222, 333, 444, 555, 666)
VARIANT_DIRS = {
    "raw": "score",
    "query_l2": "score_query_normalized",
    "train_l2": "score_train_l2_normalized",
    "query_train_l2": "score_query_train_l2_normalized",
}
OUTPUT_NAMESPACE = (
    "traj_tracin_recreate_adamw_full_polluted_endpoint_delta_l2normalized_"
    "timestamp_aware_square_first7_q0_99"
)


def parse_ids(text: str) -> list[int]:
    return [int(token) for token in text.replace(",", " ").split()]


def query_score_base(
    result_root: Path, train_seed: int, prompt: str, seed: int
) -> Path:
    return (
        result_root
        / "attribution_score"
        / "prompted_solo"
        / f"train_seed_{train_seed}"
        / f"query_{_prompt_tag(prompt)}"
        / f"initial_seed_{seed}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--query-ids", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    records = json.loads(args.query_file.read_text())["queries"]
    query_ids = parse_ids(args.query_ids)
    result_root = SHAPES_ROOT / "result" / args.experiment
    tasks = [
        (query_id, variant, directory)
        for query_id in query_ids
        for variant, directory in VARIANT_DIRS.items()
    ]

    def combine(task: tuple[int, str, str]) -> tuple[int, str, bool]:
        query_id, variant, directory = task
        record = records[query_id]
        prompt = str(record["prompt"])
        seed = int(record.get("initial_seed", record.get("seed")))
        base = query_score_base(
            result_root, args.train_seed, prompt, seed
        )
        output_dir = base / OUTPUT_NAMESPACE / directory
        required = (
            output_dir / "scores.npy",
            output_dir / "score_indices.npy",
            output_dir / "score_artifact_manifest.json",
        )
        if all(path.is_file() for path in required):
            return query_id, variant, True

        total: np.ndarray | None = None
        indices: np.ndarray | None = None
        sources = []
        for timestep in TIMESTAMPS:
            source = (
                base
                / (
                    "traj_tracin_recreate_adamw_full_polluted_endpoint_"
                    "delta_l2normalized_timestamp_aware_square_"
                    f"t{timestep:03d}_q0_99"
                )
                / directory
            )
            current_scores = np.load(source / "scores.npy").astype(
                np.float64, copy=False
            )
            current_indices = np.load(source / "score_indices.npy").astype(
                np.int64, copy=False
            )
            if total is None:
                total = np.zeros_like(current_scores, dtype=np.float64)
                indices = current_indices
            elif not np.array_equal(indices, current_indices):
                raise ValueError(
                    f"Q{query_id} {variant}: score-index mismatch at t={timestep}"
                )
            total += current_scores
            sources.append(str(source))

        assert total is not None and indices is not None
        _write_score_outputs(
            output_dir,
            total,
            indices,
            train_dir=result_root
            / "model"
            / "prompted_solo"
            / f"seed_{args.train_seed}_train_gradient"
            / "traj_tracin_recreate_adamw_dual_mc1_aligned10x1",
            query_dir=Path(sources[0]),
            algorithm="traj_tracin",
            extra_manifest={
                "formula": "sum_first7(per_timestamp_checkpoint_sum_squared)",
                "source_timestamps": list(TIMESTAMPS),
                "timestamp_count": len(TIMESTAMPS),
                "score_variant": variant,
                "query_id": query_id,
                "source_score_dirs": sources,
            },
        )
        return query_id, variant, False

    skipped = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for completed, (query_id, variant, was_skipped) in enumerate(
            executor.map(combine, tasks), start=1
        ):
            skipped += int(was_skipped)
            if completed % 20 == 0 or completed == len(tasks):
                print(
                    f"[combine] {completed}/{len(tasks)} "
                    f"latest=Q{query_id}:{variant} skipped={skipped}",
                    flush=True,
                )
    print(
        f"[done] first7 combined scores: {len(tasks)} query/variant outputs; "
        f"skipped={skipped}",
        flush=True,
    )


if __name__ == "__main__":
    main()
