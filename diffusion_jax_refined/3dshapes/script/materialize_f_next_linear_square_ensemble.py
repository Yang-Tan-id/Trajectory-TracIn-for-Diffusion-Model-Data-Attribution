#!/usr/bin/env python3
"""Materialize per-query z-normalized linear/squared f-next score ensembles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT))

from dataset_config import _prompt_tag


LINEAR_NAMESPACE = "traj_tracin"
SQUARED_NAMESPACE = "traj_tracin_f_next_dot_squared_original_f"
OUTPUT_NAMESPACE = "traj_tracin_f_next_linear_square_z50"
VARIANTS = (
    ("raw", "score"),
    ("query_l2", "score_query_normalized"),
)


def records() -> list[dict]:
    return json.loads((SHAPES_ROOT / "queries_seed_0_9.json").read_text())["queries"]


def atomic_save(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, values)
    tmp.replace(path)


def zscore(values: np.ndarray, eps: float) -> tuple[np.ndarray, float, float]:
    mean = float(np.mean(values, dtype=np.float64))
    std = float(np.std(values, dtype=np.float64))
    if not np.isfinite(std) or std <= eps:
        raise ValueError(f"score standard deviation is too small: {std}")
    return (values.astype(np.float64) - mean) / std, mean, std


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--eps", type=float, default=1e-12)
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0,1]")

    root = SHAPES_ROOT / "result" / args.experiment / "attribution_score" / "prompted_solo"
    for query_id, record in enumerate(records()):
        query_root = (
            root
            / f"train_seed_{args.train_seed}"
            / f"query_{_prompt_tag(str(record['prompt']))}"
            / f"initial_seed_{int(record['initial_seed'])}"
        )
        for variant, component in VARIANTS:
            linear_dir = query_root / LINEAR_NAMESPACE / component
            squared_dir = query_root / SQUARED_NAMESPACE / component
            output_dir = query_root / OUTPUT_NAMESPACE / component

            linear_scores = np.asarray(np.load(linear_dir / "scores.npy"), dtype=np.float64)
            squared_scores = np.asarray(np.load(squared_dir / "scores.npy"), dtype=np.float64)
            linear_indices = np.asarray(np.load(linear_dir / "score_indices.npy"), dtype=np.int64)
            squared_indices = np.asarray(np.load(squared_dir / "score_indices.npy"), dtype=np.int64)
            if not np.array_equal(linear_indices, squared_indices):
                raise ValueError(f"score index mismatch for query={query_id} variant={variant}")
            if linear_scores.shape != squared_scores.shape:
                raise ValueError(f"score shape mismatch for query={query_id} variant={variant}")

            linear_z, linear_mean, linear_std = zscore(linear_scores, args.eps)
            squared_z, squared_mean, squared_std = zscore(squared_scores, args.eps)
            ensemble = args.alpha * linear_z + (1.0 - args.alpha) * squared_z

            atomic_save(output_dir / "scores.npy", ensemble)
            atomic_save(output_dir / "score_indices.npy", linear_indices)
            manifest = {
                "algorithm": OUTPUT_NAMESPACE,
                "score_variant": variant,
                "definition": "alpha*zscore(linear_f_next)+(1-alpha)*zscore(squared_f_next)",
                "alpha_linear": args.alpha,
                "alpha_squared": 1.0 - args.alpha,
                "normalization_scope": "within_query_across_attribution_datapoints",
                "linear_source": str(linear_dir),
                "squared_source": str(squared_dir),
                "linear_mean": linear_mean,
                "linear_std": linear_std,
                "squared_mean": squared_mean,
                "squared_std": squared_std,
            }
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "score_artifact_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True)
            )
            print(
                f"[saved] query={query_id} variant={variant} output={output_dir}",
                flush=True,
            )

    print(f"[done] materialized {OUTPUT_NAMESPACE}", flush=True)


if __name__ == "__main__":
    main()
