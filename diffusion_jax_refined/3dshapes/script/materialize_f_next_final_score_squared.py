#!/usr/bin/env python3
"""Square each datapoint's fully accumulated original f-next TrajTracIn score."""

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


INPUT_NAMESPACE = "traj_tracin"
OUTPUT_NAMESPACE = "traj_tracin_f_next_final_score_squared"
VARIANTS = (
    ("raw", "score"),
    ("query_l2", "score_query_normalized"),
    ("train_l2", "score_train_l2_normalized"),
    ("query_train_l2", "score_query_train_l2_normalized"),
)


def records() -> list[dict]:
    return json.loads((SHAPES_ROOT / "queries_seed_0_9.json").read_text())["queries"]


def square_final_scores(values: np.ndarray) -> np.ndarray:
    return np.square(np.asarray(values, dtype=np.float64))


def atomic_save(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, values)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    args = parser.parse_args()

    root = SHAPES_ROOT / "result" / args.experiment / "attribution_score" / "prompted_solo"
    for query_id, record in enumerate(records()):
        query_root = (
            root
            / f"train_seed_{args.train_seed}"
            / f"query_{_prompt_tag(str(record['prompt']))}"
            / f"initial_seed_{int(record['initial_seed'])}"
        )
        for variant, component in VARIANTS:
            input_dir = query_root / INPUT_NAMESPACE / component
            output_dir = query_root / OUTPUT_NAMESPACE / component
            scores = np.asarray(np.load(input_dir / "scores.npy"), dtype=np.float64)
            indices = np.asarray(np.load(input_dir / "score_indices.npy"), dtype=np.int64)
            if scores.ndim != 1 or indices.ndim != 1 or scores.shape != indices.shape:
                raise ValueError(
                    f"invalid score/index shapes for query={query_id} variant={variant}: "
                    f"{scores.shape}, {indices.shape}"
                )
            atomic_save(output_dir / "scores.npy", square_final_scores(scores))
            atomic_save(output_dir / "score_indices.npy", indices)
            manifest = {
                "algorithm": OUTPUT_NAMESPACE,
                "score_variant": variant,
                "definition": "square(fully_accumulated_original_f_next_traj_tracin_score)",
                "source": str(input_dir),
                "square_position": "after_checkpoint_and_snapshot_accumulation",
                "learning_rate_semantics": (
                    "S_i=sum_terms(linear_learning_rate_weight*contraction); output=S_i^2"
                ),
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
