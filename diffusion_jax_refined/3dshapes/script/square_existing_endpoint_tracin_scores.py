#!/usr/bin/env python3
"""Square final per-datapoint endpoint-TracIn scores from an existing run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
REFINE_ROOT = SHAPES_ROOT.parent
for path in (SHAPES_ROOT, REFINE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.stage_artifact_runner import _write_score_outputs
from dataset_config import _prompt_tag


VARIANTS = (
    "score",
    "score_query_normalized",
    "score_train_l2_normalized",
    "score_query_train_l2_normalized",
)


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--query-ids", default=",".join(map(str, range(100))))
    parser.add_argument(
        "--source-namespace",
        default="endpoint_tracin_adamw_full_train10x10_query100x1_q0_99",
    )
    parser.add_argument(
        "--output-namespace",
        default="endpoint_tracin_adamw_full_train10x10_query100x1_final_score_squared_q0_99",
    )
    args = parser.parse_args()

    queries = json.loads(args.query_file.read_text())["queries"]
    score_root = (
        SHAPES_ROOT
        / "result"
        / args.experiment
        / "attribution_score"
        / "prompted_solo"
        / f"train_seed_{args.train_seed}"
    )
    completed = 0
    for query_id in parse_ints(args.query_ids):
        query = queries[query_id]
        seed = int(query.get("initial_seed", query.get("seed")))
        query_root = (
            score_root
            / f"query_{_prompt_tag(str(query['prompt']))}"
            / f"initial_seed_{seed}"
        )
        source_root = query_root / f"traj_tracin_{args.source_namespace}"
        output_root = query_root / f"traj_tracin_{args.output_namespace}"
        for variant in VARIANTS:
            source = source_root / variant
            output = output_root / variant
            scores_path = source / "scores.npy"
            indices_path = source / "score_indices.npy"
            if not scores_path.is_file() or not indices_path.is_file():
                raise FileNotFoundError(
                    f"Q{query_id} {variant}: missing linear score artifact under {source}"
                )
            scores = np.asarray(np.load(scores_path), dtype=np.float64)
            indices = np.asarray(np.load(indices_path), dtype=np.int64)
            _write_score_outputs(
                output,
                np.square(scores),
                indices,
                train_dir=source,
                query_dir=source,
                algorithm="endpoint_tracin_final_datapoint_score_squared",
                extra_manifest={
                    "source_score_dir": str(source),
                    "source_score_reduction": "linear",
                    "score_reduction": "final_datapoint_score_squared",
                    "formula": "square(sum_of_weighted_linear_term_scores)",
                    "prediction_sign_for_loss_utility": -1,
                },
            )
        completed += 1
        print(f"[squared] Q{query_id} complete", flush=True)
    print(f"[done] squared final datapoint scores for {completed} queries", flush=True)


if __name__ == "__main__":
    main()
