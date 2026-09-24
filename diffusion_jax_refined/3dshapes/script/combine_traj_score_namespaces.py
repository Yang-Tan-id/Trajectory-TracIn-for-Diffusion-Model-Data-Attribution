#!/usr/bin/env python3
"""Combine two compatible Traj TracIn score namespaces term-grid-wise.

The intended use is to combine two equally sized uniform timestamp grids.  A
weight of 0.5 for each 10-timestamp score therefore produces the uniform
20-timestamp linear score without loading either train-gradient artifact.
"""

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


COMPONENTS = (
    "score",
    "score_query_normalized",
    "score_train_l2_normalized",
    "score_query_train_l2_normalized",
)


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value.strip()]


def load_score(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    score_path = directory / "scores.npy"
    index_path = directory / "score_indices.npy"
    if not score_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(f"Missing score artifact in {directory}")
    scores = np.asarray(np.load(score_path), dtype=np.float64).reshape(-1)
    indices = np.asarray(np.load(index_path), dtype=np.int64).reshape(-1)
    if scores.shape != indices.shape:
        raise ValueError(
            f"Score/index shape mismatch in {directory}: {scores.shape} vs {indices.shape}"
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"Non-finite scores in {score_path}")
    return scores, indices


def write_score(
    directory: Path,
    scores: np.ndarray,
    indices: np.ndarray,
    *,
    source_a: Path,
    source_b: Path,
    weight_a: float,
    weight_b: float,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "scores.npy", np.asarray(scores, dtype=np.float64))
    np.save(directory / "score_indices.npy", np.asarray(indices, dtype=np.int64))
    order = np.argsort(-scores, kind="stable")
    top = [
        {
            "rank": rank,
            "idx": int(indices[position]),
            "idx_1based": int(indices[position]) + 1,
            "score": float(scores[position]),
        }
        for rank, position in enumerate(order[: min(2000, len(order))], start=1)
    ]
    (directory / "top_scores.json").write_text(
        json.dumps({"top": top, "num_scored": int(len(scores))}, indent=2)
    )
    (directory / "score_indices.json").write_text(
        json.dumps(
            {
                "score_indices": [int(value) for value in indices],
                "score_indices_1based": [int(value) + 1 for value in indices],
            },
            indent=2,
        )
    )
    (directory / "score_artifact_manifest.json").write_text(
        json.dumps(
            {
                "algorithm": "weighted_linear_score_namespace_combination",
                "source_a": str(source_a),
                "source_b": str(source_b),
                "weight_a": weight_a,
                "weight_b": weight_b,
                "num_scores": int(len(scores)),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--query-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--namespace-a", required=True)
    parser.add_argument("--namespace-b", required=True)
    parser.add_argument("--output-namespace", required=True)
    parser.add_argument("--weight-a", type=float, default=0.5)
    parser.add_argument("--weight-b", type=float, default=0.5)
    args = parser.parse_args()

    if not np.isfinite(args.weight_a) or not np.isfinite(args.weight_b):
        parser.error("weights must be finite")
    if not np.isclose(args.weight_a + args.weight_b, 1.0, rtol=0.0, atol=1e-12):
        parser.error("weights must sum to one")

    records = json.loads(args.query_file.read_text())["queries"]
    query_ids = parse_ints(args.query_ids)
    result_root = SHAPES_ROOT / "result" / args.experiment / "attribution_score"
    completed = 0
    for query_id in query_ids:
        if query_id < 0 or query_id >= len(records):
            raise ValueError(f"query id {query_id} is outside the manifest")
        record = records[query_id]
        query_root = (
            result_root
            / "prompted_solo"
            / f"train_seed_{args.train_seed}"
            / f"query_{_prompt_tag(str(record['prompt']))}"
            / f"initial_seed_{int(record['initial_seed'])}"
        )
        root_a = query_root / f"traj_tracin_{args.namespace_a}"
        root_b = query_root / f"traj_tracin_{args.namespace_b}"
        root_out = query_root / f"traj_tracin_{args.output_namespace}"
        for component in COMPONENTS:
            directory_a = root_a / component
            directory_b = root_b / component
            scores_a, indices_a = load_score(directory_a)
            scores_b, indices_b = load_score(directory_b)
            if not np.array_equal(indices_a, indices_b):
                raise ValueError(
                    f"Score indices differ for query={query_id}, component={component}"
                )
            combined = args.weight_a * scores_a + args.weight_b * scores_b
            write_score(
                root_out / component,
                combined,
                indices_a,
                source_a=directory_a,
                source_b=directory_b,
                weight_a=args.weight_a,
                weight_b=args.weight_b,
            )
            completed += 1
            print(
                f"[{completed}/{len(query_ids) * len(COMPONENTS)}] "
                f"query={query_id} component={component}",
                flush=True,
            )
    print(
        f"Combined {completed} score artifacts into traj_tracin_{args.output_namespace}",
        flush=True,
    )


if __name__ == "__main__":
    main()
