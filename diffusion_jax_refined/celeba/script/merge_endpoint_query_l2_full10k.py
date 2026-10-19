#!/usr/bin/env python3
"""Merge all original/complementary CelebA endpoint score banks into full 10k."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


CELEBA_ROOT = Path(__file__).resolve().parents[1]
if str(CELEBA_ROOT) not in sys.path:
    sys.path.insert(0, str(CELEBA_ROOT))

from dataset_config_attr40 import _prompt_path_tag


VARIANTS = (
    ("score", "raw"),
    ("score_query_normalized", "query_l2"),
    ("score_train_l2_normalized", "train_l2"),
    ("score_query_train_l2_normalized", "query_train_l2"),
)


def score_namespaces() -> list[str]:
    namespaces: list[str] = []
    for transform in ("raw_gradient", "adamw_full"):
        for reduction in ("linear", "termwise_squared", "timestamp_sum_squared"):
            namespaces.append(
                f"celeba_endpoint_pollute_{transform}_train1x50_query1x50_"
                f"{reduction}_shape3d_semantics"
            )
        namespaces.append(
            f"celeba_endpoint_tracin_{transform}_"
            "train_mean50_query_mean50_linear_shape3d_semantics"
        )
    return namespaces


def parse_ids(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value]


def load_queries(path: Path) -> dict[int, dict]:
    payload = json.loads(path.read_text())
    records = payload["queries"] if isinstance(payload, dict) else payload
    return {
        int(record.get("query_id", record.get("id", position))): record
        for position, record in enumerate(records)
    }


def query_seed(record: dict) -> int:
    return int(record.get("initial_seed", record.get("seed")))


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    np.save(temporary, value)
    temporary.replace(path)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def load_bank(path: Path) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(np.load(path / "scores.npy"), dtype=np.float64).reshape(-1)
    indices = np.asarray(np.load(path / "score_indices.npy"), dtype=np.int64).reshape(-1)
    if scores.shape != indices.shape:
        raise ValueError(f"score/index mismatch under {path}")
    if len(scores) != 5000 or len(np.unique(indices)) != 5000:
        raise ValueError(f"expected 5000 unique candidates under {path}")
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"non-finite scores under {path}")
    return scores, indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--query-ids", default=",".join(str(value) for value in range(72)))
    parser.add_argument("--experiment", default="attr40_experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--supplement-suffix", default="_remaining5000")
    parser.add_argument("--full-suffix", default="_full10000")
    args = parser.parse_args()

    queries = load_queries(args.query_file)
    result_root = CELEBA_ROOT / "result" / args.experiment
    attribution_root = (
        result_root
        / "attribution_score"
        / "prompted_solo"
        / f"train_seed_{args.train_seed}"
    )
    full_universe = np.arange(10_000, dtype=np.int64)

    for query_id in parse_ids(args.query_ids):
        record = queries[query_id]
        query_root = (
            attribution_root
            / f"query_{_prompt_path_tag(str(record['prompt']))}"
            / f"initial_seed_{query_seed(record)}"
        )
        merged_count = 0
        for namespace in score_namespaces():
            for score_directory, variant in VARIANTS:
                base = query_root / f"traj_tracin_{namespace}" / score_directory
                supplement = (
                    query_root
                    / f"traj_tracin_{namespace}{args.supplement_suffix}"
                    / score_directory
                )
                output = (
                    query_root
                    / f"traj_tracin_{namespace}{args.full_suffix}"
                    / score_directory
                )
                base_scores, base_indices = load_bank(base)
                supplement_scores, supplement_indices = load_bank(supplement)
                overlap = np.intersect1d(base_indices, supplement_indices)
                if len(overlap):
                    raise ValueError(
                        f"Q{query_id} {namespace}/{variant}: candidate halves "
                        f"overlap ({len(overlap)})"
                    )
                indices = np.concatenate((base_indices, supplement_indices))
                scores = np.concatenate((base_scores, supplement_scores))
                order = np.argsort(indices, kind="stable")
                indices = indices[order]
                scores = scores[order]
                if not np.array_equal(indices, full_universe):
                    raise ValueError(
                        f"Q{query_id} {namespace}/{variant}: union is not exactly 0..9999"
                    )
                atomic_npy(output / "scores.npy", scores)
                atomic_npy(output / "score_indices.npy", indices)
                atomic_json(
                    output / "manifest.json",
                    {
                        "namespace": f"{namespace}{args.full_suffix}",
                        "score_variant": variant,
                        "score_directory": score_directory,
                        "candidate_count": 10_000,
                        "candidate_indices": "exactly 0..9999",
                        "base_score_directory": str(base.resolve()),
                        "supplement_score_directory": str(supplement.resolve()),
                        "merge_semantics": (
                            "concatenate disjoint 5k banks and sort by dataset index"
                        ),
                    },
                )
                merged_count += 1
                print(
                    f"[merged] Q{query_id} {namespace}{args.full_suffix}/{variant} "
                    f"candidates={len(indices)} score_min={scores.min():.9g} "
                    f"score_max={scores.max():.9g}",
                    flush=True,
                )
        print(f"[query complete] Q{query_id} full banks={merged_count}/32", flush=True)


if __name__ == "__main__":
    main()
