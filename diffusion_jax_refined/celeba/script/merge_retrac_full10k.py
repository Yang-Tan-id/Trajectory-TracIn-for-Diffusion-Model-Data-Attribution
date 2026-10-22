#!/usr/bin/env python3
"""Merge original and complementary CelebA ReTrac score banks into full 10k."""

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


def load_bank(path: Path) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(np.load(path / "scores.npy"), dtype=np.float64).reshape(-1)
    indices = np.asarray(np.load(path / "score_indices.npy"), dtype=np.int64).reshape(-1)
    if scores.shape != indices.shape:
        raise ValueError(f"score/index shape mismatch: {path}")
    if len(scores) != 5000 or len(np.unique(indices)) != 5000:
        raise ValueError(f"expected 5000 unique candidates: {path}")
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"non-finite score: {path}")
    return scores, indices


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--query-ids", default=",".join(str(value) for value in range(72)))
    parser.add_argument("--experiment", default="attr40_experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--supplement-suffix", default="_remaining5000")
    parser.add_argument("--full-suffix", default="_full10000")
    parser.add_argument("--replay-tag", default="train_replay4_batch1")
    args = parser.parse_args()

    queries = load_queries(args.query_file)
    score_root = (
        CELEBA_ROOT
        / "result"
        / args.experiment
        / "attribution_score"
        / "prompted_solo"
        / f"train_seed_{args.train_seed}"
    )
    full_universe = np.arange(10_000, dtype=np.int64)
    score_directory = "score_query_train_l2_normalized"

    for query_id in parse_ids(args.query_ids):
        record = queries[query_id]
        query_root = (
            score_root
            / f"query_{_prompt_path_tag(str(record['prompt']))}"
            / f"initial_seed_{query_seed(record)}"
        )
        for transform in ("raw_gradient", "adamw_full"):
            namespace = (
                f"celeba_paper_retrac_{transform}_{args.replay_tag}_"
                "query_mean50_of_full_l2_linear_shape3d_semantics"
            )
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
                raise ValueError(f"Q{query_id} {transform}: halves overlap by {len(overlap)}")
            indices = np.concatenate((base_indices, supplement_indices))
            scores = np.concatenate((base_scores, supplement_scores))
            order = np.argsort(indices, kind="stable")
            indices = indices[order]
            scores = scores[order]
            if not np.array_equal(indices, full_universe):
                raise ValueError(f"Q{query_id} {transform}: union is not exactly 0..9999")
            atomic_npy(output / "scores.npy", scores)
            atomic_npy(output / "score_indices.npy", indices)
            atomic_json(
                output / "manifest.json",
                {
                    "namespace": f"{namespace}{args.full_suffix}",
                    "algorithm": "paper_retrac",
                    "transform": transform,
                    "score_variant": "query_train_l2",
                    "candidate_count": 10_000,
                    "candidate_indices": "exactly 0..9999",
                    "base_score_directory": str(base.resolve()),
                    "supplement_score_directory": str(supplement.resolve()),
                    "ranking_semantics": "stored score ascending",
                },
            )
            print(
                f"[merged] Q{query_id} {transform} candidates=10000 "
                f"score_min={scores.min():.9g} score_max={scores.max():.9g}",
                flush=True,
            )
    print("[done] ReTrac full-10000 score banks are ready", flush=True)


if __name__ == "__main__":
    main()
