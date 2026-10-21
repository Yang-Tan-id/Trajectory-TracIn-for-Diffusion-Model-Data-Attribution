#!/usr/bin/env python3
"""Compare historical 5k and merged full-10k CelebA endpoint top-k sets."""

from __future__ import annotations

import argparse
import csv
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
    ("raw", "score"),
    ("query_l2", "score_query_normalized"),
    ("train_l2", "score_train_l2_normalized"),
    ("query_train_l2", "score_query_train_l2_normalized"),
)


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


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fields = list(rows[0]) if rows else ["query_id"]
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_bank(path: Path) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(np.load(path / "scores.npy"), dtype=np.float64).reshape(-1)
    indices = np.asarray(
        np.load(path / "score_indices.npy"), dtype=np.int64
    ).reshape(-1)
    if scores.shape != indices.shape:
        raise ValueError(f"score/index mismatch under {path}")
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"duplicate score indices under {path}")
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"non-finite score under {path}")
    return scores, indices


def select_topk(
    scores: np.ndarray, indices: np.ndarray, topk: int, ranking_sign: int
) -> tuple[set[int], float]:
    ranking = ranking_sign * scores
    order = np.argsort(-ranking, kind="stable")[:topk]
    if len(order) != topk:
        raise ValueError(f"only {len(order)} candidates, cannot select top-{topk}")
    return set(map(int, indices[order])), float(ranking[order[-1]])


def methods() -> list[tuple[str, str, str, int]]:
    output: list[tuple[str, str, str, int]] = []
    for transform in ("raw_gradient", "adamw_full"):
        output.append(
            (
                "endpoint_pollute",
                transform,
                f"celeba_endpoint_pollute_{transform}_train1x50_query1x50_"
                "timestamp_sum_squared_shape3d_semantics",
                +1,
            )
        )
        output.append(
            (
                "endpoint_tracin",
                transform,
                f"celeba_endpoint_tracin_{transform}_"
                "train_mean50_query_mean50_linear_shape3d_semantics",
                -1,
            )
        )
    return output


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument(
        "--query-ids", default=",".join(str(value) for value in range(72))
    )
    parser.add_argument("--experiment", default="attr40_experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--full-suffix", default="_full10000")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    records = load_queries(args.query_file)
    query_ids = parse_ids(args.query_ids)
    result_root = CELEBA_ROOT / "result" / args.experiment
    output_dir = args.output_dir or (
        result_root / "analysis" / f"endpoint_full10k_top{args.topk}_change"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for query_id in query_ids:
        record = records[query_id]
        root = (
            result_root
            / "attribution_score"
            / "prompted_solo"
            / f"train_seed_{args.train_seed}"
            / f"query_{_prompt_path_tag(str(record['prompt']))}"
            / f"initial_seed_{query_seed(record)}"
        )
        for method, transform, namespace, ranking_sign in methods():
            for variant, directory in VARIANTS:
                historical_dir = root / f"traj_tracin_{namespace}" / directory
                full_dir = (
                    root
                    / f"traj_tracin_{namespace}{args.full_suffix}"
                    / directory
                )
                old_scores, old_indices = load_bank(historical_dir)
                full_scores, full_indices = load_bank(full_dir)
                if len(old_indices) != 5000:
                    raise ValueError(
                        f"Q{query_id} {method}/{transform}/{variant}: "
                        f"historical candidate count={len(old_indices)}, expected 5000"
                    )
                if not np.array_equal(full_indices, np.arange(10_000)):
                    raise ValueError(
                        f"Q{query_id} {method}/{transform}/{variant}: "
                        "full candidate universe is not exactly 0..9999"
                    )
                full_old_scores = full_scores[old_indices]
                maximum_score_difference = float(
                    np.max(np.abs(old_scores - full_old_scores))
                )
                if not np.allclose(old_scores, full_old_scores, rtol=0.0, atol=1e-12):
                    raise ValueError(
                        f"Q{query_id} {method}/{transform}/{variant}: historical "
                        f"scores changed during merge (max abs={maximum_score_difference})"
                    )

                old_top, old_threshold = select_topk(
                    old_scores, old_indices, args.topk, ranking_sign
                )
                full_top, full_threshold = select_topk(
                    full_scores, full_indices, args.topk, ranking_sign
                )
                original_universe = set(map(int, old_indices))
                retained = len(old_top & full_top)
                full_from_original = len(full_top & original_universe)
                rows.append(
                    {
                        "query_id": query_id,
                        "method": method,
                        "transform": transform,
                        "variant": variant,
                        "ranking_sign": ranking_sign,
                        "topk": args.topk,
                        "old_candidate_count": len(old_indices),
                        "full_candidate_count": len(full_indices),
                        "old_top_retained_count": retained,
                        "old_top_retained_fraction": retained / args.topk,
                        "topk_jaccard": retained / (2 * args.topk - retained),
                        "full_top_from_original_count": full_from_original,
                        "full_top_from_original_fraction": full_from_original
                        / args.topk,
                        "full_top_from_remaining_count": args.topk
                        - full_from_original,
                        "full_top_from_remaining_fraction": 1.0
                        - full_from_original / args.topk,
                        "old_signed_score_threshold": old_threshold,
                        "full_signed_score_threshold": full_threshold,
                        "maximum_copied_score_difference": maximum_score_difference,
                    }
                )
        print(f"[query] Q{query_id} compared 16/16 banks", flush=True)

    atomic_csv(output_dir / "per_query_topk_change.csv", rows)

    summary_rows: list[dict[str, object]] = []
    for method, transform, _, ranking_sign in methods():
        for variant, _ in VARIANTS:
            selected = [
                row
                for row in rows
                if row["method"] == method
                and row["transform"] == transform
                and row["variant"] == variant
            ]
            retention = [float(row["old_top_retained_fraction"]) for row in selected]
            from_remaining = [
                float(row["full_top_from_remaining_fraction"]) for row in selected
            ]
            retention_summary = summarize(retention)
            remaining_summary = summarize(from_remaining)
            summary_rows.append(
                {
                    "method": method,
                    "transform": transform,
                    "variant": variant,
                    "ranking_sign": ranking_sign,
                    "queries": len(selected),
                    **{
                        f"retention_{key}": value
                        for key, value in retention_summary.items()
                    },
                    **{
                        f"new_half_fraction_{key}": value
                        for key, value in remaining_summary.items()
                    },
                }
            )
    atomic_csv(output_dir / "summary_topk_change.csv", summary_rows)

    lines = [
        "CELEBA ENDPOINT TOP-K CHANGE: HISTORICAL 5K -> FULL 10K",
        f"queries={len(query_ids)} topk={args.topk}",
        "Pollute ranks stored timestamp-sum-squared descending; "
        "TracIn ranks stored linear score ascending.",
        "Reference under exchangeable old/new halves: retention~50%, new-half~50%.",
        "",
        (
            f"{'METHOD':18s} {'TRANSFORM':12s} {'VARIANT':16s} "
            f"{'RETAIN MEAN':>12s} {'RETAIN MED':>11s} "
            f"{'NEW-HALF MEAN':>13s} {'NEW-HALF MED':>12s}"
        ),
        "-" * 104,
    ]
    for row in summary_rows:
        lines.append(
            f"{str(row['method']):18s} {str(row['transform']):12s} "
            f"{str(row['variant']):16s} "
            f"{float(row['retention_mean']):12.2%} "
            f"{float(row['retention_median']):11.2%} "
            f"{float(row['new_half_fraction_mean']):13.2%} "
            f"{float(row['new_half_fraction_median']):12.2%}"
        )
    report = "\n".join(lines) + "\n"
    (output_dir / "topk_change_report.txt").write_text(report)
    print()
    print(report, end="")
    print(f"[saved] {output_dir / 'per_query_topk_change.csv'}")
    print(f"[saved] {output_dir / 'summary_topk_change.csv'}")
    print(f"[saved] {output_dir / 'topk_change_report.txt'}")


if __name__ == "__main__":
    main()
