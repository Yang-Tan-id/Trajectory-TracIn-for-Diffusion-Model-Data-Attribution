#!/usr/bin/env python3
"""Summarize exact J(delta-params) versus true adjacent-checkpoint output deltas."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT))

from analyze_reference_probe_delta_alignment import artifact_path  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--query-ids", default="0,3,7,8")
    parser.add_argument(
        "--geometry-namespace",
        default="parameter_delta_jvp_sanity_reference",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    query_ids = [int(value) for value in args.query_ids.split(",") if value]
    records = json.loads((SHAPES_ROOT / "queries_seed_0_9.json").read_text())["queries"]
    term_rows: list[dict[str, object]] = []

    for query_id in query_ids:
        path = artifact_path(args, query_id)
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            ckpts = np.asarray(payload["ckpt_indices"], dtype=np.int32)
            timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
            cosines = np.asarray(payload["parameter_delta_jvp_cosines"], dtype=np.float64)
            ratios = np.asarray(payload["parameter_delta_jvp_norm_ratios"], dtype=np.float64)
            errors = np.asarray(payload["parameter_delta_jvp_relative_errors"], dtype=np.float64)
            predicted_norms = np.asarray(
                payload["parameter_delta_jvp_predicted_norms"], dtype=np.float64
            )
            true_norms = np.asarray(
                payload["parameter_delta_jvp_true_norms"], dtype=np.float64
            )
        expected = 49 * 10
        arrays = (ckpts, timesteps, cosines, ratios, errors, predicted_norms, true_norms)
        if any(len(values) != expected for values in arrays):
            raise ValueError(f"{path}: expected {expected} terms; got {[len(x) for x in arrays]}")
        for term in range(expected):
            term_rows.append(
                {
                    "query": query_id,
                    "prompt": records[query_id]["prompt"],
                    "checkpoint": int(ckpts[term]) + 1,
                    "epoch": 4 * (int(ckpts[term]) + 1),
                    "timestep": int(timesteps[term]),
                    "cosine": float(cosines[term]),
                    "norm_ratio": float(ratios[term]),
                    "relative_error": float(errors[term]),
                    "predicted_norm": float(predicted_norms[term]),
                    "true_norm": float(true_norms[term]),
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "per_checkpoint_timestamp.csv", term_rows)

    print("EXACT PARAMETER-DELTA JVP vs TRUE PREDICTED-NOISE CHANGE")
    print("Fixed reference trajectory; query prompt conditioning retained")
    print(f"{'Q':>2s} {'N':>4s} {'COS-MEAN':>9s} {'COS-MED':>9s} {'COS>0':>7s} "
          f"{'COS>.9':>7s} {'RATIO':>9s} {'RELERR':>9s}")
    print("-" * 74)
    summary_rows: list[dict[str, object]] = []
    for query_id in query_ids:
        selected = [row for row in term_rows if row["query"] == query_id]
        cos = np.asarray([row["cosine"] for row in selected], dtype=np.float64)
        ratio = np.asarray([row["norm_ratio"] for row in selected], dtype=np.float64)
        error = np.asarray([row["relative_error"] for row in selected], dtype=np.float64)
        summary = {
            "query": query_id,
            "n": len(selected),
            "cosine_mean": float(np.mean(cos)),
            "cosine_median": float(np.median(cos)),
            "cosine_positive_fraction": float(np.mean(cos > 0.0)),
            "cosine_above_0_9_fraction": float(np.mean(cos > 0.9)),
            "norm_ratio_median": float(np.median(ratio)),
            "relative_error_median": float(np.median(error)),
        }
        summary_rows.append(summary)
        print(
            f"{query_id:2d} {len(selected):4d} {summary['cosine_mean']:+9.4f} "
            f"{summary['cosine_median']:+9.4f} "
            f"{summary['cosine_positive_fraction']:7.3f} "
            f"{summary['cosine_above_0_9_fraction']:7.3f} "
            f"{summary['norm_ratio_median']:9.4f} "
            f"{summary['relative_error_median']:9.4f}"
        )
    write_csv(args.out_dir / "by_query.csv", summary_rows)

    print("\nBY QUERY AND TIMESTAMP")
    print(f"{'Q':>2s} {'T':>4s} {'N':>3s} {'COS-MEAN':>9s} {'COS-MED':>9s} "
          f"{'RATIO':>9s} {'RELERR':>9s}")
    print("-" * 62)
    timestamp_rows: list[dict[str, object]] = []
    for query_id in query_ids:
        timestamps = sorted(
            {int(row["timestep"]) for row in term_rows if row["query"] == query_id},
            reverse=True,
        )
        for timestep in timestamps:
            selected = [
                row for row in term_rows
                if row["query"] == query_id and row["timestep"] == timestep
            ]
            cos = np.asarray([row["cosine"] for row in selected], dtype=np.float64)
            ratio = np.asarray([row["norm_ratio"] for row in selected], dtype=np.float64)
            error = np.asarray([row["relative_error"] for row in selected], dtype=np.float64)
            result = {
                "query": query_id,
                "timestep": timestep,
                "n": len(selected),
                "cosine_mean": float(np.mean(cos)),
                "cosine_median": float(np.median(cos)),
                "norm_ratio_median": float(np.median(ratio)),
                "relative_error_median": float(np.median(error)),
            }
            timestamp_rows.append(result)
            print(
                f"{query_id:2d} {timestep:4d} {len(selected):3d} "
                f"{result['cosine_mean']:+9.4f} {result['cosine_median']:+9.4f} "
                f"{result['norm_ratio_median']:9.4f} "
                f"{result['relative_error_median']:9.4f}"
            )
    write_csv(args.out_dir / "by_query_timestamp.csv", timestamp_rows)
    print(f"\n[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
