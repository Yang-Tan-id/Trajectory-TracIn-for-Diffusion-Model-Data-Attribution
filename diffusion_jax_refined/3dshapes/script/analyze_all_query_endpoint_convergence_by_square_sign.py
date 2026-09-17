#!/usr/bin/env python3
"""Compare checkpoint-own endpoint convergence for p1/m1 square query groups."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT))

from analyze_predicted_noise_probe24_output_alignment import artifact_path, write_csv


def cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left.reshape(len(left), -1)
    right = np.broadcast_to(right.reshape(1, -1), left.shape)
    return np.sum(left * right, axis=1) / np.maximum(
        np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1), 1e-12
    )


def load_square_signs(path: Path, variant: str) -> dict[int, tuple[str, float]]:
    output = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["variant"] != variant or row["prediction_sign"] != "p1":
                continue
            query = int(row["query"])
            p1_lds = float(row["cf_joint_percent"])
            output[query] = ("p1" if p1_lds >= 0.0 else "m1", abs(p1_lds))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--query-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--square-results", type=Path, required=True)
    parser.add_argument("--variant", default="query_train_l2")
    parser.add_argument("--strong-threshold", type=float, default=3.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    query_ids = [int(item) for item in args.query_ids.replace(",", " ").split()]
    signs = load_square_signs(args.square_results, args.variant)
    missing = sorted(set(query_ids) - set(signs))
    if missing:
        raise ValueError(f"square results missing queries {missing}")

    rows = []
    expected_checkpoints = None
    for query in query_ids:
        path = artifact_path(
            args.experiment,
            args.train_seed,
            args.epochs,
            query,
            args.namespace,
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            endpoints = np.asarray(
                payload["checkpoint_own_trajectory_endpoints"], dtype=np.float64
            )
            reference = np.asarray(
                payload["checkpoint_own_trajectory_reference_endpoint"],
                dtype=np.float64,
            )
            checkpoints = np.unique(
                np.asarray(payload["ckpt_indices"], dtype=np.int32)
            )
        if expected_checkpoints is None:
            expected_checkpoints = checkpoints
        elif not np.array_equal(checkpoints, expected_checkpoints):
            raise ValueError(f"checkpoint mismatch for query {query}: {path}")
        if len(endpoints) != len(checkpoints):
            raise ValueError(
                f"endpoint/checkpoint mismatch for query {query}: "
                f"{endpoints.shape} vs {checkpoints.shape}"
            )

        endpoint_flat = endpoints.reshape(len(endpoints), -1)
        reference_flat = reference.reshape(-1)
        raw_cosine = cosine(endpoint_flat, reference_flat)
        centered_endpoint = endpoint_flat - endpoint_flat.mean(axis=1, keepdims=True)
        centered_reference = reference_flat - reference_flat.mean()
        centered_cosine = cosine(centered_endpoint, centered_reference)
        difference = endpoint_flat - reference_flat[None, :]
        rmse = np.sqrt(np.mean(np.square(difference), axis=1))
        mae = np.mean(np.abs(difference), axis=1)
        sign, strength = signs[query]
        strong = strength >= args.strong_threshold
        for index, checkpoint in enumerate(checkpoints):
            rows.append(
                {
                    "query": query,
                    "square_sign": sign,
                    "square_abs_lds_percent": strength,
                    "strong_square_signal": int(strong),
                    "checkpoint": int(checkpoint) + 1,
                    "epoch": 4 * (int(checkpoint) + 1),
                    "centered_cosine": float(centered_cosine[index]),
                    "cosine": float(raw_cosine[index]),
                    "rmse": float(rmse[index]),
                    "mean_absolute_error": float(mae[index]),
                    "delta_centered_cosine": (
                        float("nan")
                        if index == 0
                        else float(centered_cosine[index] - centered_cosine[index - 1])
                    ),
                    "rmse_improvement": (
                        float("nan")
                        if index == 0
                        else float(rmse[index - 1] - rmse[index])
                    ),
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "per_query_checkpoint.csv", rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["square_sign"], int(row["strong_square_signal"]), row["checkpoint"])].append(row)
    group_rows = []
    for (sign, strong, checkpoint), group in sorted(grouped.items()):
        result = {
            "square_sign": sign,
            "strong_square_signal": strong,
            "checkpoint": checkpoint,
            "epoch": group[0]["epoch"],
            "queries": len(group),
        }
        for metric in (
            "centered_cosine",
            "cosine",
            "rmse",
            "mean_absolute_error",
            "delta_centered_cosine",
            "rmse_improvement",
        ):
            values = np.asarray([row[metric] for row in group], dtype=np.float64)
            result[f"{metric}_mean"] = float(np.nanmean(values)) if not np.all(np.isnan(values)) else float("nan")
            result[f"{metric}_std"] = float(np.nanstd(values)) if not np.all(np.isnan(values)) else float("nan")
        group_rows.append(result)
    write_csv(args.out_dir / "by_sign_strength_checkpoint.csv", group_rows)

    query_rows = []
    by_query = defaultdict(list)
    for row in rows:
        by_query[int(row["query"])].append(row)
    for query, group in sorted(by_query.items()):
        group.sort(key=lambda row: int(row["checkpoint"]))
        centered = np.asarray([row["centered_cosine"] for row in group])
        rmse = np.asarray([row["rmse"] for row in group])
        x = np.arange(len(group), dtype=np.float64)
        query_rows.append(
            {
                "query": query,
                "square_sign": group[0]["square_sign"],
                "square_abs_lds_percent": group[0]["square_abs_lds_percent"],
                "strong_square_signal": group[0]["strong_square_signal"],
                "start_centered_cosine": float(centered[0]),
                "end_centered_cosine": float(centered[-1]),
                "centered_cosine_gain": float(centered[-1] - centered[0]),
                "start_rmse": float(rmse[0]),
                "end_rmse": float(rmse[-1]),
                "rmse_reduction": float(rmse[0] - rmse[-1]),
                "centered_cosine_checkpoint_correlation": float(
                    np.corrcoef(x, centered)[0, 1]
                ),
                "rmse_checkpoint_correlation": float(np.corrcoef(x, rmse)[0, 1]),
                "positive_cosine_steps_fraction": float(np.mean(np.diff(centered) > 0)),
                "positive_rmse_improvement_fraction": float(np.mean(np.diff(rmse) < 0)),
            }
        )
    write_csv(args.out_dir / "per_query_summary.csv", query_rows)

    print(f"{args.variant.upper()} SQUARE SIGN vs OWN-ENDPOINT CONVERGENCE")
    print("Q SIGN |LDS| STRONG  C0       C49      C-GAIN   R0       R49      R-REDUCE  C-UP  R-DOWN")
    print("-" * 105)
    for row in query_rows:
        print(
            f"{int(row['query']):1d} {row['square_sign']:>4s} "
            f"{float(row['square_abs_lds_percent']):5.2f}% "
            f"{int(row['strong_square_signal']):6d} "
            f"{float(row['start_centered_cosine']):+8.4f} "
            f"{float(row['end_centered_cosine']):+8.4f} "
            f"{float(row['centered_cosine_gain']):+8.4f} "
            f"{float(row['start_rmse']):8.4f} "
            f"{float(row['end_rmse']):8.4f} "
            f"{float(row['rmse_reduction']):+9.4f} "
            f"{float(row['positive_cosine_steps_fraction']):5.2f} "
            f"{float(row['positive_rmse_improvement_fraction']):6.2f}"
        )

    print("\nGROUP CURVES AT SELECTED CHECKPOINTS")
    print("GROUP       CKPT   N  CENTERED_COS±STD       RMSE±STD")
    print("-" * 72)
    selected_checkpoints = {1, 10, 20, 30, 40, 49}
    for strong_only in (0, 1):
        for sign in ("p1", "m1"):
            for row in group_rows:
                if (
                    row["square_sign"] == sign
                    and int(row["strong_square_signal"]) == strong_only
                    and int(row["checkpoint"]) in selected_checkpoints
                ):
                    label = f"{sign}-{'strong' if strong_only else 'weak'}"
                    print(
                        f"{label:<11s} {int(row['checkpoint']):4d} "
                        f"{int(row['queries']):3d} "
                        f"{float(row['centered_cosine_mean']):+8.4f}±"
                        f"{float(row['centered_cosine_std']):6.4f} "
                        f"{float(row['rmse_mean']):8.4f}±{float(row['rmse_std']):6.4f}"
                    )
    print(f"[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
