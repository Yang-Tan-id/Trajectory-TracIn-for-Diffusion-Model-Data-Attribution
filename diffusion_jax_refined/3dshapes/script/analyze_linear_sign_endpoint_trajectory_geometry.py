#!/usr/bin/env python3
"""Relate oracle linear-score sign to own-trajectory convergence geometry."""

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


def rmse(left: np.ndarray, right: np.ndarray, axis: tuple[int, ...]) -> np.ndarray:
    return np.sqrt(np.mean(np.square(left - right, dtype=np.float64), axis=axis))


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def geometry_artifact(args: argparse.Namespace, query: int, required: tuple[str, ...]):
    namespaces = (
        args.namespace,
        "loss_direction_original_f_checkpoint_own_trajectory_states",
        "loss_direction_original_f_checkpoint_own_trajectory_endpoints_all10",
    )
    checked = []
    for namespace in dict.fromkeys(namespaces):
        path = artifact_path(
            args.experiment, args.train_seed, args.epochs, query, namespace
        )
        checked.append(path)
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as payload:
            if all(key in payload for key in required):
                return path
    formatted = "\n".join(str(path) for path in checked)
    raise FileNotFoundError(
        f"Q{query}: no artifact contains complete endpoint/state geometry; checked:\n"
        f"{formatted}"
    )


def load_oracle_signs(
    path: Path, reduction: str, variant: str
) -> dict[int, tuple[str, float]]:
    result = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["reduction"] != reduction or row["variant"] != variant:
                continue
            sign_key = "prediction_sign" if "prediction_sign" in row else "sign"
            if row[sign_key] != "p1":
                continue
            p1 = float(row["cf_joint_percent"])
            result[int(row["query"])] = ("p1" if p1 >= 0 else "m1", abs(p1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--query-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument(
        "--namespace", default="loss_direction_original_f_checkpoint_own_trajectory"
    )
    parser.add_argument("--score-results", type=Path, required=True)
    parser.add_argument("--reduction", default="linear")
    parser.add_argument("--variant", default="query_l2")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    query_ids = [int(value) for value in args.query_ids.replace(",", " ").split()]
    signs = load_oracle_signs(args.score_results, args.reduction, args.variant)
    missing = sorted(set(query_ids) - set(signs))
    if missing:
        raise ValueError(f"score results missing queries: {missing}")

    checkpoint_rows = []
    timestamp_rows = []
    query_rows = []
    required = (
        "checkpoint_own_trajectory_endpoints",
        "checkpoint_own_trajectory_reference_endpoint",
        "checkpoint_own_trajectory_states",
        "checkpoint_own_trajectory_reference_states",
        "checkpoint_own_trajectory_state_timesteps",
    )
    for query in query_ids:
        path = geometry_artifact(args, query, required)
        with np.load(path, allow_pickle=False) as payload:
            endpoints = np.asarray(
                payload["checkpoint_own_trajectory_endpoints"], dtype=np.float64
            )
            reference_endpoint = np.asarray(
                payload["checkpoint_own_trajectory_reference_endpoint"],
                dtype=np.float64,
            ).squeeze(axis=0)
            states = np.asarray(
                payload["checkpoint_own_trajectory_states"], dtype=np.float64
            )
            reference_states = np.asarray(
                payload["checkpoint_own_trajectory_reference_states"], dtype=np.float64
            )
            timesteps = np.asarray(
                payload["checkpoint_own_trajectory_state_timesteps"], dtype=np.int32
            )

        if states.shape[:2] != (len(endpoints), len(timesteps)):
            raise ValueError(f"Q{query}: inconsistent state shape {states.shape}")
        if reference_states.shape[0] != len(timesteps):
            raise ValueError(f"Q{query}: inconsistent reference-state shape")
        endpoint_axes = tuple(range(1, endpoints.ndim))
        state_axes = tuple(range(2, states.ndim))
        endpoint_reference = rmse(
            endpoints,
            reference_endpoint[None, ...],
            endpoint_axes,
        )
        state_reference = rmse(
            states,
            reference_states[None, ...],
            state_axes,
        )
        state_own_endpoint = rmse(
            states,
            endpoints[:, None, ...],
            state_axes,
        )
        next_improvement = endpoint_reference[:-1] - endpoint_reference[1:]
        sign, strength = signs[query]

        for checkpoint in range(len(endpoints)):
            checkpoint_rows.append(
                {
                    "query": query,
                    "sign": sign,
                    "abs_lds_percent": strength,
                    "checkpoint": checkpoint + 1,
                    "epoch": 4 * (checkpoint + 1),
                    "endpoint_reference_rmse": float(endpoint_reference[checkpoint]),
                    "next_endpoint_reference_rmse": (
                        float("nan")
                        if checkpoint + 1 == len(endpoints)
                        else float(endpoint_reference[checkpoint + 1])
                    ),
                    "next_is_closer": (
                        float("nan")
                        if checkpoint + 1 == len(endpoints)
                        else int(next_improvement[checkpoint] > 0)
                    ),
                    "next_rmse_improvement": (
                        float("nan")
                        if checkpoint + 1 == len(endpoints)
                        else float(next_improvement[checkpoint])
                    ),
                }
            )
        for tslot, timestep in enumerate(timesteps):
            state_ref = state_reference[:, tslot]
            state_end = state_own_endpoint[:, tslot]
            difference_from_endpoint_distance = state_ref - endpoint_reference
            timestamp_rows.append(
                {
                    "query": query,
                    "sign": sign,
                    "abs_lds_percent": strength,
                    "timestep": int(timestep),
                    "state_reference_rmse_mean": float(np.mean(state_ref)),
                    "state_own_endpoint_rmse_mean": float(np.mean(state_end)),
                    "endpoint_reference_rmse_mean": float(
                        np.mean(endpoint_reference)
                    ),
                    "state_minus_endpoint_reference_rmse_mean": float(
                        np.mean(difference_from_endpoint_distance)
                    ),
                    "state_closer_to_reference_than_endpoint_fraction": float(
                        np.mean(state_ref < endpoint_reference)
                    ),
                    "state_reference_rmse_checkpoint_correlation": safe_correlation(
                        np.arange(len(state_ref)), state_ref
                    ),
                }
            )
        query_rows.append(
            {
                "query": query,
                "sign": sign,
                "abs_lds_percent": strength,
                "endpoint_reference_rmse_start": float(endpoint_reference[0]),
                "endpoint_reference_rmse_end": float(endpoint_reference[-1]),
                "endpoint_reference_rmse_reduction": float(
                    endpoint_reference[0] - endpoint_reference[-1]
                ),
                "next_endpoint_closer_fraction": float(
                    np.mean(next_improvement > 0)
                ),
                "next_endpoint_rmse_improvement_mean": float(
                    np.mean(next_improvement)
                ),
                "endpoint_rmse_checkpoint_correlation": safe_correlation(
                    np.arange(len(endpoint_reference)), endpoint_reference
                ),
                "trajectory_state_reference_rmse_mean": float(
                    np.mean(state_reference)
                ),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "per_query_checkpoint.csv", checkpoint_rows)
    write_csv(args.out_dir / "per_query_timestamp.csv", timestamp_rows)
    write_csv(args.out_dir / "per_query_summary.csv", query_rows)

    sign_query_rows = []
    for sign in ("p1", "m1"):
        selected = [row for row in query_rows if row["sign"] == sign]
        if not selected:
            continue
        result = {"sign": sign, "queries": len(selected)}
        for key in (
            "abs_lds_percent",
            "endpoint_reference_rmse_start",
            "endpoint_reference_rmse_end",
            "endpoint_reference_rmse_reduction",
            "next_endpoint_closer_fraction",
            "next_endpoint_rmse_improvement_mean",
            "endpoint_rmse_checkpoint_correlation",
            "trajectory_state_reference_rmse_mean",
        ):
            result[f"{key}_mean"] = float(np.mean([row[key] for row in selected]))
        sign_query_rows.append(result)
    write_csv(args.out_dir / "by_sign_summary.csv", sign_query_rows)

    sign_timestamp_rows = []
    grouped = defaultdict(list)
    for row in timestamp_rows:
        grouped[(row["sign"], row["timestep"])].append(row)
    for (sign, timestep), selected in sorted(
        grouped.items(), key=lambda item: (item[0][0], -int(item[0][1]))
    ):
        result = {"sign": sign, "timestep": timestep, "queries": len(selected)}
        for key in (
            "state_reference_rmse_mean",
            "state_own_endpoint_rmse_mean",
            "endpoint_reference_rmse_mean",
            "state_minus_endpoint_reference_rmse_mean",
            "state_closer_to_reference_than_endpoint_fraction",
            "state_reference_rmse_checkpoint_correlation",
        ):
            result[key] = float(np.mean([row[key] for row in selected]))
        sign_timestamp_rows.append(result)
    write_csv(args.out_dir / "by_sign_timestamp.csv", sign_timestamp_rows)

    print(
        f"{args.reduction.upper()} {args.variant.upper()} ORACLE SIGN vs "
        "OWN-TRAJECTORY/REFERENCE GEOMETRY"
    )
    print("Q SIGN |LDS|  R-START    R-END  R-REDUCE NEXT-CLOSER NEXT-IMPROVE TRAJ-R")
    print("-" * 86)
    for row in query_rows:
        print(
            f"{int(row['query']):1d} {row['sign']:>4s} "
            f"{row['abs_lds_percent']:5.2f}% "
            f"{row['endpoint_reference_rmse_start']:8.4f} "
            f"{row['endpoint_reference_rmse_end']:8.4f} "
            f"{row['endpoint_reference_rmse_reduction']:+8.4f} "
            f"{row['next_endpoint_closer_fraction']:11.3f} "
            f"{row['next_endpoint_rmse_improvement_mean']:+12.6f} "
            f"{row['trajectory_state_reference_rmse_mean']:7.4f}"
        )

    print("\nBY SIGN")
    print("SIGN N  |LDS| R-START R-END R-REDUCE NEXT-CLOSER NEXT-IMPROVE TRAJ-R")
    print("-" * 82)
    for row in sign_query_rows:
        print(
            f"{row['sign']:>4s} {int(row['queries']):1d} "
            f"{row['abs_lds_percent_mean']:5.2f}% "
            f"{row['endpoint_reference_rmse_start_mean']:7.4f} "
            f"{row['endpoint_reference_rmse_end_mean']:7.4f} "
            f"{row['endpoint_reference_rmse_reduction_mean']:+8.4f} "
            f"{row['next_endpoint_closer_fraction_mean']:11.3f} "
            f"{row['next_endpoint_rmse_improvement_mean_mean']:+12.6f} "
            f"{row['trajectory_state_reference_rmse_mean_mean']:7.4f}"
        )

    print("\nBY SIGN AND TIMESTAMP")
    print("SIGN    T N  STATE~REF STATE~END END~REF STATE-ENDREF STATE-CLOSER")
    print("-" * 78)
    for row in sign_timestamp_rows:
        print(
            f"{row['sign']:>4s} {int(row['timestep']):4d} {int(row['queries']):1d} "
            f"{row['state_reference_rmse_mean']:10.4f} "
            f"{row['state_own_endpoint_rmse_mean']:9.4f} "
            f"{row['endpoint_reference_rmse_mean']:7.4f} "
            f"{row['state_minus_endpoint_reference_rmse_mean']:+12.4f} "
            f"{row['state_closer_to_reference_than_endpoint_fraction']:12.3f}"
        )
    print(f"[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
