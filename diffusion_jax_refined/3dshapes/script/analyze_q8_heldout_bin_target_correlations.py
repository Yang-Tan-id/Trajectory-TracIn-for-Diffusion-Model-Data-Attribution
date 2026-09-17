#!/usr/bin/env python3
"""Measure unsupervised Q8 checkpoint-bin correlations on held-out subset models."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT))

from analyze_original_f_checkpoint_sign_crossfit import load_components, write_csv
from analyze_original_f_timestamp_sign_crossfit import TARGETS, lds, target_data


def centered_covariance(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return float(np.mean((left - left.mean()) * (right - right.mean())))


def pearson_percent(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - left.mean()
    right = right - right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(100.0 * np.dot(left, right) / max(denominator, 1e-12))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--query-id", type=int, default=8)
    parser.add_argument("--variant", default="query_train_l2")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=20260916)
    parser.add_argument("--checkpoint-crossfit-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.skip_predicted = True
    args.shard_index = 0
    args.shard_count = 1
    args.run_id = "q8_heldout_bin_correlations"

    component_path = args.checkpoint_crossfit_dir / "checkpoint_components.npz"
    components, checkpoints, score_indices = load_components(
        component_path, [args.query_id]
    )
    if args.variant not in components:
        raise ValueError(f"unknown variant {args.variant!r}")
    incidence, true = target_data(args, args.query_id, score_indices)
    endpoint = np.asarray(true[TARGETS[0]], dtype=np.float64)
    trajectory = np.asarray(true[TARGETS[1]], dtype=np.float64)

    # This minus sign is the score-to-prediction convention used by the
    # checkpoint-sign crossfit experiment itself.
    checkpoint_predictions = -components[args.variant][0] @ incidence.T
    checkpoint_groups = np.array_split(np.arange(len(checkpoints)), 5)
    bin_predictions = np.stack(
        [checkpoint_predictions[group].sum(axis=0) for group in checkpoint_groups],
        axis=0,
    )
    expected_signs = np.asarray([-1, 1, -1, 1, 1], dtype=np.int8)

    rng = np.random.default_rng(args.random_seed)
    rows = []
    for repeat in range(args.repeats):
        permutation = rng.permutation(len(endpoint))
        folds = (permutation[::2], permutation[1::2])
        for heldout_fold, heldout_ids in enumerate(folds):
            for bin_index, prediction in enumerate(bin_predictions):
                selected_prediction = prediction[heldout_ids]
                endpoint_values, trajectory_values, joint_values = lds(
                    selected_prediction,
                    endpoint[heldout_ids],
                    trajectory[heldout_ids],
                )
                endpoint_corr = float(endpoint_values[0])
                trajectory_corr = float(trajectory_values[0])
                joint_corr = float(joint_values[0])
                endpoint_pearson = pearson_percent(
                    selected_prediction, endpoint[heldout_ids]
                )
                trajectory_pearson = pearson_percent(
                    selected_prediction, trajectory[heldout_ids]
                )
                joint_pearson = 0.5 * (endpoint_pearson + trajectory_pearson)
                raw_sign = 1 if joint_corr >= 0.0 else -1
                rows.append(
                    {
                        "repeat": repeat,
                        "heldout_fold": heldout_fold,
                        "models": len(heldout_ids),
                        "checkpoint_bin": bin_index + 1,
                        "checkpoint_start": int(checkpoints[checkpoint_groups[bin_index][0]]) + 1,
                        "checkpoint_end": int(checkpoints[checkpoint_groups[bin_index][-1]]) + 1,
                        "expected_flip_sign": int(expected_signs[bin_index]),
                        "raw_correlation_sign": raw_sign,
                        "matches_expected_flip": int(raw_sign == expected_signs[bin_index]),
                        "endpoint_correlation_percent": endpoint_corr,
                        "trajectory_correlation_percent": trajectory_corr,
                        "joint_correlation_percent": joint_corr,
                        "endpoint_pearson_percent": endpoint_pearson,
                        "trajectory_pearson_percent": trajectory_pearson,
                        "joint_pearson_percent": joint_pearson,
                        "endpoint_centered_covariance": centered_covariance(
                            selected_prediction, endpoint[heldout_ids]
                        ),
                        "trajectory_centered_covariance": centered_covariance(
                            selected_prediction, trajectory[heldout_ids]
                        ),
                    }
                )

    groups = defaultdict(list)
    for row in rows:
        groups[int(row["checkpoint_bin"])].append(row)
    summary = []
    for bin_index in range(1, 6):
        group = groups[bin_index]
        result = {
            "checkpoint_bin": bin_index,
            "checkpoint_start": group[0]["checkpoint_start"],
            "checkpoint_end": group[0]["checkpoint_end"],
            "expected_flip_sign": group[0]["expected_flip_sign"],
            "splits": len(group),
        }
        for metric in (
            "endpoint_correlation_percent",
            "trajectory_correlation_percent",
            "joint_correlation_percent",
            "endpoint_pearson_percent",
            "trajectory_pearson_percent",
            "joint_pearson_percent",
            "endpoint_centered_covariance",
            "trajectory_centered_covariance",
        ):
            values = np.asarray([row[metric] for row in group], dtype=np.float64)
            result[f"{metric}_mean"] = float(np.mean(values))
            result[f"{metric}_std"] = float(np.std(values))
            result[f"{metric}_positive_fraction"] = float(np.mean(values > 0.0))
        result["match_expected_fraction"] = float(
            np.mean([row["matches_expected_flip"] for row in group])
        )
        summary.append(result)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "per_heldout_split.csv", rows)
    write_csv(args.out_dir / "summary.csv", summary)

    print(
        f"Q{args.query_id} {args.variant.upper()} — RAW CHECKPOINT-BIN/TARGET "
        "CORRELATIONS ON 40 HELD-OUT FOLDS"
    )
    print("No sign is selected or fitted in this calculation.")
    print(
        "BIN CKPTS EXPECT  SPEAR-JOINT±STD  S+  PEAR-JOINT±STD  P+ MATCH"
    )
    for row in summary:
        print(
            f"{int(row['checkpoint_bin']):3d} "
            f"{int(row['checkpoint_start']):02d}-{int(row['checkpoint_end']):02d} "
            f"{int(row['expected_flip_sign']):+6d} "
            f"{row['joint_correlation_percent_mean']:+7.3f}±"
            f"{row['joint_correlation_percent_std']:6.3f} "
            f"{row['joint_correlation_percent_positive_fraction']:5.2f} "
            f"{row['joint_pearson_percent_mean']:+7.3f}±"
            f"{row['joint_pearson_percent_std']:6.3f} "
            f"{row['joint_pearson_percent_positive_fraction']:5.2f} "
            f"{row['match_expected_fraction']:5.2f}"
        )
    print(f"[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
