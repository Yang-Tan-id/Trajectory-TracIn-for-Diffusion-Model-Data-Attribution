"""Test aligned loss projection against same-direction finite response."""

import importlib
import json

import numpy as np

from endpoint_direction_mc_config import *


analysis = importlib.import_module("180_launch_endpoint_direction_mc_4gpu")


def evaluate_target(transform):
    per_direction_across_t = []
    fixed_t_across_directions = []
    all_t_integrated_across_directions = []
    pooled_direction_t_by_branch = []
    for source_index in nsdl_datapoint_indices():
        source_dir = edmc_source_dir(source_index)
        for block_index in range(len(NTCD_TIMESTAMP_BLOCKS)):
            path = source_dir / f"block_{block_index}_responses.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as archive:
                actual_sq_l2 = archive["actual_sq_l2"].astype(np.float64)
                loss = archive[
                    "loss_abs_directional_derivative"
                ].astype(np.float64)
            actual = transform(actual_sq_l2)
            for direction_index in range(EDMC_DIRECTION_COUNT):
                per_direction_across_t.append(
                    analysis.curve_metrics(
                        loss[direction_index],
                        actual[direction_index],
                        calibrate=True,
                    )
                )
            for timestamp in range(T):
                fixed_t_across_directions.append(
                    analysis.curve_metrics(
                        loss[:, timestamp],
                        actual[:, timestamp],
                        calibrate=True,
                    )
                )
            all_t_integrated_across_directions.append(
                analysis.curve_metrics(
                    loss.mean(axis=1),
                    actual.mean(axis=1),
                    calibrate=True,
                )
            )
            pooled_direction_t_by_branch.append(
                analysis.curve_metrics(
                    loss.reshape(-1),
                    actual.reshape(-1),
                    calibrate=True,
                )
            )
    return {
        "per_direction_across_1000_t": analysis.aggregate_metric_records(
            per_direction_across_t
        ),
        "fixed_t_across_100_directions": analysis.aggregate_metric_records(
            fixed_t_across_directions
        ),
        "all_t_integrated_across_100_directions": analysis.aggregate_metric_records(
            all_t_integrated_across_directions
        ),
        "pooled_direction_t_by_branch": analysis.aggregate_metric_records(
            pooled_direction_t_by_branch
        ),
    }


def print_target(label, result):
    print(f"\n[{label}] aligned loss -> same-direction finite response")
    print("scope                                  spearman  pearson  relerr  false-small")
    rows = (
        ("one direction across 1000 t", "per_direction_across_1000_t"),
        ("fixed t across 100 directions", "fixed_t_across_100_directions"),
        (
            "all-t means across 100 directions",
            "all_t_integrated_across_100_directions",
        ),
        ("all direction-t pairs per branch", "pooled_direction_t_by_branch"),
    )
    for label_text, key in rows:
        metrics = result[key]
        print(
            f"{label_text:38s} "
            f"{metrics['spearman']['mean']:+.4f}  "
            f"{metrics['pearson']['mean']:+.4f}  "
            f"{metrics['relative_error_median']['mean']:.4f}  "
            f"{metrics['false_small_rate_high_truth']['mean']:.4f}"
        )


def main():
    output = {
        "definition": {
            "proxy": "absolute aligned diffusion-loss directional derivative for direction r and timestep t",
            "target": "finite updated-minus-null predicted-noise response at the same direction r and timestep t",
            "calibration": "median multiplicative calibration within each reported curve/group",
        },
        "l2": evaluate_target(
            lambda squared: np.sqrt(np.maximum(squared, 0.0))
        ),
        "squared_l2": evaluate_target(lambda squared: squared),
    }
    output_path = EDMC_ROOT / "aligned_loss_same_direction_response.json"
    with open(output_path, "w") as handle:
        json.dump(output, handle, indent=2)
    print_target("L2", output["l2"])
    print_target("squared L2", output["squared_l2"])
    print(f"\n[saved] {output_path}")


if __name__ == "__main__":
    main()
