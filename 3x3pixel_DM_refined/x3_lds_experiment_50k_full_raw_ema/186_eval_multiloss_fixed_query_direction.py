"""Predict each fixed query direction using other loss directions."""

import importlib
import json

import numpy as np

from endpoint_direction_mc_config import *


fast = importlib.import_module("185_eval_aligned_loss_same_direction_response")
COUNTS = (1, 2, 4, 8, 10, 20, 50, 99)


def random_excluding_self(generator, count):
    selections = np.empty((EDMC_DIRECTION_COUNT, count), dtype=np.int64)
    all_indices = np.arange(EDMC_DIRECTION_COUNT)
    for query_direction in range(EDMC_DIRECTION_COUNT):
        pool = np.concatenate(
            (all_indices[:query_direction], all_indices[query_direction + 1 :])
        )
        selections[query_direction] = generator.choice(
            pool, size=count, replace=False
        )
    return selections


def main():
    metric_batches = {
        "same_direction_aligned_loss_to_l2": [],
        "same_direction_aligned_loss_to_squared_l2": [],
        "other_direction_finite_mean_to_fixed_l2": [],
        "other_direction_finite_mean_to_fixed_squared_l2": [],
        "mean_abs_other_loss_to_fixed_l2": {count: [] for count in COUNTS},
        "mean_square_other_loss_to_fixed_squared_l2": {
            count: [] for count in COUNTS
        },
    }
    for source_index in nsdl_datapoint_indices():
        source_dir = edmc_source_dir(source_index)
        for block_index in range(len(NTCD_TIMESTAMP_BLOCKS)):
            path = source_dir / f"block_{block_index}_responses.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as archive:
                actual_squared = archive["actual_sq_l2"].astype(np.float64)
                loss_abs = archive[
                    "loss_abs_directional_derivative"
                ].astype(np.float64)
            actual_l2 = np.sqrt(np.maximum(actual_squared, 0.0))
            metric_batches["same_direction_aligned_loss_to_l2"].append(
                fast.batch_curve_metrics(loss_abs, actual_l2)
            )
            metric_batches[
                "same_direction_aligned_loss_to_squared_l2"
            ].append(fast.batch_curve_metrics(loss_abs, actual_squared))
            other_l2_mean = (
                actual_l2.sum(axis=0, keepdims=True) - actual_l2
            ) / float(EDMC_DIRECTION_COUNT - 1)
            other_squared_mean = (
                actual_squared.sum(axis=0, keepdims=True) - actual_squared
            ) / float(EDMC_DIRECTION_COUNT - 1)
            metric_batches["other_direction_finite_mean_to_fixed_l2"].append(
                fast.batch_curve_metrics(other_l2_mean, actual_l2)
            )
            metric_batches[
                "other_direction_finite_mean_to_fixed_squared_l2"
            ].append(
                fast.batch_curve_metrics(other_squared_mean, actual_squared)
            )
            generator = np.random.default_rng(
                EDMC_DIRECTION_SEED_BASE + 31 * source_index + block_index
            )
            for count in COUNTS:
                repeats = 1 if count == EDMC_DIRECTION_COUNT - 1 else EDMC_SUBSET_REPEATS
                abs_proxies = []
                square_proxies = []
                for _ in range(repeats):
                    selection = random_excluding_self(generator, count)
                    selected = loss_abs[selection]
                    abs_proxies.append(selected.mean(axis=1))
                    square_proxies.append(np.square(selected).mean(axis=1))
                abs_proxy = np.concatenate(abs_proxies, axis=0)
                square_proxy = np.concatenate(square_proxies, axis=0)
                repeated_l2 = np.tile(actual_l2, (repeats, 1))
                repeated_squared = np.tile(actual_squared, (repeats, 1))
                metric_batches["mean_abs_other_loss_to_fixed_l2"][count].append(
                    fast.batch_curve_metrics(abs_proxy, repeated_l2)
                )
                metric_batches[
                    "mean_square_other_loss_to_fixed_squared_l2"
                ][count].append(
                    fast.batch_curve_metrics(square_proxy, repeated_squared)
                )
        print(f"[analyzed] source={source_index}", flush=True)

    output = {
        "definition": {
            "query_target": "finite response of one fixed query pollution direction q",
            "loss_directions": "R independently selected directions excluding q",
            "mean_abs": "mean_r |grad L_r dot parameter_delta|",
            "mean_square": "mean_r (grad L_r dot parameter_delta)^2",
            "calibration": "median multiplicative calibration separately for each fixed-query curve",
        },
        "controls": {
            name: fast.aggregate_metric_batches(batches)
            for name, batches in metric_batches.items()
            if not isinstance(batches, dict)
        },
        "mean_abs_other_loss_to_fixed_l2": {
            str(count): fast.aggregate_metric_batches(batches)
            for count, batches in metric_batches[
                "mean_abs_other_loss_to_fixed_l2"
            ].items()
        },
        "mean_square_other_loss_to_fixed_squared_l2": {
            str(count): fast.aggregate_metric_batches(batches)
            for count, batches in metric_batches[
                "mean_square_other_loss_to_fixed_squared_l2"
            ].items()
        },
    }
    output_path = EDMC_ROOT / "multiloss_fixed_query_direction.json"
    with open(output_path, "w") as handle:
        json.dump(output, handle, indent=2)

    print("\nfixed query direction; loss directions exclude query direction")
    print("R    mean-abs -> fixed L2             mean-square -> fixed squared-L2")
    print("     rho / relerr / false             rho / relerr / false")
    for count in COUNTS:
        absolute = output["mean_abs_other_loss_to_fixed_l2"][str(count)]
        squared = output[
            "mean_square_other_loss_to_fixed_squared_l2"
        ][str(count)]
        print(
            f"{count:2d}   "
            f"{absolute['spearman']['mean']:+.4f} / "
            f"{absolute['relative_error_median']['mean']:.4f} / "
            f"{absolute['false_small_rate_high_truth']['mean']:.4f}       "
            f"{squared['spearman']['mean']:+.4f} / "
            f"{squared['relative_error_median']['mean']:.4f} / "
            f"{squared['false_small_rate_high_truth']['mean']:.4f}"
        )
    print("\ncontrols")
    for name, metrics in output["controls"].items():
        print(
            f"{name:52s} "
            f"rho={metrics['spearman']['mean']:+.4f} "
            f"relerr={metrics['relative_error_median']['mean']:.4f} "
            f"false={metrics['false_small_rate_high_truth']['mean']:.4f}"
        )
    print(f"\n[saved] {output_path}")


if __name__ == "__main__":
    main()
