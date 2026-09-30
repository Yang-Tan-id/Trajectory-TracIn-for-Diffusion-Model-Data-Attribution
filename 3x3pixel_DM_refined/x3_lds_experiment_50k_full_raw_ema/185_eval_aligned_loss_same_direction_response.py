"""Test aligned loss projection against same-direction finite response."""

import json

import numpy as np

from endpoint_direction_mc_config import *


SCOPES = (
    "per_direction_across_1000_t",
    "fixed_t_across_100_directions",
    "all_t_integrated_across_100_directions",
    "pooled_direction_t_by_branch",
)


def rowwise_correlation(left, right):
    left = left - left.mean(axis=1, keepdims=True)
    right = right - right.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    result = np.full(len(left), np.nan, dtype=np.float64)
    # Correlation is scale invariant. A fixed absolute epsilon incorrectly
    # classifies small but non-constant squared-response curves as constants.
    valid = np.isfinite(denominator) & (denominator > 0.0)
    result[valid] = np.sum(left[valid] * right[valid], axis=1) / denominator[valid]
    return result


def ordinal_ranks(values):
    order = np.argsort(values, axis=1)
    ranks = np.empty_like(order, dtype=np.float64)
    row = np.arange(len(values))[:, None]
    ranks[row, order] = np.arange(values.shape[1], dtype=np.float64)[None, :]
    return ranks


def batch_curve_metrics(proxy, truth):
    proxy = np.atleast_2d(np.asarray(proxy, dtype=np.float64))
    truth = np.atleast_2d(np.asarray(truth, dtype=np.float64))
    proxy_median = np.median(proxy, axis=1)
    truth_median = np.median(truth, axis=1)
    scale = truth_median / np.maximum(proxy_median, NSDL_EPS)
    estimate = proxy * scale[:, None]
    relative_error = np.abs(estimate - truth) / np.maximum(truth, NSDL_EPS)
    high_truth = truth >= truth_median[:, None]
    false_small = (estimate < 0.5 * truth) & high_truth
    return {
        "pearson": rowwise_correlation(estimate, truth),
        "spearman": rowwise_correlation(
            ordinal_ranks(estimate), ordinal_ranks(truth)
        ),
        "scale": scale,
        "relative_error_mean": relative_error.mean(axis=1),
        "relative_error_median": np.median(relative_error, axis=1),
        "false_small_rate_high_truth": false_small.sum(axis=1)
        / np.maximum(high_truth.sum(axis=1), 1),
    }


def aggregate_metric_batches(batches):
    output = {}
    for metric in batches[0]:
        values = np.concatenate([batch[metric] for batch in batches])
        output[metric] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
            "median": float(np.nanmedian(values)),
            "min": float(np.nanmin(values)),
            "max": float(np.nanmax(values)),
        }
    output["count"] = int(sum(len(batch["spearman"]) for batch in batches))
    return output


def evaluate_targets():
    batches = {
        target: {scope: [] for scope in SCOPES}
        for target in ("l2", "squared_l2")
    }
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
            targets = {
                "l2": np.sqrt(np.maximum(actual_sq_l2, 0.0)),
                "squared_l2": actual_sq_l2,
            }
            for target_name, actual in targets.items():
                batches[target_name]["per_direction_across_1000_t"].append(
                    batch_curve_metrics(loss, actual)
                )
                batches[target_name]["fixed_t_across_100_directions"].append(
                    batch_curve_metrics(loss.T, actual.T)
                )
                batches[target_name][
                    "all_t_integrated_across_100_directions"
                ].append(
                    batch_curve_metrics(
                        loss.mean(axis=1), actual.mean(axis=1)
                    )
                )
                batches[target_name]["pooled_direction_t_by_branch"].append(
                    batch_curve_metrics(loss.reshape(-1), actual.reshape(-1))
                )
        print(f"[analyzed] source={source_index}", flush=True)
    return {
        target: {
            scope: aggregate_metric_batches(scope_batches)
            for scope, scope_batches in target_batches.items()
        }
        for target, target_batches in batches.items()
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
    results = evaluate_targets()
    output = {
        "definition": {
            "proxy": "absolute aligned diffusion-loss directional derivative for direction r and timestep t",
            "target": "finite updated-minus-null predicted-noise response at the same direction r and timestep t",
            "calibration": "median multiplicative calibration within each reported curve/group",
        },
        **results,
    }
    output_path = EDMC_ROOT / "aligned_loss_same_direction_response.json"
    with open(output_path, "w") as handle:
        json.dump(output, handle, indent=2)
    print_target("L2", output["l2"])
    print_target("squared L2", output["squared_l2"])
    print(f"\n[saved] {output_path}")


if __name__ == "__main__":
    main()
