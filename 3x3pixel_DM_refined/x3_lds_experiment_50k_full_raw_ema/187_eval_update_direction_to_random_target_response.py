"""Test whether the direction used for the update predicts random target axes.

Each fresh-SGD branch was produced by one update on one source datapoint using
one fixed +epsilon direction over one 250-timestamp block.  For a random target
pollution direction r, the cached loss directional derivative is

    |g_target(r, t)^T Delta theta_source|

and, because this is a fresh-SGD one-step branch,

    Delta theta_source = -lr * clip_scale * g_source(+epsilon, block).

Thus this cache is exactly the requested update-direction cross-gradient proxy,
up to one positive branch-wide scale.  We compare it with the finite predicted-
noise response of the updated model on 100 random target directions.
"""

import json

import numpy as np

from endpoint_direction_mc_config import *


fast = __import__("185_eval_aligned_loss_same_direction_response")


def summarize_cosines(directions, source_direction):
    source = source_direction.reshape(-1).astype(np.float64)
    target = directions.reshape(len(directions), -1).astype(np.float64)
    source /= max(np.linalg.norm(source), NSDL_EPS)
    target /= np.maximum(np.linalg.norm(target, axis=1, keepdims=True), NSDL_EPS)
    cosine = target @ source
    return cosine


def metric_store():
    scopes = (
        "fixed_random_direction_across_t",
        "fixed_t_across_random_directions",
        "all_direction_t_pairs_per_branch",
        "all_t_integrated_across_directions",
        "direction_mean_curve_across_t",
    )
    return {scope: [] for scope in scopes}


def append_scope_metrics(store, proxy, truth):
    store["fixed_random_direction_across_t"].append(
        fast.batch_curve_metrics(proxy, truth)
    )
    store["fixed_t_across_random_directions"].append(
        fast.batch_curve_metrics(proxy.T, truth.T)
    )
    store["all_direction_t_pairs_per_branch"].append(
        fast.batch_curve_metrics(proxy.reshape(-1), truth.reshape(-1))
    )
    store["all_t_integrated_across_directions"].append(
        fast.batch_curve_metrics(proxy.mean(axis=1), truth.mean(axis=1))
    )
    store["direction_mean_curve_across_t"].append(
        fast.batch_curve_metrics(proxy.mean(axis=0), truth.mean(axis=0))
    )


def aggregate(store):
    return {
        scope: fast.aggregate_metric_batches(batches)
        for scope, batches in store.items()
    }


def print_metrics(label, results):
    rows = (
        ("one random direction across 1000 t", "fixed_random_direction_across_t"),
        ("fixed t across 100 random directions", "fixed_t_across_random_directions"),
        ("all random direction-t pairs", "all_direction_t_pairs_per_branch"),
        ("all-t means across random directions", "all_t_integrated_across_directions"),
        ("100-direction mean across 1000 t", "direction_mean_curve_across_t"),
    )
    print(f"\n[{label}]")
    print("scope                                      spearman  pearson  relerr  false-small")
    for row_label, key in rows:
        value = results[key]
        print(
            f"{row_label:42s} "
            f"{value['spearman']['mean']:+.4f}  "
            f"{value['pearson']['mean']:+.4f}  "
            f"{value['relative_error_median']['mean']:.4f}  "
            f"{value['false_small_rate_high_truth']['mean']:.4f}"
        )


def main():
    l2_store = metric_store()
    squared_store = metric_store()
    branch_proxy = []
    branch_l2 = []
    branch_squared = []
    direction_cosines = []
    branches = []

    for source_index in nsdl_datapoint_indices():
        source_dir = edmc_source_dir(source_index)
        fresh_dir = ntcd_source_dir(source_index, "fresh_sgd")
        fresh_archive_path = fresh_dir / "target_direction_prediction_deltas.npz"
        if not fresh_archive_path.is_file():
            raise FileNotFoundError(fresh_archive_path)
        with np.load(fresh_archive_path, allow_pickle=False) as fresh_archive:
            source_direction = fresh_archive["source_direction"].astype(np.float64)
        for block_index, timestamp_block in enumerate(NTCD_TIMESTAMP_BLOCKS):
            path = source_dir / f"block_{block_index}_responses.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as archive:
                actual_squared = archive["actual_sq_l2"].astype(np.float64)
                update_cross_gradient = archive[
                    "loss_abs_directional_derivative"
                ].astype(np.float64)
                directions = archive["directions"].astype(np.float64)

            actual_l2 = np.sqrt(np.maximum(actual_squared, 0.0))
            append_scope_metrics(l2_store, update_cross_gradient, actual_l2)
            append_scope_metrics(
                squared_store,
                np.square(update_cross_gradient),
                actual_squared,
            )
            cosine = summarize_cosines(directions, source_direction)
            direction_cosines.append(cosine)
            branch_proxy.append(float(update_cross_gradient.mean()))
            branch_l2.append(float(actual_l2.mean()))
            branch_squared.append(float(actual_squared.mean()))
            branches.append(
                {
                    "source_datapoint": int(source_index),
                    "target_datapoint": int(ntcd_target_index(source_index)),
                    "timestamp_block": int(block_index),
                    "update_timestamp_start": int(timestamp_block[0]),
                    "update_timestamp_end": int(timestamp_block[-1]),
                    "mean_abs_cross_gradient": float(update_cross_gradient.mean()),
                    "mean_finite_l2": float(actual_l2.mean()),
                    "mean_finite_squared_l2": float(actual_squared.mean()),
                }
            )
        print(f"[analyzed] source={source_index}", flush=True)

    l2 = aggregate(l2_store)
    squared = aggregate(squared_store)
    branch_metrics = {
        "abs_cross_gradient_to_mean_l2": fast.aggregate_metric_batches(
            [fast.batch_curve_metrics(branch_proxy, branch_l2)]
        ),
        "squared_cross_gradient_to_mean_squared_l2": fast.aggregate_metric_batches(
            [
                fast.batch_curve_metrics(
                    np.square(branch_proxy), branch_squared
                )
            ]
        ),
    }
    cosines = np.concatenate(direction_cosines)
    output = {
        "definition": {
            "source_update": "one fresh-SGD step on one source datapoint, its fixed +epsilon direction, and one 250-timestamp block",
            "proxy": "|g_target(random direction r, t)^T Delta theta_source|; exactly proportional to |g_target^T g_source_update| within a branch",
            "finite_target": "norm of predicted-noise(updated)-predicted-noise(null) on the target endpoint polluted along random direction r at t",
            "l2_pairing": "absolute cross-gradient predicts finite L2",
            "squared_pairing": "squared cross-gradient predicts finite squared L2",
            "calibration": "median multiplicative calibration independently within every evaluated curve",
        },
        "source_update_vs_target_direction_cosine": {
            "mean": float(cosines.mean()),
            "std": float(cosines.std()),
            "mean_absolute": float(np.abs(cosines).mean()),
            "max_absolute": float(np.abs(cosines).max()),
        },
        "absolute_cross_gradient_to_finite_l2": l2,
        "squared_cross_gradient_to_finite_squared_l2": squared,
        "across_40_update_branches": branch_metrics,
        "branches": branches,
    }
    output_path = EDMC_ROOT / "update_direction_to_random_target_response.json"
    with open(output_path, "w") as handle:
        json.dump(output, handle, indent=2)

    cosine_summary = output["source_update_vs_target_direction_cosine"]
    print(
        "\nsource-update noise vs random target pollution directions: "
        f"cos={cosine_summary['mean']:+.4f}±{cosine_summary['std']:.4f} "
        f"mean|cos|={cosine_summary['mean_absolute']:.4f}"
    )
    print_metrics("|g_target^T Delta-theta-source| -> finite L2", l2)
    print_metrics(
        "(g_target^T Delta-theta-source)^2 -> finite squared L2", squared
    )
    print("\n[across 40 independent source-update branches]")
    for name, value in branch_metrics.items():
        print(
            f"{name:54s} "
            f"rho={value['spearman']['mean']:+.4f} "
            f"pearson={value['pearson']['mean']:+.4f} "
            f"relerr={value['relative_error_median']['mean']:.4f} "
            f"false={value['false_small_rate_high_truth']['mean']:.4f}"
        )
    print(f"\n[saved] {output_path}")


if __name__ == "__main__":
    main()
