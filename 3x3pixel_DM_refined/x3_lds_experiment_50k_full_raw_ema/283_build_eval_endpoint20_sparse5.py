"""Build the nearest available sparse-5 from endpoint20 per-t scores."""

import json
import os

import numpy as np
from scipy.stats import spearmanr, wilcoxon

from endpoint20_meanloss_pairing_config import *


SOURCE_METHOD = (
    "tracin_das_endpoint20_inverse_noise_10ckpt_20t_per_timestamp_aligned_"
    "adamw_full_next_delta_projected4096_old_trajectory_projection_raw_"
    "timestamp_sum_squared"
)
OUTPUT_METHOD = (
    "tracin_das_endpoint20_old_projection_sparse5_nearest_old_grid_"
    "t10_t60_t121_t181_t201"
)
OLD_RESULT = LDS_DIR / "tracin_das_trajectory_pairing_100q_significance.json"
TARGET_OLD_GRID = (0, 60, 121, 181, 242)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary, path)


def evaluate(scores, membership, observed):
    prediction = membership @ scores.T
    return {
        metric: np.asarray(
            [
                spearmanr(-prediction[:, q], target[q]).statistic
                for q in range(scores.shape[0])
            ],
            dtype=np.float64,
        )
        for metric, target in observed.items()
    }


def paired(candidate, reference):
    difference = candidate - reference
    finite = difference[np.isfinite(difference)]
    try:
        pvalue = float(wilcoxon(finite, alternative="two-sided").pvalue)
    except ValueError:
        pvalue = 1.0
    return {
        "mean_difference": float(np.nanmean(difference)),
        "wilcoxon_two_sided_p": pvalue,
        "wins": int(np.count_nonzero(difference > 0)),
        "query_count": int(len(finite)),
        "difference_per_query": difference.tolist(),
    }


def main():
    available = np.asarray(E20_TIMESTEPS, dtype=np.int64)
    selected_indices = tuple(
        int(np.argmin(np.abs(available - target))) for target in TARGET_OLD_GRID
    )
    if len(set(selected_indices)) != len(selected_indices):
        raise ValueError(f"nearest timestamps are not unique: {selected_indices}")
    selected_timesteps = tuple(int(available[index]) for index in selected_indices)

    by_timestamp = np.stack(
        [
            np.load(
                ATTR_DIR / SOURCE_METHOD / f"q{query_id:02d}"
                / "scores_by_timestamp.npy"
            ).astype(np.float64)
            for query_id in E20_QUERY_IDS
        ],
        axis=0,
    )
    sparse5 = by_timestamp[:, selected_indices, :].mean(axis=1)
    dense20 = by_timestamp.mean(axis=1)
    for position, query_id in enumerate(E20_QUERY_IDS):
        output = ATTR_DIR / OUTPUT_METHOD / f"q{query_id:02d}"
        output.mkdir(parents=True, exist_ok=True)
        np.save(output / "scores.npy", sparse5[position].astype(np.float32))
        atomic_json(
            output / "info.json",
            {
                "source_method": SOURCE_METHOD,
                "target_old_grid": list(TARGET_OLD_GRID),
                "selected_indices": list(selected_indices),
                "selected_timesteps": list(selected_timesteps),
                "aggregate": "mean of selected per-t squared responses",
            },
        )

    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    observed = {
        metric: np.load(LDS_DIR / f"observed_{metric}.npy").astype(np.float64)[
            list(E20_QUERY_IDS)
        ]
        for metric in LDS_METRICS
    }
    sparse_values = evaluate(sparse5, membership, observed)
    dense_values = evaluate(dense20, membership, observed)
    with open(OLD_RESULT) as handle:
        old = json.load(handle)

    result = {
        "method": OUTPUT_METHOD,
        "query_ids": list(E20_QUERY_IDS),
        "target_old_grid": list(TARGET_OLD_GRID),
        "selected_indices": list(selected_indices),
        "selected_timesteps": list(selected_timesteps),
        "projection_namespace": "trajectory_bridge_projection",
        "results": {},
    }
    lines = [
        "ENDPOINT20 NEAREST-AVAILABLE SPARSE-5",
        f"target old grid={list(TARGET_OLD_GRID)}",
        f"available nearest grid={list(selected_timesteps)}",
        "All comparisons use trajectory_bridge_projection; q00-q09; sign=-1.",
        "",
        "target                              approx5    dense20    exact-old5 "
        " approx-dense/p/wins   approx-exact/p/wins",
    ]
    for metric in LDS_METRICS:
        exact = np.asarray(
            old["methods"]["trajectory_aligned"]["q1"][metric]
            ["per_query"][:10],
            dtype=np.float64,
        )
        vs_dense = paired(sparse_values[metric], dense_values[metric])
        vs_exact = paired(sparse_values[metric], exact)
        result["results"][metric] = {
            "approximate_sparse5": {
                "mean": float(np.nanmean(sparse_values[metric])),
                "std": float(np.nanstd(sparse_values[metric])),
                "per_query": sparse_values[metric].tolist(),
            },
            "dense20": {
                "mean": float(np.nanmean(dense_values[metric])),
                "std": float(np.nanstd(dense_values[metric])),
                "per_query": dense_values[metric].tolist(),
            },
            "exact_old_sparse5": {
                "mean": float(np.nanmean(exact)),
                "std": float(np.nanstd(exact)),
                "per_query": exact.tolist(),
            },
            "approximate_vs_dense": vs_dense,
            "approximate_vs_exact": vs_exact,
        }
        lines.append(
            f"{metric:34s} "
            f"{np.nanmean(sparse_values[metric]):+.6f} "
            f"{np.nanmean(dense_values[metric]):+.6f} "
            f"{np.nanmean(exact):+.6f}   "
            f"{vs_dense['mean_difference']:+.6f}/"
            f"{vs_dense['wilcoxon_two_sided_p']:.6g}/"
            f"{vs_dense['wins']:02d}/10   "
            f"{vs_exact['mean_difference']:+.6f}/"
            f"{vs_exact['wilcoxon_two_sided_p']:.6g}/"
            f"{vs_exact['wins']:02d}/10"
        )

    json_path = LDS_DIR / f"{OUTPUT_METHOD}_q00_q09.json"
    text_path = LDS_DIR / f"{OUTPUT_METHOD}_q00_q09.txt"
    atomic_json(json_path, result)
    text_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[saved] {json_path}")
    print(f"[saved] {text_path}")


if __name__ == "__main__":
    main()
