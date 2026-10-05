"""Evaluate classic EMA-parameter MC10 Trajectory-TracIn on q00-q99."""

import json

import numpy as np
from scipy.stats import spearmanr

from exp_config import ATTR_DIR, LDS_DIR, LDS_METRICS, MASK_DIR, TRACIN_CONTRACTIONS


QUERY_IDS = tuple(range(100))
SUFFIX = "mc10_100q"
METHODS = tuple(
    f"traj_projected_first_ema_{contraction}_{SUFFIX}"
    for contraction in TRACIN_CONTRACTIONS
)


def main():
    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    result = {
        "query_ids": list(QUERY_IDS),
        "parameter_source": "ema",
        "parameter_transform": "none (ordinary projected gradients)",
        "trajectory_timestamps": 100,
        "train_mc": 10,
        "train_noise": "independent",
        "projection": "CountSketch4096",
        "results": {},
    }
    for method in METHODS:
        scores = np.stack(
            [
                np.load(ATTR_DIR / method / f"q{query_id:02d}" / "scores.npy")
                .astype(np.float64)
                for query_id in QUERY_IDS
            ]
        )
        method_result = {}
        print(f"\nMETHOD: {method}", flush=True)
        for metric in LDS_METRICS:
            observed = np.load(LDS_DIR / f"observed_{metric}.npy").astype(
                np.float64
            )[list(QUERY_IDS)]
            positive = np.asarray(
                [
                    spearmanr(membership @ scores[q], observed[q]).statistic
                    for q in range(len(QUERY_IDS))
                ]
            )
            signs = {}
            for name, values in (("negative", -positive), ("positive", positive)):
                signs[name] = {
                    "mean": float(np.nanmean(values)),
                    "std": float(np.nanstd(values)),
                    "per_query": values.tolist(),
                }
            method_result[metric] = signs
            print(
                f"{metric:30s} "
                f"sign=-1 {signs['negative']['mean']:+.6f}±{signs['negative']['std']:.6f} | "
                f"sign=+1 {signs['positive']['mean']:+.6f}±{signs['positive']['std']:.6f}",
                flush=True,
            )
        result["results"][method] = method_result

    output = LDS_DIR / "traj_tracin_ema_mc10_100q.json"
    with open(output, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"[saved] {output}", flush=True)


if __name__ == "__main__":
    main()
