"""Evaluate q00-q09 projected Traj with antithetic training gradients."""

import json

import numpy as np
from scipy.stats import spearmanr

from exp_config import ATTR_DIR, LDS_DIR, LDS_METRICS, MASK_DIR, TRACIN_CONTRACTIONS


QUERY_IDS = tuple(range(10))
SUFFIX = "antithetic10pairs_q00_q09"
METHODS = tuple(
    f"traj_projected_first_raw_{contraction}_{SUFFIX}"
    for contraction in TRACIN_CONTRACTIONS
)


def main():
    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    output = {
        "query_ids": list(QUERY_IDS),
        "query_count": len(QUERY_IDS),
        "checkpoint_direction": "forward/next",
        "parameter_source": "raw",
        "projection": "CountSketch4096",
        "trajectory_timestamps": 100,
        "query_train_alignment": "timestamp_only",
        "train_noise_sampling": "10 antithetic pairs (+epsilon,-epsilon)",
        "train_loss_terms": 20,
        "results": {},
    }
    for method in METHODS:
        scores = [
            np.load(ATTR_DIR / method / f"q{query_id:02d}" / "scores.npy").astype(
                np.float64
            )
            for query_id in QUERY_IDS
        ]
        method_result = {}
        print(f"\nMETHOD: {method}", flush=True)
        for metric in LDS_METRICS:
            observed = np.load(LDS_DIR / f"observed_{metric}.npy").astype(
                np.float64
            )
            positive = np.asarray(
                [
                    spearmanr(
                        membership @ scores[position], observed[query_id]
                    ).statistic
                    for position, query_id in enumerate(QUERY_IDS)
                ],
                dtype=np.float64,
            )
            signs = {}
            for sign_name, values in (("negative", -positive), ("positive", positive)):
                signs[sign_name] = {
                    "mean": float(np.nanmean(values)),
                    "std": float(np.nanstd(values)),
                    "per_query": values.tolist(),
                }
            method_result[metric] = signs
            print(
                f"{metric:30s} "
                f"sign=-1 {signs['negative']['mean']:+.6f}±"
                f"{signs['negative']['std']:.6f} | "
                f"sign=+1 {signs['positive']['mean']:+.6f}±"
                f"{signs['positive']['std']:.6f}",
                flush=True,
            )
        output["results"][method] = method_result

    output_path = LDS_DIR / "projected_traj_antithetic10pairs_q00_q09.json"
    with open(output_path, "w") as handle:
        json.dump(output, handle, indent=2)
    print(f"[saved] {output_path}", flush=True)


if __name__ == "__main__":
    main()
