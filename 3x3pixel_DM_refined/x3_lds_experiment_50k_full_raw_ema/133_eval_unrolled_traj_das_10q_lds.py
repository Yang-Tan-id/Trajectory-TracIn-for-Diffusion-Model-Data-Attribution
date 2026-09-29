"""Evaluate higher-noise-aligned fully-unrolled trajectory DAS on q00-q09."""

import json

import numpy as np
from scipy.stats import spearmanr

from unrolled_traj_das_config import *


def main():
    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    output = {
        "method": UNROLLED_TRAJ_DAS_METHOD,
        "query_ids": list(UNROLLED_TRAJ_DAS_QUERY_IDS),
        "query_count": len(UNROLLED_TRAJ_DAS_QUERY_IDS),
        "trajectory_probe_count": int(UNROLLED_TRAJ_DAS_PROBES),
        "projection_dim": int(UNROLLED_TRAJ_DAS_PROJECTION_DIM),
        "das_timestamp_count": len(DAS_TIMESTEPS),
        "das_outer_mc": int(DAS_NUM_MC),
        "train_gradient_mc": int(DAS_TRAIN_GRAD_MC),
        "timestamp_alignment": "for each trajectory state t, average over training loss timestamps s>=t",
        "lambdas": [float(value) for value in DAS_LAMBDAS],
        "results": {},
    }
    for lam_raw in DAS_LAMBDAS:
        lam = float(lam_raw)
        scores = [
            np.load(
                ATTR_DIR
                / UNROLLED_TRAJ_DAS_METHOD
                / f"q{query_id:02d}"
                / f"lambda_{lambda_tag(lam)}"
                / "scores.npy"
            ).astype(np.float64)
            for query_id in UNROLLED_TRAJ_DAS_QUERY_IDS
        ]
        lambda_result = {}
        print(f"\nLAMBDA={lam:g}", flush=True)
        for metric in LDS_METRICS:
            observed = np.load(LDS_DIR / f"observed_{metric}.npy").astype(np.float64)
            positive = np.asarray(
                [
                    spearmanr(
                        membership @ scores[position], observed[query_id]
                    ).statistic
                    for position, query_id in enumerate(
                        UNROLLED_TRAJ_DAS_QUERY_IDS
                    )
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
            lambda_result[metric] = signs
            print(
                f"{metric:30s} "
                f"sign=-1 {signs['negative']['mean']:+.6f}±"
                f"{signs['negative']['std']:.6f} | "
                f"sign=+1 {signs['positive']['mean']:+.6f}±"
                f"{signs['positive']['std']:.6f}",
                flush=True,
            )
        output["results"][lambda_tag(lam)] = lambda_result
    path = LDS_DIR / (
        "unrolled_trajectory_higher_noise_avg_probe4_10q_lambda_sweep.json"
    )
    with open(path, "w") as handle:
        json.dump(output, handle, indent=2)
    print(f"[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
