"""Evaluate the 100-query trajectory inverse-noise DAS lambda sweep."""

import json

import numpy as np
from scipy.stats import spearmanr

from trajectory_inverse_noise_das_config import *


def main():
    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    output = {
        "method": TRAJECTORY_INVERSE_DAS_100Q_METHOD,
        "query_ids": list(TRAJECTORY_INVERSE_DAS_ALL_QUERY_IDS),
        "query_count": 100,
        "families": {"prompted": 75, "unprompted": 25},
        "parameter_source": "final EMA per family",
        "projection_dim": TRAJECTORY_INVERSE_DAS_PROJ_DIM,
        "outer_probe_count": int(DAS_NUM_MC),
        "endpoint_excluded": True,
        "included_timestamp_indices": list(range(99)),
        "lambdas": [float(value) for value in DAS_LAMBDAS],
        "results": {},
    }
    for lam_raw in DAS_LAMBDAS:
        lam = float(lam_raw)
        scores = [
            np.load(
                ATTR_DIR
                / TRAJECTORY_INVERSE_DAS_100Q_METHOD
                / f"q{query_id:02d}"
                / f"lambda_{lambda_tag(lam)}"
                / "scores.npy"
            ).astype(np.float64)
            for query_id in TRAJECTORY_INVERSE_DAS_ALL_QUERY_IDS
        ]
        lambda_result = {}
        print(f"\nLAMBDA={lam:g}", flush=True)
        for metric in LDS_METRICS:
            observed = np.load(LDS_DIR / f"observed_{metric}.npy").astype(
                np.float64
            )
            positive = np.asarray(
                [
                    spearmanr(membership @ scores[query_id], observed[query_id]).statistic
                    for query_id in TRAJECTORY_INVERSE_DAS_ALL_QUERY_IDS
                ],
                dtype=np.float64,
            )
            signs = {}
            for sign_name, values in (("negative", -positive), ("positive", positive)):
                signs[sign_name] = {
                    "mean": float(np.nanmean(values)),
                    "std": float(np.nanstd(values)),
                    "prompted_mean": float(np.nanmean(values[:75])),
                    "unprompted_mean": float(np.nanmean(values[75:])),
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
    path = LDS_DIR / "trajectory_inverse_noise_das_100q_lambda_sweep.json"
    with open(path, "w") as handle:
        json.dump(output, handle, indent=2)
    print(f"[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
