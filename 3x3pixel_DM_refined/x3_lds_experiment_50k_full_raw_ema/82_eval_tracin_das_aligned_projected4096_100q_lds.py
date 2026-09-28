"""Evaluate aligned checkpoint-noise projected4096 TracIn-DAS on q00-q99."""

import json

import numpy as np
from scipy.stats import spearmanr

from tracin_das_config import *


def main():
    methods = tracin_das_methods("checkpoint", "projected4096", "aligned")
    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    output = {
        "query_ids": list(TRACIN_DAS_ALL_QUERY_IDS),
        "noise_mode": "checkpoint",
        "parameter_projection": "projected4096",
        "parameter_projection_dim": TRACIN_PROJ_DIM,
        "train_noise_mode": "aligned",
        "methods": {},
    }
    for contraction, method in methods.items():
        scores = [
            np.load(ATTR_DIR / method / f"q{query_id:02d}" / "scores.npy").astype(
                np.float64
            )
            for query_id in TRACIN_DAS_ALL_QUERY_IDS
        ]
        method_result = {"contraction": contraction, "metrics": {}}
        print(f"\nMETHOD: {method}", flush=True)
        for metric in LDS_METRICS:
            observed = np.load(LDS_DIR / f"observed_{metric}.npy").astype(np.float64)
            signs = {}
            for sign_name, sign in (("negative", -1.0), ("positive", 1.0)):
                per_query = [
                    float(
                        spearmanr(
                            sign * (membership @ scores[query_id]),
                            observed[query_id],
                        ).statistic
                    )
                    for query_id in TRACIN_DAS_ALL_QUERY_IDS
                ]
                signs[sign_name] = {
                    "mean": float(np.nanmean(per_query)),
                    "per_query": per_query,
                }
            method_result["metrics"][metric] = signs
            print(
                f"{metric:30s} sign=-1 {signs['negative']['mean']:+.6f} | "
                f"sign=+1 {signs['positive']['mean']:+.6f}",
                flush=True,
            )
        output["methods"][method] = method_result

    LDS_DIR.mkdir(parents=True, exist_ok=True)
    path = LDS_DIR / (
        "tracin_das_endpoint_next_delta_checkpoint_noise_projected4096_"
        "aligned_q00_q99.json"
    )
    with open(path, "w") as handle:
        json.dump(output, handle, indent=2)
    print(f"[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
