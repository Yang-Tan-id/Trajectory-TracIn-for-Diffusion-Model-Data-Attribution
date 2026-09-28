"""Evaluate endpoint-noise query-MC5/train-MC1 TracIn-DAS on q00-q99."""

import json

import numpy as np
from scipy.stats import spearmanr

from tracin_das_config import *


QUERY_MC = 5
TRAIN_NOISE_MODE = "independent-mc1"


def main():
    methods = tracin_das_methods(
        "checkpoint",
        "projected4096",
        TRAIN_NOISE_MODE,
        QUERY_MC,
    )
    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    output = {
        "query_ids": list(TRACIN_DAS_ALL_QUERY_IDS),
        "query_count": len(TRACIN_DAS_ALL_QUERY_IDS),
        "checkpoint_count": 50,
        "transition_count": 49,
        "timestamp_count": 100,
        "noise_mode": "checkpoint",
        "parameter_projection": "projected4096",
        "parameter_projection_dim": TRACIN_PROJ_DIM,
        "query_mc": QUERY_MC,
        "train_noise_mode": TRAIN_NOISE_MODE,
        "train_mc": 1,
        "methods": {},
    }
    for contraction, method in methods.items():
        scores = [
            np.load(
                ATTR_DIR / method / f"q{query_id:02d}" / "scores.npy"
            ).astype(np.float64)
            for query_id in TRACIN_DAS_ALL_QUERY_IDS
        ]
        method_result = {"contraction": contraction, "metrics": {}}
        print(f"\nMETHOD: {method}", flush=True)
        for metric in LDS_METRICS:
            observed = np.load(LDS_DIR / f"observed_{metric}.npy").astype(
                np.float64
            )
            positive = np.asarray(
                [
                    spearmanr(
                        membership @ scores[query_id], observed[query_id]
                    ).statistic
                    for query_id in TRACIN_DAS_ALL_QUERY_IDS
                ],
                dtype=np.float64,
            )
            signs = {}
            for sign_name, values in (
                ("negative", -positive),
                ("positive", positive),
            ):
                signs[sign_name] = {
                    "mean": float(np.nanmean(values)),
                    "std": float(np.nanstd(values)),
                    "per_query": values.tolist(),
                }
            method_result["metrics"][metric] = signs
            print(
                f"{metric:30s} "
                f"sign=-1 {signs['negative']['mean']:+.6f}±{signs['negative']['std']:.6f} | "
                f"sign=+1 {signs['positive']['mean']:+.6f}±{signs['positive']['std']:.6f}",
                flush=True,
            )
        output["methods"][method] = method_result

    LDS_DIR.mkdir(parents=True, exist_ok=True)
    path = LDS_DIR / (
        "tracin_das_endpoint_next_delta_checkpoint_projected4096_"
        "query_mc5_independent_mc1_q00_q99.json"
    )
    with open(path, "w") as handle:
        json.dump(output, handle, indent=2)
    print(f"[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
