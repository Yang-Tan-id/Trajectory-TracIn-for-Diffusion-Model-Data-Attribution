"""Evaluate trajectory inverse-noise TracIn-DAS on all LDS targets."""

import json

import numpy as np
from scipy.stats import spearmanr

from trajectory_inverse_noise_tracin_das_config import *


def main():
    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    output = {
        "query_ids": list(INVERSE_NOISE_QUERY_IDS),
        "query_count": len(INVERSE_NOISE_QUERY_IDS),
        "methods": INVERSE_NOISE_METHODS,
        "endpoint_excluded": True,
        "included_timestamp_indices": list(range(99)),
        "timestamp_weight": 1.0 / 99.0,
        "results": {},
    }
    for contraction, method in INVERSE_NOISE_METHODS.items():
        scores = [
            np.load(ATTR_DIR / method / f"q{query_id:02d}" / "scores.npy").astype(
                np.float64
            )
            for query_id in INVERSE_NOISE_QUERY_IDS
        ]
        method_result = {}
        print(f"\nMETHOD: {method}", flush=True)
        for metric in LDS_METRICS:
            observed = np.load(LDS_DIR / f"observed_{metric}.npy").astype(
                np.float64
            )
            positive = np.asarray(
                [
                    spearmanr(membership @ scores[position], observed[query_id]).statistic
                    for position, query_id in enumerate(INVERSE_NOISE_QUERY_IDS)
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
        output["results"][contraction] = method_result
    path = LDS_DIR / "trajectory_inverse_noise_tracin_das_q00_q09.json"
    with open(path, "w") as handle:
        json.dump(output, handle, indent=2)
    print(f"[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
