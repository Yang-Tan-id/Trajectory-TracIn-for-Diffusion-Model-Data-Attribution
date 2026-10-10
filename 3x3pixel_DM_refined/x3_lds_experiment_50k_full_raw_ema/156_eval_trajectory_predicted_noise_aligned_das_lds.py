"""Evaluate all lambdas and LDS targets for predicted-noise aligned DAS."""

import json

import numpy as np
from scipy.stats import spearmanr

from trajectory_predicted_noise_aligned_das_config import *


def main():
    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    output = {
        "method": TPNA_DAS_METHOD,
        "query_ids": list(TPNA_DAS_QUERY_IDS),
        "parameter_source": "final EMA",
        "projection_dim": TPNA_DAS_PROJ_DIM,
        "outer_probe_count": int(DAS_NUM_MC),
        "endpoint_excluded": True,
        "lambdas": [float(value) for value in DAS_LAMBDAS],
        "results": {},
    }
    for lam_raw in DAS_LAMBDAS:
        lam = float(lam_raw)
        scores = [
            np.load(
                ATTR_DIR
                / TPNA_DAS_METHOD
                / f"q{query_id:02d}"
                / f"lambda_{lambda_tag(lam)}"
                / "scores.npy"
            ).astype(np.float64)
            for query_id in TPNA_DAS_QUERY_IDS
        ]
        lambda_result = {}
        print(f"\nLAMBDA={lam:g}", flush=True)
        for metric in LDS_METRICS:
            observed = np.load(LDS_DIR / f"observed_{metric}.npy").astype(
                np.float64
            )
            positive = np.asarray(
                [
                    spearmanr(
                        membership @ scores[position], observed[query_id]
                    ).statistic
                    for position, query_id in enumerate(TPNA_DAS_QUERY_IDS)
                ]
            )
            signs = {
                sign_name: {
                    "mean": float(np.nanmean(values)),
                    "std": float(np.nanstd(values)),
                    "per_query": values.tolist(),
                }
                for sign_name, values in (
                    ("negative", -positive),
                    ("positive", positive),
                )
            }
            lambda_result[metric] = signs
            print(
                f"{metric:30s} sign=-1 "
                f"{signs['negative']['mean']:+.6f}±"
                f"{signs['negative']['std']:.6f} | sign=+1 "
                f"{signs['positive']['mean']:+.6f}±"
                f"{signs['positive']['std']:.6f}",
                flush=True,
            )
        output["results"][lambda_tag(lam)] = lambda_result
    TPNA_DAS_LDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TPNA_DAS_LDS_PATH, "w") as handle:
        json.dump(output, handle, indent=2)
    print(f"[saved] {TPNA_DAS_LDS_PATH}", flush=True)


if __name__ == "__main__":
    main()
