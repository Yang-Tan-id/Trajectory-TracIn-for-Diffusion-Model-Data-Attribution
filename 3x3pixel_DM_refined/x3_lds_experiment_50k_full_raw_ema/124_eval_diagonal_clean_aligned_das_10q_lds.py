"""Evaluate the timestamp-diagonal predicted-clean DAS lambda sweep."""

import argparse
import json

import numpy as np
from scipy.stats import spearmanr

from diagonal_clean_das_config import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp-count", type=int, choices=(10, 100), default=100)
    args = parser.parse_args()
    method = diagonal_clean_method(args.timestamp_count)
    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    output = {
        "method": method,
        "query_ids": list(DIAGONAL_CLEAN_QUERY_IDS),
        "query_count": len(DIAGONAL_CLEAN_QUERY_IDS),
        "timestamp_count": args.timestamp_count,
        "timestamp_endpoint_pairing": "diagonal",
        "das_mc": int(DAS_NUM_MC),
        "lambdas": [float(value) for value in DAS_LAMBDAS],
        "results": {},
    }
    for lam_raw in DAS_LAMBDAS:
        lam = float(lam_raw)
        scores = [
            np.load(
                ATTR_DIR
                / method
                / f"q{query_id:02d}"
                / f"lambda_{lambda_tag(lam)}"
                / "scores.npy"
            ).astype(np.float64)
            for query_id in DIAGONAL_CLEAN_QUERY_IDS
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
                    for position, query_id in enumerate(DIAGONAL_CLEAN_QUERY_IDS)
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
                f"sign=-1 {signs['negative']['mean']:+.6f}±{signs['negative']['std']:.6f} | "
                f"sign=+1 {signs['positive']['mean']:+.6f}±{signs['positive']['std']:.6f}",
                flush=True,
            )
        output["results"][lambda_tag(lam)] = lambda_result
    path = LDS_DIR / (
        f"diagonal_clean_aligned_das_100q_{args.timestamp_count}timestamp_"
        "lambda_sweep.json"
    )
    with open(path, "w") as handle:
        json.dump(output, handle, indent=2)
    print(f"[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
