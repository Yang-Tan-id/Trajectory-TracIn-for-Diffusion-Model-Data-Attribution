"""Evaluate both signs of all three TracIn-DAS contractions on q00-q09."""

import json

import numpy as np
from scipy.stats import spearmanr

from tracin_das_config import *


def main():
    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    output = {"methods": {}}
    for contraction, method in TRACIN_DAS_METHODS.items():
        method_result = {"contraction": contraction, "metrics": {}}
        scores = [
            np.load(ATTR_DIR / method / f"q{query_id:02d}" / "scores.npy").astype(np.float64)
            for query_id in TRACIN_DAS_QUERY_IDS
        ]
        print(f"\nMETHOD: {method}", flush=True)
        for metric in LDS_METRICS:
            observed = np.load(LDS_DIR / f"observed_{metric}.npy").astype(np.float64)
            signs = {}
            for sign_name, sign in (("negative", -1.0), ("positive", 1.0)):
                per_query = []
                for query_position, query_id in enumerate(TRACIN_DAS_QUERY_IDS):
                    prediction = sign * (membership @ scores[query_position])
                    per_query.append(
                        float(spearmanr(prediction, observed[query_id]).statistic)
                    )
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
    path = LDS_DIR / "tracin_das_endpoint_next_delta_checkpoint_noise_q00_q09.json"
    with open(path, "w") as handle:
        json.dump(output, handle, indent=2)
    print(f"[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
