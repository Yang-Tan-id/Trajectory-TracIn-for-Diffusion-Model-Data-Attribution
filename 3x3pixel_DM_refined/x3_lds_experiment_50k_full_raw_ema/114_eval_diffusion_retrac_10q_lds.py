"""Evaluate replayed Diffusion-TracIn and Diffusion-ReTrac on all LDS targets."""

import json

import numpy as np
from scipy.stats import spearmanr

from diffusion_retrac_config import *


def main():
    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    output = {
        "query_ids": list(RETRAC_QUERY_IDS),
        "query_count": len(RETRAC_QUERY_IDS),
        "methods": {},
    }
    for label, method in RETRAC_METHODS.items():
        scores = np.stack(
            [
                np.load(ATTR_DIR / method / f"q{query_id:02d}" / "scores.npy")
                for query_id in RETRAC_QUERY_IDS
            ]
        ).astype(np.float64)
        method_result = {}
        print(f"\nMETHOD: {method}", flush=True)
        for metric in LDS_METRICS:
            observed = np.load(LDS_DIR / f"observed_{metric}.npy").astype(np.float64)
            positive = np.asarray(
                [
                    spearmanr(membership @ scores[position], observed[query_id]).statistic
                    for position, query_id in enumerate(RETRAC_QUERY_IDS)
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
                f"sign=-1 {signs['negative']['mean']:+.6f}±{signs['negative']['std']:.6f} | "
                f"sign=+1 {signs['positive']['mean']:+.6f}±{signs['positive']['std']:.6f}",
                flush=True,
            )
        output["methods"][label] = {"artifact": method, "targets": method_result}
    path = LDS_DIR / "diffusion_tracin_vs_retrac_adamw_full_replayed_100q.json"
    with open(path, "w") as handle:
        json.dump(output, handle, indent=2)
    print(f"[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
