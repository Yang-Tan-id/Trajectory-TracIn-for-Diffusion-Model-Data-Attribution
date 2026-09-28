"""Evaluate all-100-query reference-forward-10 aligned-loss TracIn LDS."""

import json

import numpy as np
from scipy.stats import spearmanr

from reference_forward10_config import *


def main():
    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    output = {
        "query_ids": list(range(100)),
        "query_count": 100,
        "checkpoint_count": 50,
        "reference_timestamp_count": 100,
        "delta_t": REF_FORWARD10_DELTA_T,
        "maximum_one_based_loss_timestep": T + REF_FORWARD10_DELTA_T,
        "query_train_noise_aligned": True,
        "query_scalar": (
            "dot(predicted_noise_at_t_plus_10, unit(aligned_noise))"
        ),
        "query_uses_loss": False,
        "train_uses_diffusion_loss": True,
        "methods": {},
    }
    for contraction, method in REF_FORWARD10_METHODS.items():
        scores = [
            np.load(ATTR_DIR / method / f"q{query_id:02d}" / "scores.npy").astype(
                np.float64
            )
            for query_id in range(100)
        ]
        method_result = {"contraction": contraction, "metrics": {}}
        print(f"\nMETHOD: {method}", flush=True)
        for metric in LDS_METRICS:
            observed = np.load(LDS_DIR / f"observed_{metric}.npy").astype(np.float64)
            positive = np.asarray(
                [
                    spearmanr(
                        membership @ scores[query_id], observed[query_id]
                    ).statistic
                    for query_id in range(100)
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
            method_result["metrics"][metric] = signs
            print(
                f"{metric:30s} "
                f"sign=-1 {signs['negative']['mean']:+.6f}±{signs['negative']['std']:.6f} | "
                f"sign=+1 {signs['positive']['mean']:+.6f}±{signs['positive']['std']:.6f}",
                flush=True,
            )
        output["methods"][method] = method_result
    path = LDS_DIR / "reference_forward10_prednoise_aligned_loss_100q.json"
    with open(path, "w") as handle:
        json.dump(output, handle, indent=2)
    print(f"[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
