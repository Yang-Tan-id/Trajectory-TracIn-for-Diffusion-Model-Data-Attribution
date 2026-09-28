"""Evaluate q00-q09 reference-forward-10 next-delta TracIn LDS."""

import json

import numpy as np
from scipy.stats import spearmanr

from reference_forward10_config import *


QUERY_IDS = tuple(range(10))


def main():
    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    output = {
        "query_ids": list(QUERY_IDS),
        "query_count": len(QUERY_IDS),
        "checkpoint_count": 50,
        "checkpoint_transition_count": 49,
        "reference_timestamp_count": 100,
        "delta_t": REF_FORWARD10_DELTA_T,
        "maximum_one_based_loss_timestep": T + REF_FORWARD10_DELTA_T,
        "query_train_noise_aligned": True,
        "query_scalar": (
            "dot(epsilon_current, normalize(epsilon_next-epsilon_current)) "
            "at the same reference-forward-10 state"
        ),
        "checkpoint_target": "next",
        "query_uses_loss": False,
        "train_uses_diffusion_loss": True,
        "methods": {},
    }
    for contraction, method in REF_FORWARD10_METHODS.items():
        scores = [
            np.load(ATTR_DIR / method / f"q{query_id:02d}" / "scores.npy").astype(
                np.float64
            )
            for query_id in QUERY_IDS
        ]
        method_result = {"contraction": contraction, "metrics": {}}
        print(f"\nMETHOD: {method}", flush=True)
        for metric in LDS_METRICS:
            observed = np.load(LDS_DIR / f"observed_{metric}.npy").astype(np.float64)
            positive = np.asarray(
                [
                    spearmanr(
                        membership @ scores[position], observed[query_id]
                    ).statistic
                    for position, query_id in enumerate(QUERY_IDS)
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
    path = LDS_DIR / "reference_forward10_next_delta_aligned_loss_q00_q09.json"
    with open(path, "w") as handle:
        json.dump(output, handle, indent=2)
    print(f"[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
