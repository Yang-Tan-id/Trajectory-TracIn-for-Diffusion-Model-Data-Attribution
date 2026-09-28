"""Evaluate q00-q98 aligned-noise DAS for all lambdas and four targets."""

import json

import numpy as np
from scipy.stats import spearmanr

from exp_config import *
from tracin_das_config import TRACIN_DAS_FIRST99_QUERY_IDS


METHOD = "das_ema_aligned_noise"
TARGETS = {
    "simple_loss": ("simple_loss_ema", "simple_loss_raw"),
    "traj_ref": ("traj_ref_ema", "traj_ref_raw"),
    "endpoint_deviation": ("endpoint_deviation_ema", "endpoint_deviation_raw"),
    "trajectory_state_mse": ("trajectory_state_mse_ema", "trajectory_state_mse_raw"),
}


def tag(value):
    return str(float(value)).replace(".", "p")


def main():
    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    observed = {
        metric: np.load(LDS_DIR / f"observed_{metric}.npy").astype(np.float64)
        for metrics in TARGETS.values()
        for metric in metrics
    }
    output = {
        "method": METHOD,
        "query_ids": list(TRACIN_DAS_FIRST99_QUERY_IDS),
        "noise_alignment": "same noise on query and train sides per timestamp/MC",
        "lambdas": [float(value) for value in DAS_LAMBDAS],
        "targets": TARGETS,
        "results": {},
    }
    for lam_raw in DAS_LAMBDAS:
        lam = float(lam_raw)
        scores = np.stack(
            [
                np.load(
                    ATTR_DIR
                    / METHOD
                    / f"q{query_id:02d}"
                    / f"lambda_{tag(lam)}"
                    / "scores.npy"
                ).astype(np.float64)
                for query_id in TRACIN_DAS_FIRST99_QUERY_IDS
            ],
            axis=0,
        )
        predictions = membership @ scores.T
        lambda_result = {}
        print(f"\nLAMBDA={lam:g}", flush=True)
        for target_name, metrics in TARGETS.items():
            target_result = {}
            for metric in metrics:
                signs = {}
                for sign_name, sign in (("negative", -1.0), ("positive", 1.0)):
                    per_query = [
                        float(
                            spearmanr(
                                sign * predictions[:, query_id],
                                observed[metric][query_id],
                            ).statistic
                        )
                        for query_id in TRACIN_DAS_FIRST99_QUERY_IDS
                    ]
                    signs[sign_name] = {
                        "mean": float(np.nanmean(per_query)),
                        "per_query": per_query,
                    }
                target_result[metric] = signs
                print(
                    f"  {metric:28s} -1={signs['negative']['mean']:+.6f} "
                    f"+1={signs['positive']['mean']:+.6f}",
                    flush=True,
                )
            lambda_result[target_name] = target_result
        output["results"][tag(lam)] = lambda_result

    path = LDS_DIR / "das_ema_aligned_noise_99q_lambda_sweep.json"
    with open(path, "w") as handle:
        json.dump(output, handle, indent=2)
    print(f"[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
