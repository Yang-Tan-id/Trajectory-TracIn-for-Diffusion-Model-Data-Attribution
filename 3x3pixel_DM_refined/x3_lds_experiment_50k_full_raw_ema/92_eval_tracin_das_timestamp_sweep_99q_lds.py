"""Evaluate the q00-q98 timestamp-count sweep for all targets and signs."""

import json

import numpy as np
from scipy.stats import spearmanr

from tracin_das_config import *


TARGETS = {
    "simple_loss": ("simple_loss_ema", "simple_loss_raw"),
    "traj_ref": ("traj_ref_ema", "traj_ref_raw"),
    "endpoint_deviation": ("endpoint_deviation_ema", "endpoint_deviation_raw"),
    "trajectory_state_mse": ("trajectory_state_mse_ema", "trajectory_state_mse_raw"),
}


def main():
    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    observed = {
        metric: np.load(LDS_DIR / f"observed_{metric}.npy").astype(np.float64)
        for metrics in TARGETS.values()
        for metric in metrics
    }
    output = {
        "query_ids": list(TRACIN_DAS_FIRST99_QUERY_IDS),
        "checkpoint_bank_count": 50,
        "actual_transition_count": 49,
        "timestamp_counts": list(TRACIN_DAS_TIMESTAMP_COUNTS),
        "learning_rate_source": "source checkpoint saved eta",
        "targets": TARGETS,
        "results": {},
    }
    for count in TRACIN_DAS_TIMESTAMP_COUNTS:
        count_result = {
            "selected_timestamp_indices": list(tracin_das_timestamp_indices(count)),
            "methods": {},
        }
        for contraction, method in tracin_das_checkpoint_lr_timestamp_methods(count).items():
            scores = np.stack(
                [
                    np.load(
                        ATTR_DIR / method / f"q{query_id:02d}" / "scores.npy"
                    ).astype(np.float64)
                    for query_id in TRACIN_DAS_FIRST99_QUERY_IDS
                ],
                axis=0,
            )
            predictions = membership @ scores.T
            method_result = {"contraction": contraction, "targets": {}}
            print(f"\nTIMESTAMPS={count} METHOD={method}", flush=True)
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
                method_result["targets"][target_name] = target_result
            count_result["methods"][method] = method_result
        output["results"][str(count)] = count_result
    path = LDS_DIR / "tracin_das_checkpoint_lr_99q_timestamp_sweep.json"
    with open(path, "w") as handle:
        json.dump(output, handle, indent=2)
    print(f"[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
