"""Evaluate both signs of the last-noise delta-direction score."""

import json

import numpy as np

from checkpoint_counterfactual_config import CF_DIRECTION_SCORE_METHOD
from checkpoint_counterfactual_metrics import spearman_correlation
from exp_config import ATTR_DIR, LDS_DIR, MASK_DIR


METRICS = (
    "endpoint_deviation_ema",
    "endpoint_deviation_raw",
    "trajectory_state_mse_ema",
    "trajectory_state_mse_raw",
)


def main():
    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    output = {"method": CF_DIRECTION_SCORE_METHOD, "results": {}}
    for metric in METRICS:
        observed = np.load(LDS_DIR / f"observed_{metric}.npy").astype(np.float64)
        by_sign = {}
        for sign_name, sign in (("saved_score", 1.0), ("negated_score", -1.0)):
            query_values = []
            for query_id in range(observed.shape[0]):
                score = np.load(
                    ATTR_DIR / CF_DIRECTION_SCORE_METHOD / f"q{query_id:02d}" / "scores.npy"
                ).astype(np.float64)
                prediction = sign * (membership @ score)
                rho = spearman_correlation(prediction, observed[query_id])
                query_values.append({"query_id": query_id, "spearman": rho})
            mean = float(np.nanmean([item["spearman"] for item in query_values]))
            by_sign[sign_name] = {"multiplier": sign, "mean": mean, "queries": query_values}
            print(f"[LDS] {metric} {sign_name}: mean={mean:.6f}", flush=True)
        output["results"][metric] = by_sign
    path = LDS_DIR / f"{CF_DIRECTION_SCORE_METHOD}_both_signs.json"
    with open(path, "w") as handle:
        json.dump(output, handle, indent=2)
    print(f"[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
