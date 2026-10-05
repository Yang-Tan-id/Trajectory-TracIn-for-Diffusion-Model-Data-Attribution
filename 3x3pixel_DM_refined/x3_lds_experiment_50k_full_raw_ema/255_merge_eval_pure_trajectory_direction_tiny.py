"""Merge and evaluate q00-q03 pure trajectory-direction tiny experiment."""

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from exp_config import ATTR_DIR, LDS_DIR, LDS_METRICS, MASK_DIR, N_TRAIN


QUERY_IDS = tuple(range(4))
CONTRACTIONS = ("linear", "termwise_squared", "timestamp_sum_squared")
ROOT = ATTR_DIR / "_tracin_das_pure_trajectory_direction_tiny_shards"


def method_name(contraction):
    return (
        "tracin_das_pure_trajectory_direction_"
        "10ckpt_5smallt_mc1_adamw_full_next_delta_projected4096_"
        f"raw_{contraction}"
    )


def main():
    scores = {
        contraction: np.empty((len(QUERY_IDS), N_TRAIN), dtype=np.float64)
        for contraction in CONTRACTIONS
    }
    for position, query_id in enumerate(QUERY_IDS):
        path = ROOT / f"q{query_id:02d}" / "partial_scores.npz"
        with np.load(path) as values:
            for contraction in CONTRACTIONS:
                scores[contraction][position] = values[
                    f"{contraction}__small"
                ][0]

    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    result = {
        "query_ids": list(QUERY_IDS),
        "timestamp_indices": [0, 6, 12, 18, 24],
        "noise": "pure endpoint-to-reference-state implied noise",
        "results": {},
    }
    print("PURE TRAJECTORY-DIRECTION TINY TRACIN-DAS (sign=-1)")
    for contraction in CONTRACTIONS:
        method = method_name(contraction)
        for position, query_id in enumerate(QUERY_IDS):
            output = ATTR_DIR / method / f"q{query_id:02d}"
            output.mkdir(parents=True, exist_ok=True)
            np.save(output / "scores.npy", scores[contraction][position])
        prediction = membership @ scores[contraction].T
        entry = {"method": method, "targets": {}}
        print(f"\n{contraction}")
        for metric in LDS_METRICS:
            observed = np.load(LDS_DIR / f"observed_{metric}.npy").astype(
                np.float64
            )[list(QUERY_IDS)]
            per_query = [
                float(spearmanr(-prediction[:, p], observed[p]).statistic)
                for p in range(len(QUERY_IDS))
            ]
            entry["targets"][metric] = {
                "negative": {
                    "mean": float(np.nanmean(per_query)),
                    "std": float(np.nanstd(per_query)),
                    "per_query": per_query,
                }
            }
            print(
                f"{metric:30s} {np.nanmean(per_query):+.6f}"
                f" +/- {np.nanstd(per_query):.6f}"
            )
        result["results"][contraction] = entry

    output = LDS_DIR / "tracin_das_pure_trajectory_direction_tiny_q00_q03.json"
    temporary = output.with_suffix(".tmp.json")
    with open(temporary, "w") as handle:
        json.dump(result, handle, indent=2)
    temporary.replace(output)
    print(f"\n[saved] {output}")


if __name__ == "__main__":
    main()
