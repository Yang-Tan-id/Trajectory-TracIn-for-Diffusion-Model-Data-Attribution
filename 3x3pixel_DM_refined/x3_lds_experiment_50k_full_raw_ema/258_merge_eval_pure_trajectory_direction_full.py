"""Merge full pure trajectory-direction scores and compare with aligned AdamW."""

import argparse
import json

import numpy as np
from scipy.stats import spearmanr

from exp_config import ATTR_DIR, LDS_DIR, LDS_METRICS, MASK_DIR, N_TRAIN


QUERY_IDS = tuple(range(10))
CONTRACTIONS = ("linear", "termwise_squared", "timestamp_sum_squared")
GROUPS = ("all", "q1", "q2", "q3", "q4")
ROOT = ATTR_DIR / "_tracin_das_pure_trajectory_direction_full_q00_q09_shards"
BASELINE_PATH = (
    LDS_DIR / "tracin_das_norm4_gradient_and_adamw_full_q00_q99.json"
)


def method_name(contraction, group):
    return (
        "tracin_das_pure_trajectory_direction_"
        "49pair_100t_mc1_adamw_full_next_delta_projected4096_"
        f"raw_{contraction}_{group}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-shard-count", type=int, default=4)
    args = ap.parse_args()
    scores = {
        contraction: {
            group: np.empty((len(QUERY_IDS), N_TRAIN), dtype=np.float64)
            for group in GROUPS
        }
        for contraction in CONTRACTIONS
    }
    for shard_index in range(args.query_shard_count):
        root = ROOT / (
            f"query_shard_{shard_index:02d}_of_{args.query_shard_count:02d}"
        )
        with open(root / "done.json") as handle:
            info = json.load(handle)
        query_ids = [int(value) for value in info["query_ids"]]
        with np.load(root / "partial_scores.npz") as values:
            for local, query_id in enumerate(query_ids):
                for contraction in CONTRACTIONS:
                    for group in GROUPS:
                        scores[contraction][group][query_id] = values[
                            f"{contraction}__{group}"
                        ][local]

    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    result = {
        "query_ids": list(QUERY_IDS),
        "checkpoint_pairs": 49,
        "timestamps": 100,
        "noise": "pure endpoint-to-reference-state implied noise",
        "results": {},
        "comparison_to_aligned_adamw": {},
    }
    for contraction in CONTRACTIONS:
        result["results"][contraction] = {}
        for group in GROUPS:
            method = method_name(contraction, group)
            for query_id in QUERY_IDS:
                output = ATTR_DIR / method / f"q{query_id:02d}"
                output.mkdir(parents=True, exist_ok=True)
                np.save(output / "scores.npy", scores[contraction][group][query_id])
            prediction = membership @ scores[contraction][group].T
            entry = {"method": method, "targets": {}}
            for metric in LDS_METRICS:
                observed = np.load(
                    LDS_DIR / f"observed_{metric}.npy"
                ).astype(np.float64)[list(QUERY_IDS)]
                per_query = [
                    float(spearmanr(-prediction[:, q], observed[q]).statistic)
                    for q in range(len(QUERY_IDS))
                ]
                entry["targets"][metric] = {
                    "negative": {
                        "mean": float(np.nanmean(per_query)),
                        "std": float(np.nanstd(per_query)),
                        "per_query": per_query,
                    }
                }
            result["results"][contraction][group] = entry

    with open(BASELINE_PATH) as handle:
        baseline = json.load(handle)
    print("PURE TRAJECTORY DIRECTION vs 100x1 ALIGNED FULL-ADAMW (q00-q09)")
    for contraction in CONTRACTIONS:
        print(f"\n{contraction} / all timestamps")
        result["comparison_to_aligned_adamw"][contraction] = {}
        for metric in LDS_METRICS:
            pure = np.asarray(
                result["results"][contraction]["all"]["targets"][metric]
                ["negative"]["per_query"]
            )
            aligned = np.asarray(
                baseline["variants"]["adamw_full"]["raw"][contraction]
                ["targets"][metric]["negative"]["per_query"][:10]
            )
            delta = pure - aligned
            comparison = {
                "pure_mean": float(np.nanmean(pure)),
                "aligned_mean": float(np.nanmean(aligned)),
                "paired_delta_mean": float(np.nanmean(delta)),
                "paired_delta_per_query": delta.tolist(),
            }
            result["comparison_to_aligned_adamw"][contraction][metric] = comparison
            print(
                f"{metric:30s} pure={np.nanmean(pure):+.6f} "
                f"aligned={np.nanmean(aligned):+.6f} "
                f"delta={np.nanmean(delta):+.6f}"
            )

    output = LDS_DIR / "tracin_das_pure_trajectory_direction_full_q00_q09.json"
    temporary = output.with_suffix(".tmp.json")
    with open(temporary, "w") as handle:
        json.dump(result, handle, indent=2)
    temporary.replace(output)
    print(f"\n[saved] {output}")


if __name__ == "__main__":
    main()
