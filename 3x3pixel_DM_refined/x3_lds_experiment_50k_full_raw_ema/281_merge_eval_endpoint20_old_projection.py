"""Evaluate endpoint20 under the old projection and compare projection seeds."""

import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr, wilcoxon

from endpoint20_meanloss_pairing_config import *


ROOT_NAME = "_tracin_das_endpoint20_old_projection_per_t_10q_shards"
METHOD_PREFIX = (
    "tracin_das_endpoint20_inverse_noise_10ckpt_20t_per_timestamp_aligned_"
    "adamw_full_next_delta_projected4096_old_trajectory_projection_raw"
)
BASELINE_PATH = (
    LDS_DIR / "tracin_das_endpoint20_inverse_noise_pairing_q00_q09.json"
)


def shard_root(checkpoint_shard_index, checkpoint_shard_count):
    return (
        ATTR_DIR
        / ROOT_NAME
        / f"checkpoint_shard_{int(checkpoint_shard_index):02d}_of_"
          f"{int(checkpoint_shard_count):02d}"
    )


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary, path)


def paired_test(candidate, baseline):
    difference = candidate - baseline
    finite = difference[np.isfinite(difference)]
    try:
        pvalue = float(wilcoxon(finite, alternative="two-sided").pvalue)
    except ValueError:
        pvalue = 1.0
    return {
        "mean_difference_old_minus_new_projection": float(np.nanmean(difference)),
        "wilcoxon_two_sided_p": pvalue,
        "old_projection_wins": int(np.count_nonzero(difference > 0)),
        "query_count": int(len(finite)),
        "difference_per_query": difference.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-shard-count", type=int, default=4)
    args = parser.parse_args()

    score_shape = (len(E20_QUERY_IDS), N_TRAIN)
    response_shape = (len(E20_QUERY_IDS), len(E20_TIMESTEPS), N_TRAIN)
    linear = np.zeros(score_shape, dtype=np.float64)
    termwise = np.zeros(score_shape, dtype=np.float64)
    response = np.zeros(response_shape, dtype=np.float32)
    covered = []
    for shard_index in range(args.checkpoint_shard_count):
        root = shard_root(shard_index, args.checkpoint_shard_count)
        with open(root / "done.json") as handle:
            info = json.load(handle)
        if info.get("projection_namespace") != "trajectory_bridge_projection":
            raise ValueError(f"unexpected projection namespace in {root}")
        covered.extend(int(value) for value in info["checkpoint_pairs"])
        with np.load(root / "partial_scores.npz") as partial:
            linear += partial["per_timestamp_aligned__linear"].astype(np.float64)
            termwise += partial[
                "per_timestamp_aligned__termwise_squared"
            ].astype(np.float64)
            response += partial[
                "per_timestamp_aligned__timestamp_response"
            ].astype(np.float32)
    if sorted(covered) != sorted(NPA_CHECKPOINT_PAIRS):
        raise ValueError(f"checkpoint coverage mismatch: {sorted(covered)}")

    scores = {
        "linear": linear,
        "termwise_squared": termwise,
        "timestamp_sum_squared": np.square(response).mean(axis=1),
    }
    scores_by_timestamp = np.square(response)
    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    observed = {
        metric: np.load(LDS_DIR / f"observed_{metric}.npy").astype(np.float64)[
            list(E20_QUERY_IDS)
        ]
        for metric in LDS_METRICS
    }
    with open(BASELINE_PATH) as handle:
        baseline = json.load(handle)

    result = {
        "query_ids": list(E20_QUERY_IDS),
        "checkpoint_pairs": list(NPA_CHECKPOINT_PAIRS),
        "timesteps": list(E20_TIMESTEPS),
        "projection_namespace": "trajectory_bridge_projection",
        "baseline_projection_namespace": "endpoint20_meanloss_projection",
        "results": {},
    }
    lines = [
        "ENDPOINT20 PROJECTION-SEED CONTROL",
        "Only projection namespace differs: old trajectory vs endpoint20.",
        "old-minus-new paired comparison; q00-q09; sign=-1",
        "",
    ]
    for contraction, method_scores in scores.items():
        method = f"{METHOD_PREFIX}_{contraction}"
        result["results"][contraction] = {
            "method": method,
            "targets": {},
        }
        for position, query_id in enumerate(E20_QUERY_IDS):
            output = ATTR_DIR / method / f"q{query_id:02d}"
            output.mkdir(parents=True, exist_ok=True)
            np.save(output / "scores.npy", method_scores[position])
            if contraction == "timestamp_sum_squared":
                np.save(
                    output / "scores_by_timestamp.npy",
                    scores_by_timestamp[position],
                )
        prediction = membership @ method_scores.T
        lines.extend(
            [
                f"[{contraction}]",
                "target                              old-proj   new-proj   "
                "delta       p       wins",
            ]
        )
        for metric in LDS_METRICS:
            candidate = np.asarray(
                [
                    spearmanr(-prediction[:, q], observed[metric][q]).statistic
                    for q in range(len(E20_QUERY_IDS))
                ],
                dtype=np.float64,
            )
            reference = np.asarray(
                baseline["results"][contraction]["per_timestamp_aligned"]
                ["targets"][metric]["per_query"],
                dtype=np.float64,
            )
            test = paired_test(candidate, reference)
            result["results"][contraction]["targets"][metric] = {
                "old_projection": {
                    "mean": float(np.nanmean(candidate)),
                    "std": float(np.nanstd(candidate)),
                    "per_query": candidate.tolist(),
                },
                "new_projection": {
                    "mean": float(np.nanmean(reference)),
                    "std": float(np.nanstd(reference)),
                    "per_query": reference.tolist(),
                },
                "paired": test,
            }
            lines.append(
                f"{metric:34s} {np.nanmean(candidate):+.6f} "
                f"{np.nanmean(reference):+.6f} "
                f"{test['mean_difference_old_minus_new_projection']:+.6f} "
                f"{test['wilcoxon_two_sided_p']:.6g} "
                f"{test['old_projection_wins']:02d}/10"
            )
        lines.append("")

    json_path = LDS_DIR / "tracin_das_endpoint20_projection_seed_control_q00_q09.json"
    text_path = json_path.with_suffix(".txt")
    atomic_json(json_path, result)
    text_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[saved] {json_path}")
    print(f"[saved] {text_path}")


if __name__ == "__main__":
    main()
