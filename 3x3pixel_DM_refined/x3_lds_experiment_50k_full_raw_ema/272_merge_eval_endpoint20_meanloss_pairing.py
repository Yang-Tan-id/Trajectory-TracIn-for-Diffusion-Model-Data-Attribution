"""Merge and evaluate endpoint20 per-t vs mean-noise/mean-loss pairing."""

import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr, wilcoxon

from endpoint20_meanloss_pairing_config import *


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-shard-count", type=int, default=4)
    args = parser.parse_args()
    score_shape = (len(E20_QUERY_IDS), N_TRAIN)
    response_shape = (len(E20_QUERY_IDS), len(E20_TIMESTEPS), N_TRAIN)
    linear = {mode: np.zeros(score_shape, dtype=np.float64) for mode in E20_MODES}
    termwise = {mode: np.zeros(score_shape, dtype=np.float64) for mode in E20_MODES}
    response = {
        mode: np.zeros(response_shape, dtype=np.float64) for mode in E20_MODES
    }
    covered = []
    for shard_index in range(args.checkpoint_shard_count):
        root = e20_shard_root(shard_index, args.checkpoint_shard_count)
        with open(root / "done.json") as handle:
            info = json.load(handle)
        covered.extend(int(value) for value in info["checkpoint_pairs"])
        with np.load(root / "partial_scores.npz") as partial:
            for mode in E20_MODES:
                linear[mode] += partial[f"{mode}__linear"].astype(np.float64)
                termwise[mode] += partial[
                    f"{mode}__termwise_squared"
                ].astype(np.float64)
                response[mode] += partial[
                    f"{mode}__timestamp_response"
                ].astype(np.float64)
    if sorted(covered) != sorted(NPA_CHECKPOINT_PAIRS):
        raise ValueError(f"checkpoint coverage mismatch: {sorted(covered)}")

    scores = {
        mode: {
            "linear": linear[mode],
            "termwise_squared": termwise[mode],
            "timestamp_sum_squared": np.square(response[mode]).mean(axis=1),
        }
        for mode in E20_MODES
    }
    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    observed = {
        metric: np.load(LDS_DIR / f"observed_{metric}.npy").astype(np.float64)[
            list(E20_QUERY_IDS)
        ]
        for metric in LDS_METRICS
    }
    result = {
        "query_ids": list(E20_QUERY_IDS),
        "checkpoint_pairs": list(NPA_CHECKPOINT_PAIRS),
        "timestamp_positions": list(E20_TIMESTAMP_POSITIONS),
        "timesteps": list(E20_TIMESTEPS),
        "modes": list(E20_MODES),
        "results": {},
        "paired_comparison": {},
    }
    print("ENDPOINT20 INVERSE-NOISE LOSS PAIRING (sign=-1)")
    for contraction in E20_CONTRACTIONS:
        result["results"][contraction] = {}
        result["paired_comparison"][contraction] = {}
        values = {}
        print(f"\n{contraction}")
        for mode in E20_MODES:
            method = e20_method(mode, contraction)
            method_scores = scores[mode][contraction]
            for position, query_id in enumerate(E20_QUERY_IDS):
                output = ATTR_DIR / method / f"q{query_id:02d}"
                output.mkdir(parents=True, exist_ok=True)
                np.save(output / "scores.npy", method_scores[position])
                with open(output / "info.json", "w") as handle:
                    json.dump(
                        {
                            "method": method,
                            "query_id": query_id,
                            "mode": mode,
                            "contraction": contraction,
                            "checkpoint_pairs": list(NPA_CHECKPOINT_PAIRS),
                            "timesteps": list(E20_TIMESTEPS),
                        },
                        handle,
                        indent=2,
                    )
            prediction = membership @ method_scores.T
            entry = {"method": method, "targets": {}}
            values[mode] = {}
            cells = []
            for metric in LDS_METRICS:
                per_query = np.asarray(
                    [
                        spearmanr(-prediction[:, q], observed[metric][q]).statistic
                        for q in range(len(E20_QUERY_IDS))
                    ],
                    dtype=np.float64,
                )
                values[mode][metric] = per_query
                entry["targets"][metric] = {
                    "mean": float(np.nanmean(per_query)),
                    "std": float(np.nanstd(per_query)),
                    "per_query": per_query.tolist(),
                }
                cells.append(f"{metric}={np.nanmean(per_query):+.4f}")
            result["results"][contraction][mode] = entry
            print(f"  {mode:25s} " + " | ".join(cells))

        print("  paired: per_timestamp_aligned - mean_noise_mean_loss")
        for metric in LDS_METRICS:
            difference = (
                values["per_timestamp_aligned"][metric]
                - values["mean_noise_mean_loss"][metric]
            )
            finite = difference[np.isfinite(difference)]
            try:
                pvalue = float(
                    wilcoxon(finite, alternative="two-sided").pvalue
                )
            except ValueError:
                pvalue = 1.0
            paired = {
                "mean_difference": float(np.nanmean(difference)),
                "wilcoxon_two_sided_p": pvalue,
                "per_timestamp_wins": int(np.count_nonzero(difference > 0)),
                "query_count": int(len(finite)),
                "difference_per_query": difference.tolist(),
            }
            result["paired_comparison"][contraction][metric] = paired
            print(
                f"    {metric:30s} delta={paired['mean_difference']:+.6f} "
                f"p={pvalue:.6g} wins={paired['per_timestamp_wins']}/10"
            )

    output = LDS_DIR / "tracin_das_endpoint20_inverse_noise_pairing_q00_q09.json"
    atomic_json(output, result)
    print(f"[saved] {output}", flush=True)


if __name__ == "__main__":
    main()
