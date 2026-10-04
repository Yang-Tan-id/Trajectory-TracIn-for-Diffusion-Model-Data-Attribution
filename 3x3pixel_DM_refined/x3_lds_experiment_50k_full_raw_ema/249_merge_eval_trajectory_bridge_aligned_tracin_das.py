"""Merge and evaluate query-dependent trajectory-bridge TracIn-DAS."""

import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr

from trajectory_bridge_alignment_config import *


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary, path)


def score_key(contraction, group):
    return "__".join((contraction, group))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp-shard-count", type=int, default=4)
    args = parser.parse_args()
    totals = {
        score_key(contraction, group): np.zeros(
            (len(TBA_QUERY_IDS), N_TRAIN), dtype=np.float64
        )
        for contraction in TBA_CONTRACTIONS
        for group in NPA_TIMESTAMP_GROUPS
    }
    covered = []
    for shard_index in range(args.timestamp_shard_count):
        root = tba_shard_root(shard_index, args.timestamp_shard_count)
        with open(root / "done.json") as handle:
            info = json.load(handle)
        covered.extend(int(value) for value in info["timestamp_indices"])
        with np.load(root / "partial_scores.npz") as partial:
            for key in totals:
                totals[key] += partial[key].astype(np.float64)
    if sorted(covered) != sorted(NPA_TIMESTAMP_INDICES):
        raise ValueError(f"timestamp coverage mismatch: {sorted(covered)}")

    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    result = {
        "query_ids": list(TBA_QUERY_IDS),
        "checkpoint_pairs": list(NPA_CHECKPOINT_PAIRS),
        "timestamp_groups": {
            key: list(value) for key, value in NPA_TIMESTAMP_GROUPS.items()
        },
        "noise_path": "random-to-reference-trajectory bridge",
        "results": {},
    }
    print("TRAJECTORY-BRIDGE ALIGNED TRACIN-DAS (sign=-1)")
    for contraction in TBA_CONTRACTIONS:
        result["results"][contraction] = {}
        print(f"\n{contraction}")
        for group in NPA_TIMESTAMP_GROUPS:
            scores = totals[score_key(contraction, group)]
            method = tba_method(contraction, group)
            for position, query_id in enumerate(TBA_QUERY_IDS):
                output = ATTR_DIR / method / f"q{query_id:02d}"
                output.mkdir(parents=True, exist_ok=True)
                np.save(output / "scores.npy", scores[position])
            predicted = membership @ scores.T
            entry = {"method": method, "targets": {}}
            cells = []
            for metric in LDS_METRICS:
                observed = np.load(LDS_DIR / f"observed_{metric}.npy").astype(
                    np.float64
                )[list(TBA_QUERY_IDS)]
                positive = np.asarray(
                    [
                        spearmanr(predicted[:, position], observed[position]).statistic
                        for position in range(len(TBA_QUERY_IDS))
                    ]
                )
                negative = -positive
                entry["targets"][metric] = {
                    "negative": {
                        "mean": float(np.nanmean(negative)),
                        "std": float(np.nanstd(negative)),
                        "per_query": negative.tolist(),
                    },
                    "positive": {
                        "mean": float(np.nanmean(positive)),
                        "std": float(np.nanstd(positive)),
                        "per_query": positive.tolist(),
                    },
                }
                cells.append(f"{metric}={np.nanmean(negative):+.4f}")
            result["results"][contraction][group] = entry
            print(f"  {group:4s} " + " | ".join(cells))

    output = LDS_DIR / "tracin_das_trajectory_bridge_aligned_10ckpt_20t_mc10_q00_q09.json"
    atomic_json(output, result)
    print(f"[saved] {output}", flush=True)


if __name__ == "__main__":
    main()
