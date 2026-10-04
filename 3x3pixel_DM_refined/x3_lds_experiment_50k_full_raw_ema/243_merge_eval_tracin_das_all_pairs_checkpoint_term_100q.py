"""Merge/evaluate all-pairs, within-checkpoint direction-sum-square scores."""

import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr

from noise_pairing_ablation_config import *


CONTRACTION = "checkpoint_direction_sum_squared"


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary, path)


def atomic_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        handle.write(value)
    os.replace(temporary, path)


def score_key(group):
    return "__".join(("all_pairs", "raw", CONTRACTION, group))


def method_name(group):
    return (
        "tracin_das_noise_pairing_all_pairs_10ckpt_20t_mc10_"
        "adamw_full_next_delta_projected4096_"
        f"raw_{CONTRACTION}_lr_outside_{group}_q00_q99"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp-shard-count", type=int, default=4)
    args = parser.parse_args()
    query_ids = tuple(range(100))
    totals = {
        group: np.zeros((100, N_TRAIN), dtype=np.float64)
        for group in NPA_TIMESTAMP_GROUPS
    }
    for family in ("prompted", "unprompted"):
        family_ids = npa100_query_ids(family)
        family_totals = {
            group: np.zeros((len(family_ids), N_TRAIN), dtype=np.float64)
            for group in NPA_TIMESTAMP_GROUPS
        }
        covered = []
        for shard_index in range(args.timestamp_shard_count):
            root = npa100_cross_term_shard_root(
                family, shard_index, args.timestamp_shard_count
            )
            with open(root / "done.json") as handle:
                info = json.load(handle)
            covered.extend(int(value) for value in info["timestamp_indices"])
            with np.load(root / "partial_scores.npz") as partial:
                for group in NPA_TIMESTAMP_GROUPS:
                    family_totals[group] += partial[score_key(group)].astype(
                        np.float64
                    )
        if sorted(covered) != sorted(NPA_TIMESTAMP_INDICES):
            raise ValueError(f"timestamp coverage mismatch for {family}: {covered}")
        family_slice = np.asarray(family_ids, dtype=np.int64)
        for group in NPA_TIMESTAMP_GROUPS:
            totals[group][family_slice] = family_totals[group]

    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    result = {
        "query_ids": list(query_ids),
        "formula": (
            "mean_t sum_checkpoint lr_checkpoint * "
            "(mean_direction_pair projected_adamw_direction_without_lr_inner_product)^2"
        ),
        "checkpoint_pairs": list(NPA_CHECKPOINT_PAIRS),
        "timestamp_groups": {
            key: list(value) for key, value in NPA_TIMESTAMP_GROUPS.items()
        },
        "groups": {},
    }
    for group in NPA_TIMESTAMP_GROUPS:
        scores = totals[group]
        method = method_name(group)
        for query_id in query_ids:
            output = ATTR_DIR / method / f"q{query_id:02d}"
            output.mkdir(parents=True, exist_ok=True)
            np.save(output / "scores.npy", scores[query_id])
        predicted = membership @ scores.T
        entry = {"method": method, "targets": {}}
        for metric in LDS_METRICS:
            observed = np.load(LDS_DIR / f"observed_{metric}.npy").astype(
                np.float64
            )[list(query_ids)]
            positive = np.asarray(
                [
                    spearmanr(predicted[:, position], observed[position]).statistic
                    for position in range(len(query_ids))
                ],
                dtype=np.float64,
            )
            entry["targets"][metric] = {
                "negative": {
                    "mean": float(np.nanmean(-positive)),
                    "std": float(np.nanstd(-positive)),
                    "per_query": (-positive).tolist(),
                },
                "positive": {
                    "mean": float(np.nanmean(positive)),
                    "std": float(np.nanstd(positive)),
                    "per_query": positive.tolist(),
                },
            }
        result["groups"][group] = entry

    lines = [
        "ALL-PAIRS WITHIN-CHECKPOINT DIRECTION-SUM-SQUARED TRACIN-DAS",
        "=" * 92,
        "score = mean_t sum_c lr_c [mean_(m,n) <q_(c,t,m), h_(i,c,t,n)/lr_c>]^2",
        "reported LDS sign = -1",
        "",
        "ALL TIMESTAMPS",
        "-" * 92,
        f"{'target':30s} {'mean':>12s} {'std(query)':>14s}",
    ]
    for metric in LDS_METRICS:
        value = result["groups"]["all"]["targets"][metric]["negative"]
        lines.append(f"{metric:30s} {value['mean']:+12.6f} {value['std']:14.6f}")
    lines.extend(["", "TIMESTAMP GROUPS", "-" * 92])
    for metric in LDS_METRICS:
        lines.append(f"\n{metric}")
        for group in NPA_TIMESTAMP_GROUPS:
            value = result["groups"][group]["targets"][metric]["negative"]
            lines.append(
                f"  {group:8s} mean={value['mean']:+.6f} std={value['std']:.6f}"
            )

    output = (
        LDS_DIR
        / "tracin_das_all_pairs_checkpoint_direction_sum_squared_lr_outside_10ckpt_20t_mc10_q00_q99.json"
    )
    atomic_json(output, result)
    atomic_text(output.with_suffix(".txt"), "\n".join(lines) + "\n")
    print(f"[saved] {output}", flush=True)
    print(f"[saved] {output.with_suffix('.txt')}", flush=True)


if __name__ == "__main__":
    main()
