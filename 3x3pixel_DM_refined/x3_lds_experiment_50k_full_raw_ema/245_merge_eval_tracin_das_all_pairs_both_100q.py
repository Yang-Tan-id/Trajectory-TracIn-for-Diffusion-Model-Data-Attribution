"""Merge/evaluate both all-pairs contractions computed in one GPU pass."""

import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr

from noise_pairing_ablation_config import *


CONTRACTIONS = (
    "timestamp_sum_squared",
    "checkpoint_direction_sum_squared",
)


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


def score_key(contraction, group):
    return "__".join(("all_pairs", "raw", contraction, group))


def method_name(contraction, group):
    suffix = "_lr_outside" if contraction == "checkpoint_direction_sum_squared" else ""
    return (
        "tracin_das_noise_pairing_all_pairs_10ckpt_20t_mc10_"
        "adamw_full_next_delta_projected4096_"
        f"raw_{contraction}{suffix}_{group}_q00_q99"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp-shard-count", type=int, default=4)
    args = parser.parse_args()
    query_ids = tuple(range(100))
    totals = {
        (contraction, group): np.zeros((100, N_TRAIN), dtype=np.float64)
        for contraction in CONTRACTIONS
        for group in NPA_TIMESTAMP_GROUPS
    }
    for family in ("prompted", "unprompted"):
        family_ids = npa100_query_ids(family)
        family_totals = {
            key: np.zeros((len(family_ids), N_TRAIN), dtype=np.float64)
            for key in totals
        }
        covered = []
        for shard_index in range(args.timestamp_shard_count):
            root = npa100_cross_both_shard_root(
                family, shard_index, args.timestamp_shard_count
            )
            with open(root / "done.json") as handle:
                info = json.load(handle)
            covered.extend(int(value) for value in info["timestamp_indices"])
            with np.load(root / "partial_scores.npz") as partial:
                for contraction, group in family_totals:
                    family_totals[(contraction, group)] += partial[
                        score_key(contraction, group)
                    ].astype(np.float64)
        if sorted(covered) != sorted(NPA_TIMESTAMP_INDICES):
            raise ValueError(f"timestamp coverage mismatch for {family}: {covered}")
        family_slice = np.asarray(family_ids, dtype=np.int64)
        for key in totals:
            totals[key][family_slice] = family_totals[key]

    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    result = {
        "query_ids": list(query_ids),
        "checkpoint_pairs": list(NPA_CHECKPOINT_PAIRS),
        "timestamp_groups": {
            key: list(value) for key, value in NPA_TIMESTAMP_GROUPS.items()
        },
        "contractions": {},
    }
    for contraction in CONTRACTIONS:
        result["contractions"][contraction] = {}
        for group in NPA_TIMESTAMP_GROUPS:
            scores = totals[(contraction, group)]
            method = method_name(contraction, group)
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
            result["contractions"][contraction][group] = entry

    lines = [
        "SHARED-BANK ALL-PAIRS: TWO CONTRACTIONS FROM ONE GPU PASS",
        "=" * 108,
        "timestamp_sum_squared: mean_(m,n),t [sum_c d_(c,t,m,n)]^2",
        "checkpoint_direction_sum_squared: mean_t sum_c lr_c [mean_(m,n) d_(c,t,m,n)/lr_c]^2",
        "reported LDS sign = -1",
        "",
        f"{'target':30s} {'checkpoint-summed':>20s} {'within-checkpoint':>20s}",
        "-" * 108,
    ]
    for metric in LDS_METRICS:
        first = result["contractions"]["timestamp_sum_squared"]["all"]["targets"][
            metric
        ]["negative"]
        second = result["contractions"]["checkpoint_direction_sum_squared"]["all"][
            "targets"
        ][metric]["negative"]
        lines.append(
            f"{metric:30s} {first['mean']:+.6f} +/- {first['std']:.6f} "
            f"{second['mean']:+.6f} +/- {second['std']:.6f}"
        )
    lines.extend(["", "TIMESTAMP GROUPS", "-" * 108])
    for metric in LDS_METRICS:
        lines.append(f"\n{metric}")
        lines.append(f"{'group':8s} {'checkpoint-summed':>20s} {'within-checkpoint':>20s}")
        for group in NPA_TIMESTAMP_GROUPS:
            first = result["contractions"]["timestamp_sum_squared"][group][
                "targets"
            ][metric]["negative"]["mean"]
            second = result["contractions"]["checkpoint_direction_sum_squared"][
                group
            ]["targets"][metric]["negative"]["mean"]
            lines.append(f"{group:8s} {first:+20.6f} {second:+20.6f}")

    output = (
        LDS_DIR
        / "tracin_das_all_pairs_both_contractions_10ckpt_20t_mc10_q00_q99.json"
    )
    atomic_json(output, result)
    atomic_text(output.with_suffix(".txt"), "\n".join(lines) + "\n")
    print(f"[saved] {output}", flush=True)
    print(f"[saved] {output.with_suffix('.txt')}", flush=True)


if __name__ == "__main__":
    main()
