"""Merge and evaluate the q00-q99 controlled noise-pairing ablation."""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
from scipy.stats import spearmanr

from noise_pairing_ablation_config import *


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary, path)


def score_key(pairing, group):
    return "__".join((pairing, "raw", "timestamp_sum_squared", group))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp-shard-count", type=int, default=4)
    args = parser.parse_args()
    query_ids = tuple(range(100))
    totals = {
        score_key(pairing, group): np.zeros(
            (len(query_ids), N_TRAIN), dtype=np.float64
        )
        for pairing in NPA_PAIRINGS
        for group in NPA_TIMESTAMP_GROUPS
    }
    covered_by_family = {}
    for family in ("prompted", "unprompted"):
        family_ids = npa100_query_ids(family)
        family_totals = {
            key: np.zeros((len(family_ids), N_TRAIN), dtype=np.float64)
            for key in totals
        }
        covered = []
        for shard_index in range(args.timestamp_shard_count):
            root = npa100_shard_root(
                family, shard_index, args.timestamp_shard_count
            )
            with open(root / "done.json") as handle:
                info = json.load(handle)
            if tuple(info["query_ids"]) != family_ids:
                raise ValueError(f"query mismatch in {root}")
            covered.extend(int(value) for value in info["timestamp_indices"])
            with np.load(root / "partial_scores.npz") as partial:
                for key in family_totals:
                    family_totals[key] += partial[key].astype(np.float64)
        if sorted(covered) != sorted(NPA_TIMESTAMP_INDICES):
            raise ValueError(f"timestamp coverage mismatch for {family}: {covered}")
        covered_by_family[family] = sorted(covered)
        totals_slice = np.asarray(family_ids, dtype=np.int64)
        for key in totals:
            totals[key][totals_slice] = family_totals[key]

    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    result = {
        "query_ids": list(query_ids),
        "families": {"prompted": list(range(75)), "unprompted": list(range(75, 100))},
        "checkpoint_pairs": list(NPA_CHECKPOINT_PAIRS),
        "timestamp_groups": {
            key: list(value) for key, value in NPA_TIMESTAMP_GROUPS.items()
        },
        "results": {},
        "paired_improvements": {},
    }
    per_query = {}
    for pairing in NPA_PAIRINGS:
        result["results"][pairing] = {"raw": {"timestamp_sum_squared": {}}}
        per_query[pairing] = {}
        for group in NPA_TIMESTAMP_GROUPS:
            scores = totals[score_key(pairing, group)]
            method = npa100_method(pairing, group)
            for query_id in query_ids:
                output = ATTR_DIR / method / f"q{query_id:02d}"
                output.mkdir(parents=True, exist_ok=True)
                np.save(output / "scores.npy", scores[query_id])
            predicted = membership @ scores.T
            entry = {"method": method, "targets": {}}
            per_query[pairing][group] = {}
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
                signs = {
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
                entry["targets"][metric] = signs
                per_query[pairing][group][metric] = signs
            result["results"][pairing]["raw"]["timestamp_sum_squared"][
                group
            ] = entry

    for control in ("cyclic", "random_permutation", "independent"):
        result["paired_improvements"][control] = {
            "raw": {"timestamp_sum_squared": {}}
        }
        for group in NPA_TIMESTAMP_GROUPS:
            comparisons = {}
            for metric in LDS_METRICS:
                aligned = np.asarray(
                    per_query["aligned"][group][metric]["negative"]["per_query"]
                )
                other = np.asarray(
                    per_query[control][group][metric]["negative"]["per_query"]
                )
                delta = aligned - other
                comparisons[metric] = {
                    "mean": float(np.nanmean(delta)),
                    "std": float(np.nanstd(delta)),
                    "per_query": delta.tolist(),
                }
            result["paired_improvements"][control]["raw"][
                "timestamp_sum_squared"
            ][group] = comparisons

    output = (
        LDS_DIR
        / "tracin_das_noise_pairing_ablation_10ckpt_20t_mc10_q00_q99.json"
    )
    atomic_json(output, result)
    print(f"[saved] {output}", flush=True)
    subprocess.run(
        [
            sys.executable,
            "238_print_tracin_das_noise_pairing_ablation_txt.py",
            "--input",
            str(output),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
