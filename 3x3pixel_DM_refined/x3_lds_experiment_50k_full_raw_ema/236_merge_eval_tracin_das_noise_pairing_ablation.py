"""Merge and evaluate the controlled TracIn-DAS noise-pairing ablation."""

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


def score_key(pairing, variant, contraction, group):
    return "__".join((pairing, variant, contraction, group))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp-shard-count", type=int, default=4)
    args = parser.parse_args()
    totals = {
        score_key(pairing, variant, contraction, group): np.zeros(
            (len(NPA_QUERY_IDS), N_TRAIN), dtype=np.float64
        )
        for pairing in NPA_PAIRINGS
        for variant in NPA_VARIANTS
        for contraction in NPA_CONTRACTIONS
        for group in NPA_TIMESTAMP_GROUPS
    }
    covered = []
    for shard_index in range(args.timestamp_shard_count):
        root = npa_shard_root(shard_index, args.timestamp_shard_count)
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
        "query_ids": list(NPA_QUERY_IDS),
        "checkpoint_pairs": list(NPA_CHECKPOINT_PAIRS),
        "timestamp_groups": {
            key: list(value) for key, value in NPA_TIMESTAMP_GROUPS.items()
        },
        "results": {},
        "paired_improvements": {},
    }
    per_query_results = {}
    for pairing in NPA_PAIRINGS:
        result["results"][pairing] = {}
        per_query_results[pairing] = {}
        for variant in NPA_VARIANTS:
            result["results"][pairing][variant] = {}
            per_query_results[pairing][variant] = {}
            for contraction in NPA_CONTRACTIONS:
                result["results"][pairing][variant][contraction] = {}
                per_query_results[pairing][variant][contraction] = {}
                for group in NPA_TIMESTAMP_GROUPS:
                    key = score_key(pairing, variant, contraction, group)
                    scores = totals[key]
                    method = npa_method(pairing, variant, contraction, group)
                    for query_position, query_id in enumerate(NPA_QUERY_IDS):
                        output = ATTR_DIR / method / f"q{query_id:02d}"
                        output.mkdir(parents=True, exist_ok=True)
                        np.save(output / "scores.npy", scores[query_position])
                    entry = {"method": method, "targets": {}}
                    predicted = membership @ scores.T
                    for metric in LDS_METRICS:
                        observed = np.load(
                            LDS_DIR / f"observed_{metric}.npy"
                        ).astype(np.float64)[list(NPA_QUERY_IDS)]
                        positive = np.asarray(
                            [
                                spearmanr(
                                    predicted[:, position], observed[position]
                                ).statistic
                                for position in range(len(NPA_QUERY_IDS))
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
                        per_query_results[pairing][variant][contraction].setdefault(
                            group, {}
                        )[metric] = signs
                    result["results"][pairing][variant][contraction][group] = entry

    for baseline in ("cyclic", "random_permutation", "independent"):
        result["paired_improvements"][baseline] = {}
        for variant in NPA_VARIANTS:
            result["paired_improvements"][baseline][variant] = {}
            for contraction in NPA_CONTRACTIONS:
                result["paired_improvements"][baseline][variant][contraction] = {}
                for group in NPA_TIMESTAMP_GROUPS:
                    comparisons = {}
                    for metric in LDS_METRICS:
                        aligned = np.asarray(
                            per_query_results["aligned"][variant][contraction][group][
                                metric
                            ]["negative"]["per_query"]
                        )
                        other = np.asarray(
                            per_query_results[baseline][variant][contraction][group][
                                metric
                            ]["negative"]["per_query"]
                        )
                        delta = aligned - other
                        comparisons[metric] = {
                            "mean": float(np.nanmean(delta)),
                            "std": float(np.nanstd(delta)),
                            "per_query": delta.tolist(),
                        }
                    result["paired_improvements"][baseline][variant][contraction][
                        group
                    ] = comparisons

    print("ALIGNED vs pairing controls (sign=-1, all timestamps)")
    for variant in NPA_VARIANTS:
        for contraction in NPA_CONTRACTIONS:
            print(f"\n{variant} / {contraction}")
            for metric in LDS_METRICS:
                aligned = result["results"]["aligned"][variant][contraction]["all"][
                    "targets"
                ][metric]["negative"]
                controls = []
                for pairing in ("cyclic", "random_permutation", "independent"):
                    value = result["results"][pairing][variant][contraction]["all"][
                        "targets"
                    ][metric]["negative"]
                    delta = result["paired_improvements"][pairing][variant][
                        contraction
                    ]["all"][metric]["mean"]
                    controls.append(
                        f"{pairing}={value['mean']:+.4f} (delta={delta:+.4f})"
                    )
                print(
                    f"{metric:30s} aligned={aligned['mean']:+.4f} | "
                    + " | ".join(controls)
                )

    output = LDS_DIR / "tracin_das_noise_pairing_ablation_10ckpt_20t_mc10_q00_q09.json"
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
