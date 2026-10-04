"""Merge all-pairs shared-bank cross terms and compare against aligned pairing."""

import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr

from noise_pairing_ablation_config import *
from importlib import import_module


stats_report = import_module("238_print_tracin_das_noise_pairing_ablation_txt")


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
    return "__".join(("all_pairs", "raw", "timestamp_sum_squared", group))


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
            root = npa100_cross_shard_root(
                family, shard_index, args.timestamp_shard_count
            )
            with open(root / "done.json") as handle:
                info = json.load(handle)
            if tuple(info["query_ids"]) != family_ids:
                raise ValueError(f"query mismatch in {root}")
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

    baseline_path = (
        LDS_DIR
        / "tracin_das_noise_pairing_ablation_10ckpt_20t_mc10_q00_q99.json"
    )
    with open(baseline_path) as handle:
        baseline = json.load(handle)
    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    result = {
        "query_ids": list(query_ids),
        "definition": (
            "mean over all 10x10 shared-bank query/train direction pairs of the "
            "squared checkpoint-summed projected inner product"
        ),
        "checkpoint_pairs": list(NPA_CHECKPOINT_PAIRS),
        "timestamp_groups": {
            key: list(value) for key, value in NPA_TIMESTAMP_GROUPS.items()
        },
        "groups": {},
    }
    report_lines = [
        "SHARED-BANK ALL-PAIRS CROSS-TERM TRACIN-DAS",
        "=" * 112,
        "score = mean_(m,n),t [(sum_c <q_(c,t,m), h_(i,c,t,n)>)^2]",
        "all_pairs includes 10 diagonal and 90 off-diagonal direction pairs",
        "delta = aligned diagonal-only LDS - all_pairs LDS",
        "",
    ]
    test_index = 0
    for group in NPA_TIMESTAMP_GROUPS:
        scores = totals[group]
        method = npa100_method("all_pairs", group)
        for query_id in query_ids:
            output = ATTR_DIR / method / f"q{query_id:02d}"
            output.mkdir(parents=True, exist_ok=True)
            np.save(output / "scores.npy", scores[query_id])
        predicted = membership @ scores.T
        group_result = {"method": method, "targets": {}}
        if group == "all":
            report_lines.extend(
                [
                    "ALL TIMESTAMPS",
                    "-" * 112,
                    (
                        f"{'target':30s} {'aligned':>10s} {'all_pairs':>10s} "
                        f"{'delta':>10s} {'p1':>10s} {'p2':>10s} "
                        f"{'bootstrap 95% CI':>27s} {'wins':>8s}"
                    ),
                ]
            )
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
            all_pairs_values = -positive
            aligned_entry = baseline["results"]["aligned"]["raw"][
                "timestamp_sum_squared"
            ][group]["targets"][metric]["negative"]
            aligned_values = np.asarray(aligned_entry["per_query"], dtype=np.float64)
            delta = aligned_values - all_pairs_values
            p1, p2, test_kind = stats_report.paired_sign_flip_p(
                delta, seed=20261004 + test_index
            )
            low, high = stats_report.paired_bootstrap_ci(
                delta, seed=20261004 + test_index
            )
            entry = {
                "aligned": aligned_entry,
                "all_pairs": {
                    "mean": float(np.nanmean(all_pairs_values)),
                    "std": float(np.nanstd(all_pairs_values)),
                    "per_query": all_pairs_values.tolist(),
                },
                "aligned_minus_all_pairs": {
                    "mean": float(np.nanmean(delta)),
                    "std": float(np.nanstd(delta)),
                    "per_query": delta.tolist(),
                    "one_sided_p": p1,
                    "two_sided_p": p2,
                    "test": test_kind,
                    "bootstrap_95": [low, high],
                    "wins": int(np.count_nonzero(delta > 0)),
                },
            }
            group_result["targets"][metric] = entry
            if group == "all":
                report_lines.append(
                    f"{metric:30s} {aligned_entry['mean']:+10.6f} "
                    f"{entry['all_pairs']['mean']:+10.6f} "
                    f"{entry['aligned_minus_all_pairs']['mean']:+10.6f} "
                    f"{p1:10.6f} {p2:10.6f} "
                    f"[{low:+.6f}, {high:+.6f}] "
                    f"{entry['aligned_minus_all_pairs']['wins']:>3d}/100"
                )
            test_index += 1
        result["groups"][group] = group_result

    report_lines.extend(["", "TIMESTAMP GROUPS", "-" * 112])
    for metric in LDS_METRICS:
        report_lines.append(f"\n{metric}")
        report_lines.append(
            f"{'group':8s} {'aligned':>10s} {'all_pairs':>10s} {'delta':>10s}"
        )
        for group in NPA_TIMESTAMP_GROUPS:
            entry = result["groups"][group]["targets"][metric]
            report_lines.append(
                f"{group:8s} {entry['aligned']['mean']:+10.6f} "
                f"{entry['all_pairs']['mean']:+10.6f} "
                f"{entry['aligned_minus_all_pairs']['mean']:+10.6f}"
            )

    output = (
        LDS_DIR
        / "tracin_das_noise_pairing_all_pairs_cross_10ckpt_20t_mc10_q00_q99.json"
    )
    atomic_json(output, result)
    atomic_text(output.with_suffix(".txt"), "\n".join(report_lines) + "\n")
    print(f"[saved] {output}", flush=True)
    print(f"[saved] {output.with_suffix('.txt')}", flush=True)


if __name__ == "__main__":
    main()
