"""Merge q00-q99 trajectory pairing and run paired significance tests by t range."""

import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr, wilcoxon

from exp_config import ATTR_DIR, LDS_DIR, LDS_METRICS, MASK_DIR, N_TRAIN
from noise_pairing_ablation_config import (
    NPA_TIMESTAMP_GROUPS,
    npa100_method,
    npa100_query_ids,
)


MODES = ("trajectory_aligned", "trajectory_independent")
ROOT_MODES = {
    "trajectory_aligned": "aligned",
    "trajectory_independent": "independent",
}
CONTROLS = ("random_aligned", "cyclic", "random_permutation", "independent")
QUERY_IDS = tuple(range(100))
QUARTERS = ("q1", "q2", "q3", "q4")
GROUP_COMPONENTS = {
    "q1": ("q1",),
    "q2": ("q2",),
    "q3": ("q3",),
    "q4": ("q4",),
    "q1_q2": ("q1", "q2"),
    "q1_q3": ("q1", "q2", "q3"),
    "all": QUARTERS,
}
PRINT_GROUPS = ("all", "q1_q3", "q1_q2", "q1", "q4")
ROOT = ATTR_DIR / "_tracin_das_trajectory_pairing_100q_shards"
BOOTSTRAP_DRAWS = 200_000
SIGN_DRAWS = 200_000
RNG_SEED = 20261005


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary, path)


def trajectory_method(mode, group):
    count = 5 * len(GROUP_COMPONENTS[group])
    return (
        "tracin_das_pure_trajectory_direction_10ckpt_"
        f"{count}t_mc1_adamw_full_{mode}_next_delta_projected4096_"
        f"raw_timestamp_sum_squared_{group}"
    )


def target_lds(scores, membership, observed):
    predicted = membership @ scores.T
    return np.asarray(
        [
            spearmanr(-predicted[:, q], observed[q]).statistic
            for q in range(len(QUERY_IDS))
        ],
        dtype=np.float64,
    )


def paired_test(delta, rng):
    delta = np.asarray(delta, dtype=np.float64)
    delta = delta[np.isfinite(delta)]
    n = len(delta)
    chunk = 5_000
    boot = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for start in range(0, BOOTSTRAP_DRAWS, chunk):
        end = min(start + chunk, BOOTSTRAP_DRAWS)
        indices = rng.integers(0, n, size=(end - start, n))
        boot[start:end] = delta[indices].mean(axis=1)
    observed = abs(float(delta.mean()))
    extreme = 0
    for start in range(0, SIGN_DRAWS, chunk):
        end = min(start + chunk, SIGN_DRAWS)
        signs = rng.integers(0, 2, size=(end - start, n), dtype=np.int8) * 2 - 1
        means = (signs * delta).mean(axis=1)
        extreme += int(np.count_nonzero(np.abs(means) >= observed))
    try:
        p_wilcoxon = float(wilcoxon(delta, alternative="two-sided").pvalue)
    except ValueError:
        p_wilcoxon = 1.0
    return {
        "mean_delta": float(delta.mean()),
        "std_delta": float(delta.std()),
        "bootstrap_95_ci": [float(x) for x in np.quantile(boot, (0.025, 0.975))],
        "wilcoxon_p_two_sided": p_wilcoxon,
        "signflip_p_two_sided": float((extreme + 1) / (SIGN_DRAWS + 1)),
        "wins": int(np.count_nonzero(delta > 0)),
        "ties": int(np.count_nonzero(delta == 0)),
        "n": n,
        "per_query_delta": delta.tolist(),
    }


def load_trajectory_scores(mode, shard_count):
    result = {
        group: np.zeros((len(QUERY_IDS), N_TRAIN), dtype=np.float64)
        for group in GROUP_COMPONENTS
    }
    for family in ("prompted", "unprompted"):
        family_ids = npa100_query_ids(family)
        family_scores = {
            group: np.zeros((len(family_ids), N_TRAIN), dtype=np.float64)
            for group in GROUP_COMPONENTS
        }
        covered = []
        for shard in range(shard_count):
            root = (
                ROOT / ROOT_MODES[mode] / family
                / f"shard_{shard:02d}_of_{shard_count:02d}"
            )
            with open(root / "done.json") as handle:
                info = json.load(handle)
            if tuple(info["query_ids"]) != family_ids:
                raise ValueError(f"query mismatch in {root}")
            covered.extend(int(value) for value in info["timestamp_indices"])
            with np.load(root / "partial_scores.npz") as partial:
                for group in GROUP_COMPONENTS:
                    family_scores[group] += partial[
                        f"timestamp_sum_squared__{group}"
                    ].astype(np.float64)
        expected = tuple(
            value for quarter in QUARTERS for value in NPA_TIMESTAMP_GROUPS[quarter]
        )
        if sorted(covered) != sorted(expected):
            raise ValueError(f"timestamp mismatch in {family}/{mode}: {covered}")
        for group in GROUP_COMPONENTS:
            result[group][np.asarray(family_ids)] = family_scores[group]
    return result


def load_control_scores(pairing):
    quarter_scores = {
        quarter: np.zeros((len(QUERY_IDS), N_TRAIN), dtype=np.float64)
        for quarter in QUARTERS
    }
    for quarter in QUARTERS:
        method = npa100_method(pairing, quarter)
        for query_id in QUERY_IDS:
            quarter_scores[quarter][query_id] = np.load(
                ATTR_DIR / method / f"q{query_id:02d}" / "scores.npy"
            ).astype(np.float64)
    return {
        group: np.mean(
            np.stack([quarter_scores[value] for value in components], axis=0),
            axis=0,
        )
        for group, components in GROUP_COMPONENTS.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp-shard-count", type=int, default=4)
    args = parser.parse_args()
    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)

    score_by_method = {
        mode: load_trajectory_scores(mode, args.timestamp_shard_count)
        for mode in MODES
    }
    control_map = {
        "random_aligned": "aligned",
        "cyclic": "cyclic",
        "random_permutation": "random_permutation",
        "independent": "independent",
    }
    for label, pairing in control_map.items():
        score_by_method[label] = load_control_scores(pairing)

    for mode in MODES:
        for group, scores in score_by_method[mode].items():
            method = trajectory_method(mode, group)
            for query_id in QUERY_IDS:
                output = ATTR_DIR / method / f"q{query_id:02d}"
                output.mkdir(parents=True, exist_ok=True)
                np.save(output / "scores.npy", scores[query_id])

    rng = np.random.default_rng(RNG_SEED)
    result = {
        "query_ids": list(QUERY_IDS),
        "timestamp_groups": {
            group: [
                value
                for component in components
                for value in NPA_TIMESTAMP_GROUPS[component]
            ]
            for group, components in GROUP_COMPONENTS.items()
        },
        "methods": {},
        "paired_comparisons_vs_trajectory_aligned": {},
        "statistics": {
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "signflip_draws": SIGN_DRAWS,
            "rng_seed": RNG_SEED,
        },
    }
    lds = {}
    for label, grouped_scores in score_by_method.items():
        result["methods"][label] = {}
        lds[label] = {}
        for group, scores in grouped_scores.items():
            result["methods"][label][group] = {}
            lds[label][group] = {}
            for metric in LDS_METRICS:
                observed = np.load(LDS_DIR / f"observed_{metric}.npy").astype(
                    np.float64
                )[list(QUERY_IDS)]
                values = target_lds(scores, membership, observed)
                lds[label][group][metric] = values
                result["methods"][label][group][metric] = {
                    "mean": float(np.nanmean(values)),
                    "std": float(np.nanstd(values)),
                    "per_query": values.tolist(),
                }

    lines = []

    def emit(text=""):
        print(text)
        lines.append(text)

    emit("TRAJECTORY-ALIGNED 100-QUERY PAIRED TEST")
    emit("Groups: all=100%, q1_q3=75%, q1_q2=50%, q1=small-t 25%, q4=large-t 25%")
    labels = ("trajectory_aligned", "trajectory_independent", *CONTROLS)
    for group in PRINT_GROUPS:
        emit(f"\n[{group}]")
        emit("target                           traj-A    traj-I   rand-A   cyclic   rand-P    indep")
        for metric in LDS_METRICS:
            means = [result["methods"][label][group][metric]["mean"] for label in labels]
            emit(f"{metric:30s}" + "".join(f" {value:+.5f}" for value in means))

    for group in PRINT_GROUPS:
        result["paired_comparisons_vs_trajectory_aligned"][group] = {}
        for control in ("trajectory_independent", *CONTROLS):
            result["paired_comparisons_vs_trajectory_aligned"][group][control] = {}
            emit(f"\n[{group}] traj-A minus {control}")
            emit("target                           delta       bootstrap 95% CI       p-sign    p-wilcox wins")
            for metric in LDS_METRICS:
                test = paired_test(
                    lds["trajectory_aligned"][group][metric]
                    - lds[control][group][metric],
                    rng,
                )
                result["paired_comparisons_vs_trajectory_aligned"][group][control][metric] = test
                lo, hi = test["bootstrap_95_ci"]
                emit(
                    f"{metric:30s} {test['mean_delta']:+.6f} "
                    f"[{lo:+.6f},{hi:+.6f}] {test['signflip_p_two_sided']:.6g} "
                    f"{test['wilcoxon_p_two_sided']:.6g} {test['wins']:3d}/{test['n']}"
                )

    output = LDS_DIR / "tracin_das_trajectory_pairing_100q_significance.json"
    atomic_json(output, result)
    text_output = output.with_suffix(".txt")
    with open(text_output, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"\n[saved] {output}")
    print(f"[saved] {text_output}")


if __name__ == "__main__":
    main()
