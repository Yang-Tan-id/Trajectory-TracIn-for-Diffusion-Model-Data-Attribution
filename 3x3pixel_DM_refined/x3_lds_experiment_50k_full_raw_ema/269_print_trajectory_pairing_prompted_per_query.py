"""Print per-query LDS for all six prompted trajectory-pairing methods."""

from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from exp_config import ATTR_DIR, LDS_DIR, LDS_METRICS, MASK_DIR, N_TRAIN
from noise_pairing_ablation_config import npa100_method


QUERY_IDS = tuple(range(75))
QUARTERS = ("q1", "q2", "q3", "q4")
GROUPS = {
    "all": QUARTERS,
    "q1_q3": ("q1", "q2", "q3"),
    "q1_q2": ("q1", "q2"),
    "q1": ("q1",),
    "q4": ("q4",),
}
METHODS = ("traj-A", "traj-I", "rand-A", "cyclic", "rand-P", "indep")
CONTROL_PAIRINGS = {
    "rand-A": "aligned",
    "cyclic": "cyclic",
    "rand-P": "random_permutation",
    "indep": "independent",
}
CURRENT_ROOT = ATTR_DIR / "_tracin_das_trajectory_pairing_100q_shards"


def current_scores(mode, group):
    total = np.zeros((len(QUERY_IDS), N_TRAIN), dtype=np.float64)
    for shard in range(4):
        path = (
            CURRENT_ROOT
            / mode
            / "prompted"
            / f"shard_{shard:02d}_of_04"
            / "partial_scores.npz"
        )
        with np.load(path) as partial:
            total += partial[f"timestamp_sum_squared__{group}"].astype(np.float64)
    return total


def control_scores(pairing, components):
    total = np.zeros((len(QUERY_IDS), N_TRAIN), dtype=np.float64)
    for component in components:
        method = npa100_method(pairing, component)
        for query_id in QUERY_IDS:
            total[query_id] += (
                np.load(ATTR_DIR / method / f"q{query_id:02d}" / "scores.npy")
                .astype(np.float64)
                / len(components)
            )
    return total


def evaluate(scores, membership, observed):
    prediction = membership @ scores.T
    return np.asarray(
        [
            spearmanr(-prediction[:, position], observed[position]).statistic
            for position in range(len(QUERY_IDS))
        ],
        dtype=np.float64,
    )


def main():
    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    observed = {
        metric: np.load(LDS_DIR / f"observed_{metric}.npy").astype(np.float64)[
            list(QUERY_IDS)
        ]
        for metric in LDS_METRICS
    }
    lines = [
        "PROMPTED q00-q74: SIX TRAJECTORY-PAIRING METHODS PER QUERY",
        "=" * 126,
        "traj-A=trajectory-aligned MC1 | traj-I=trajectory/independent MC1 | "
        "rand-A=random aligned MC10",
        "cyclic=cyclic MC10 | rand-P=random permutation MC10 | "
        "indep=random independent MC10",
        "Contraction=timestamp_sum_squared | full AdamW | projected4096 | sign=-1",
    ]

    for group, components in GROUPS.items():
        scores = {
            "traj-A": current_scores("aligned", group),
            "traj-I": current_scores("independent", group),
        }
        for label, pairing in CONTROL_PAIRINGS.items():
            scores[label] = control_scores(pairing, components)

        values = {
            label: {
                metric: evaluate(method_scores, membership, observed[metric])
                for metric in LDS_METRICS
            }
            for label, method_scores in scores.items()
        }

        for metric in LDS_METRICS:
            lines.extend(
                [
                    "",
                    f"GROUP={group}  TARGET={metric}",
                    "-" * 126,
                    "query     traj-A     traj-I     rand-A     cyclic     rand-P      indep",
                ]
            )
            for position, query_id in enumerate(QUERY_IDS):
                lines.append(
                    f"q{query_id:02d}  "
                    + " ".join(
                        f"{values[label][metric][position]:+10.6f}"
                        for label in METHODS
                    )
                )
            lines.append(
                "mean "
                + " ".join(
                    f"{np.nanmean(values[label][metric]):+10.6f}"
                    for label in METHODS
                )
            )

        del scores, values

    output = LDS_DIR / "trajectory_pairing_prompted_q00_q74_six_way_per_query.txt"
    output.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[saved] {output}")


if __name__ == "__main__":
    main()
