"""Print prompted-only trajectory pairing against cached random controls."""

from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

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
METHODS = (
    "traj-A",
    "traj-I",
    "rand-A",
    "cyclic",
    "rand-P",
    "indep",
)
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
            CURRENT_ROOT / mode / "prompted" / f"shard_{shard:02d}_of_04"
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
            total[query_id] += np.load(
                ATTR_DIR / method / f"q{query_id:02d}" / "scores.npy"
            ).astype(np.float64) / len(components)
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
        "PROMPTED q00-q74: TRAJECTORY PAIRING VS RANDOM-NOISE CONTROLS",
        "=" * 118,
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
        lines.extend(
            [
                "",
                f"GROUP: {group}",
                "-" * 118,
                "target                           traj-A    traj-I    rand-A    "
                "cyclic    rand-P     indep",
            ]
        )
        for metric in LDS_METRICS:
            line = f"{metric:30s}" + "".join(
                f" {np.nanmean(values[label][metric]):+8.5f}" for label in METHODS
            )
            lines.append(line)

        lines.extend(
            [
                "",
                "PAIRED: traj-A minus control",
                "target/control                       delta     p-wilcoxon   wins/75",
            ]
        )
        for metric in LDS_METRICS:
            aligned = values["traj-A"][metric]
            for control in METHODS[1:]:
                delta = aligned - values[control][metric]
                finite = delta[np.isfinite(delta)]
                try:
                    pvalue = float(wilcoxon(finite, alternative="two-sided").pvalue)
                except ValueError:
                    pvalue = 1.0
                lines.append(
                    f"{metric}/{control:44s} "
                    f"{np.nanmean(delta):+9.6f} {pvalue:12.6g} "
                    f"{int(np.count_nonzero(delta > 0)):02d}/75"
                )

        del scores, values

    output = LDS_DIR / "trajectory_pairing_prompted_q00_q74_six_way.txt"
    output.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[saved] {output}")


if __name__ == "__main__":
    main()
