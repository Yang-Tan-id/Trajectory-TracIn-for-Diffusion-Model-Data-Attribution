"""Compare controlled TracIn-DAS noise pairing at small versus large t."""

import argparse
import json
from pathlib import Path

import numpy as np

from exp_config import DAS_TIMESTEPS


DEFAULT_INPUT = Path(
    "x3_lds_exp_50k/lds/"
    "tracin_das_noise_pairing_ablation_10ckpt_20t_mc10_q00_q99.json"
)
CONTROLS = ("cyclic", "random_permutation", "independent")


def sign_flip_p(values, seed, draws=500_000):
    """Paired one/two-sided randomization p-values for a positive mean."""
    values = np.asarray(values, dtype=np.float64)
    observed = float(np.nanmean(values))
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    one_extreme = 0
    two_extreme = 0
    completed = 0
    while completed < draws:
        count = min(10_000, draws - completed)
        signs = 2.0 * rng.integers(0, 2, size=(count, len(values))) - 1.0
        null = (signs @ values) / len(values)
        one_extreme += int(np.count_nonzero(null >= observed))
        two_extreme += int(np.count_nonzero(np.abs(null) >= abs(observed)))
        completed += count
    return (
        (one_extreme + 1.0) / (draws + 1.0),
        (two_extreme + 1.0) / (draws + 1.0),
    )


def bootstrap_ci(values, seed, draws=100_000):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    return tuple(np.quantile(values[indices].mean(axis=1), (0.025, 0.975)))


def per_query(result, pairing, group, metric):
    return np.asarray(
        result["results"][pairing]["raw"]["timestamp_sum_squared"][group]
        ["targets"][metric]["negative"]["per_query"],
        dtype=np.float64,
    )


def fmt(value):
    return f"{float(value):+.6f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with open(args.input) as handle:
        result = json.load(handle)

    groups = result["timestamp_groups"]
    small_group, large_group = "q1", "q4"
    small_steps = [int(DAS_TIMESTEPS[index]) for index in groups[small_group]]
    large_steps = [int(DAS_TIMESTEPS[index]) for index in groups[large_group]]
    metrics = tuple(
        result["results"]["aligned"]["raw"]["timestamp_sum_squared"]["all"]
        ["targets"]
    )

    lines = [
        "TRACIN-DAS NOISE ALIGNMENT: SMALL-t VS LARGE-t",
        "=" * 126,
        f"small-t indices={groups[small_group]} diffusion_t={small_steps}",
        f"large-t indices={groups[large_group]} diffusion_t={large_steps}",
        "reported LDS sign=-1; improvement=aligned-control",
        "interaction=(large-t improvement)-(small-t improvement)",
        "Positive interaction means alignment helps more at large t.",
        "",
    ]

    test_index = 0
    for metric in metrics:
        lines.extend(
            [
                metric,
                "-" * 126,
                (
                    f"{'control':19s} {'small A/C':>21s} {'small delta':>12s} "
                    f"{'large A/C':>21s} {'large delta':>12s} "
                    f"{'interaction':>12s} {'p1':>9s} {'p2':>9s} {'95% CI':>25s}"
                ),
            ]
        )
        for control in CONTROLS:
            small_aligned = per_query(result, "aligned", small_group, metric)
            small_control = per_query(result, control, small_group, metric)
            large_aligned = per_query(result, "aligned", large_group, metric)
            large_control = per_query(result, control, large_group, metric)
            small_delta = small_aligned - small_control
            large_delta = large_aligned - large_control
            interaction = large_delta - small_delta
            p1, p2 = sign_flip_p(interaction, 20261004 + test_index)
            low, high = bootstrap_ci(interaction, 20262004 + test_index)
            lines.append(
                f"{control:19s} "
                f"{fmt(np.nanmean(small_aligned))}/{fmt(np.nanmean(small_control)):>9s} "
                f"{fmt(np.nanmean(small_delta)):>12s} "
                f"{fmt(np.nanmean(large_aligned))}/{fmt(np.nanmean(large_control)):>9s} "
                f"{fmt(np.nanmean(large_delta)):>12s} "
                f"{fmt(np.nanmean(interaction)):>12s} {p1:9.6f} {p2:9.6f} "
                f"[{fmt(low)}, {fmt(high)}]"
            )
            test_index += 1
        lines.append("")

    output = args.output or args.input.with_name(
        "tracin_das_noise_pairing_small_vs_large_t_q00_q99.txt"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[saved] {output}")


if __name__ == "__main__":
    main()
