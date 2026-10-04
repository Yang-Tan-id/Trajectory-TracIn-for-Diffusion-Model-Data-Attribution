"""Render the controlled noise-pairing LDS ablation as a readable text report."""

import argparse
import itertools
import json
import os
import random
from pathlib import Path


DEFAULT_JSON = Path(
    "x3_lds_exp_50k/lds/"
    "tracin_das_noise_pairing_ablation_10ckpt_20t_mc10_q00_q09.json"
)
PAIRINGS = ("aligned", "cyclic", "random_permutation", "independent")
CONTROLS = PAIRINGS[1:]


def atomic_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        handle.write(value)
    os.replace(temporary, path)


def signed(value):
    return f"{float(value):+.6f}"


def exact_sign_flip_p(values):
    """Exact paired randomization p-values for H1: mean(values) > 0."""
    values = [float(value) for value in values]
    observed = sum(values) / len(values)
    null = [
        sum(sign * value for sign, value in zip(signs, values)) / len(values)
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    tolerance = 1e-15
    one_sided = sum(value >= observed - tolerance for value in null) / len(null)
    two_sided = sum(
        abs(value) >= abs(observed) - tolerance for value in null
    ) / len(null)
    return one_sided, two_sided


def paired_bootstrap_ci(values, seed, draws=100000):
    values = [float(value) for value in values]
    rng = random.Random(seed)
    means = sorted(
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(draws)
    )
    return means[int(0.025 * draws)], means[int(0.975 * draws) - 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with open(args.input) as handle:
        result = json.load(handle)
    output = args.output or args.input.with_suffix(".txt")

    query_ids = [int(value) for value in result["query_ids"]]
    aligned_results = result["results"]["aligned"]
    variants = tuple(aligned_results)
    contractions = tuple(next(iter(aligned_results.values())))
    groups = tuple(result["timestamp_groups"])
    metrics = tuple(
        next(iter(next(iter(aligned_results.values())).values()))["all"]["targets"]
    )

    lines = [
        "CONTROLLED TRACIN-DAS NOISE-PAIRING ABLATION",
        "=" * 96,
        f"queries             : {query_ids[0]:02d}-{query_ids[-1]:02d} ({len(query_ids)})",
        f"checkpoint pairs    : {result['checkpoint_pairs']}",
        f"timestamp groups    : {result['timestamp_groups']}",
        "reported LDS sign   : -1",
        "delta definition    : aligned LDS - control LDS (paired across queries)",
        "pairings            : aligned, cyclic, random_permutation, independent",
        "",
        "ALL-TIMESTAMP SUMMARY",
        "=" * 96,
    ]

    for variant in variants:
        for contraction in contractions:
            lines.extend(
                [
                    "",
                    f"VARIANT={variant}  CONTRACTION={contraction}",
                    "-" * 96,
                    (
                        f"{'target':30s} {'aligned':>19s} {'cyclic':>19s} "
                        f"{'random_perm':>19s} {'independent':>19s}"
                    ),
                    (
                        f"{'':30s} {'mean +/- std':>19s} {'mean / delta':>19s} "
                        f"{'mean / delta':>19s} {'mean / delta':>19s}"
                    ),
                ]
            )
            for metric in metrics:
                aligned = result["results"]["aligned"][variant][contraction]["all"][
                    "targets"
                ][metric]["negative"]
                cells = [f"{signed(aligned['mean'])} +/- {aligned['std']:.6f}"]
                for control in CONTROLS:
                    current = result["results"][control][variant][contraction]["all"][
                        "targets"
                    ][metric]["negative"]
                    delta = result["paired_improvements"][control][variant][contraction][
                        "all"
                    ][metric]["mean"]
                    cells.append(f"{signed(current['mean'])} / {signed(delta)}")
                lines.append(
                    f"{metric:30s} " + " ".join(f"{cell:>19s}" for cell in cells)
                )

    lines.extend(["", "", "TIMESTAMP-GROUP BREAKDOWN", "=" * 96])
    for variant in variants:
        for contraction in contractions:
            for metric in metrics:
                lines.extend(
                    [
                        "",
                        f"VARIANT={variant}  CONTRACTION={contraction}  TARGET={metric}",
                        "-" * 96,
                        (
                            f"{'group':8s} {'aligned':>10s} {'cyclic':>10s} "
                            f"{'random':>10s} {'indep':>10s} "
                            f"{'d-cyclic':>10s} {'d-random':>10s} {'d-indep':>10s}"
                        ),
                    ]
                )
                for group in groups:
                    means = {
                        pairing: result["results"][pairing][variant][contraction][group][
                            "targets"
                        ][metric]["negative"]["mean"]
                        for pairing in PAIRINGS
                    }
                    deltas = {
                        control: result["paired_improvements"][control][variant][
                            contraction
                        ][group][metric]["mean"]
                        for control in CONTROLS
                    }
                    lines.append(
                        f"{group:8s} {signed(means['aligned']):>10s} "
                        f"{signed(means['cyclic']):>10s} "
                        f"{signed(means['random_permutation']):>10s} "
                        f"{signed(means['independent']):>10s} "
                        f"{signed(deltas['cyclic']):>10s} "
                        f"{signed(deltas['random_permutation']):>10s} "
                        f"{signed(deltas['independent']):>10s}"
                    )

    lines.extend(
        [
            "",
            "",
            "PAIRED SIGNIFICANCE TESTS (ALL TIMESTAMPS, SIGN=-1)",
            "=" * 116,
            "H1: aligned LDS > control LDS",
            "p1 = exact one-sided paired sign-flip randomization p-value",
            "p2 = exact two-sided paired sign-flip randomization p-value",
            "CI = paired-query bootstrap 95% confidence interval for aligned - control",
            "Unadjusted exploratory p-values; apply a multiple-testing correction when making global claims.",
        ]
    )
    test_index = 0
    for variant in variants:
        for contraction in contractions:
            lines.extend(
                [
                    "",
                    f"VARIANT={variant}  CONTRACTION={contraction}",
                    "-" * 116,
                    (
                        f"{'target':30s} {'control':19s} {'delta':>10s} "
                        f"{'p1':>10s} {'p2':>10s} {'bootstrap 95% CI':>27s} {'wins':>7s}"
                    ),
                ]
            )
            for metric in metrics:
                for control in CONTROLS:
                    values = result["paired_improvements"][control][variant][
                        contraction
                    ]["all"][metric]["per_query"]
                    mean = sum(values) / len(values)
                    p1, p2 = exact_sign_flip_p(values)
                    low, high = paired_bootstrap_ci(
                        values, seed=20261004 + test_index
                    )
                    wins = sum(float(value) > 0.0 for value in values)
                    lines.append(
                        f"{metric:30s} {control:19s} {signed(mean):>10s} "
                        f"{p1:10.6f} {p2:10.6f} "
                        f"[{signed(low)}, {signed(high)}] {wins:>3d}/{len(values):<3d}"
                    )
                    test_index += 1

    lines.extend(["", "", "PER-QUERY RESULTS (ALL TIMESTAMPS, SIGN=-1)", "=" * 96])
    for variant in variants:
        for contraction in contractions:
            for metric in metrics:
                lines.extend(
                    [
                        "",
                        f"VARIANT={variant}  CONTRACTION={contraction}  TARGET={metric}",
                        "-" * 112,
                        (
                            f"{'query':>5s} {'aligned':>10s} {'cyclic':>10s} "
                            f"{'random':>10s} {'indep':>10s} "
                            f"{'d-cyclic':>10s} {'d-random':>10s} {'d-indep':>10s}"
                        ),
                    ]
                )
                values = {
                    pairing: result["results"][pairing][variant][contraction]["all"][
                        "targets"
                    ][metric]["negative"]["per_query"]
                    for pairing in PAIRINGS
                }
                deltas = {
                    control: result["paired_improvements"][control][variant][
                        contraction
                    ]["all"][metric]["per_query"]
                    for control in CONTROLS
                }
                for position, query_id in enumerate(query_ids):
                    lines.append(
                        f"q{query_id:02d}   {signed(values['aligned'][position]):>10s} "
                        f"{signed(values['cyclic'][position]):>10s} "
                        f"{signed(values['random_permutation'][position]):>10s} "
                        f"{signed(values['independent'][position]):>10s} "
                        f"{signed(deltas['cyclic'][position]):>10s} "
                        f"{signed(deltas['random_permutation'][position]):>10s} "
                        f"{signed(deltas['independent'][position]):>10s}"
                    )

    atomic_text(output, "\n".join(lines) + "\n")
    print(f"[saved] {output}", flush=True)


if __name__ == "__main__":
    main()
