#!/usr/bin/env python3
"""Evaluate individual and combined fixed-reference timestamp-shared probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
REFINE_ROOT = SHAPES_ROOT.parent
for root in (SHAPES_ROOT, REFINE_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from analyze_predicted_noise_probe8_choose4 import (  # noqa: E402
    TARGETS,
    cache_group,
    load_target_data,
    spearman,
    write_csv,
)
from analyze_two_timestamp_shared_probe_score_average import (  # noqa: E402
    VARIANTS,
    load_scores,
)
from dataset_config import _prompt_tag  # noqa: E402


INDIVIDUAL = {
    "linear": "predicted_noise_jvp_signed",
    "square": "predicted_noise_jvp_l2_squared",
    "absolute": "predicted_noise_jvp_absolute",
}
COMBINED = {
    "square_mean": "predicted_noise_jvp_l2_squared_probe4",
    "absolute_mean": "predicted_noise_jvp_absolute_probe4",
    "term_root": "predicted_noise_jvp_probe_l2_probe4",
    "timestamp_root": "predicted_noise_jvp_timestamp_probe_l2_probe4",
}


def evaluate(
    score_root: Path,
    eval_root: Path,
    namespaces: dict[str, str],
    metadata: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    target_indices = incidence = true_values = None
    for reduction, namespace in namespaces.items():
        for variant, score_dir in VARIANTS.items():
            indices, scores = load_scores(score_root / f"traj_tracin_{namespace}" / score_dir)
            if target_indices is None:
                target_indices = indices
                incidence, true_values = load_target_data(cache_group(eval_root), indices)
            elif not np.array_equal(target_indices, indices):
                raise ValueError(f"score indices differ for {namespace}/{variant}")
            assert incidence is not None and true_values is not None
            prediction = scores @ incidence.T
            for target in TARGETS:
                rows.append(
                    {
                        **metadata,
                        "reduction": reduction,
                        "variant": variant,
                        "target": target,
                        "sign": "p1",
                        "lds_percent": 100.0 * spearman(prediction, true_values[target]),
                    }
                )
    return rows


def print_block(title: str, rows: list[dict[str, object]], kind: str, probe: int | None) -> None:
    selected = [
        row for row in rows
        if row["kind"] == kind and (probe is None or row.get("probe") == probe)
    ]
    print(f"\n{title} — FIXED P1")
    print(
        f"{'REDUCTION':15s} {'TARGET':24s} {'Q':>2s} {'RAW':>9s} "
        f"{'QUERY-L2':>9s} {'TRAIN-L2':>9s} {'BOTH-L2':>9s}"
    )
    print("-" * 86)
    for reduction in dict.fromkeys(str(row["reduction"]) for row in selected):
        for target in TARGETS:
            for query in range(10):
                lookup = {
                    str(row["variant"]): float(row["lds_percent"])
                    for row in selected
                    if row["reduction"] == reduction
                    and row["target"] == target
                    and row["query"] == query
                }
                if not lookup:
                    continue
                print(
                    f"{reduction:15s} {target:24s} {query:2d} "
                    + " ".join(f"{lookup[name]:+8.3f}%" for name in VARIANTS)
                )

    print("\n10-query mean")
    print(
        f"{'REDUCTION':15s} {'TARGET':24s} {'RAW':>9s} {'QUERY-L2':>9s} "
        f"{'TRAIN-L2':>9s} {'BOTH-L2':>9s}"
    )
    print("-" * 83)
    for reduction in dict.fromkeys(str(row["reduction"]) for row in selected):
        for target in TARGETS:
            values = []
            for variant in VARIANTS:
                group = [
                    float(row["lds_percent"])
                    for row in selected
                    if row["reduction"] == reduction
                    and row["target"] == target
                    and row["variant"] == variant
                ]
                values.append(float(np.mean(group)))
            print(
                f"{reduction:15s} {target:24s} "
                + " ".join(f"{value:+8.3f}%" for value in values)
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--probe-seeds", required=True)
    parser.add_argument("--combined-suffix", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    seeds = [int(value) for value in args.probe_seeds.split(",")]
    if len(seeds) != 4:
        raise ValueError(f"expected four probe seeds, got {seeds}")
    result_root = SHAPES_ROOT / "result" / args.experiment
    records = json.loads((SHAPES_ROOT / "queries_seed_0_9.json").read_text())["queries"]
    rows: list[dict[str, object]] = []

    for query, record in enumerate(records):
        prompt_tag = _prompt_tag(str(record["prompt"]))
        initial_seed = int(record["initial_seed"])
        score_root = (
            result_root / "attribution_score" / "prompted_solo"
            / f"train_seed_{args.train_seed}" / f"query_{prompt_tag}"
            / f"initial_seed_{initial_seed}"
        )
        eval_root = (
            result_root / "eval" / "prompted_solo" / f"query_{prompt_tag}"
            / f"initial_seed_{initial_seed}"
        )
        for probe, seed in enumerate(seeds, start=1):
            suffix = f"timestamp_shared_reference_seed{seed}"
            namespaces = {
                reduction: f"{base}_{suffix}"
                for reduction, base in INDIVIDUAL.items()
            }
            rows.extend(
                evaluate(
                    score_root,
                    eval_root,
                    namespaces,
                    {"kind": "individual", "probe": probe, "probe_seed": seed, "query": query},
                )
            )
        namespaces = {
            reduction: f"{base}_{args.combined_suffix}"
            for reduction, base in COMBINED.items()
        }
        rows.extend(
            evaluate(
                score_root,
                eval_root,
                namespaces,
                {"kind": "combined", "probe": "", "probe_seed": "", "query": query},
            )
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "per_query.csv", rows)
    for probe, seed in enumerate(seeds, start=1):
        print_block(f"INDIVIDUAL P{probe} seed={seed}", rows, "individual", probe)
    print_block("FOUR-PROBE COMBINATIONS", rows, "combined", None)
    print(f"\n[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
