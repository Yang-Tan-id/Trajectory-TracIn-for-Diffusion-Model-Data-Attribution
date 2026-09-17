#!/usr/bin/env python3
"""Print LDS for each independently generated timestamp-shared probe."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
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


REDUCTIONS = {
    "square": "predicted_noise_jvp_l2_squared",
    "root": "predicted_noise_jvp_probe_l2",
    "timestamp_root": "predicted_noise_jvp_timestamp_probe_l2",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--probe-seeds", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    seeds = [int(value) for value in args.probe_seeds.split(",")]
    result_root = SHAPES_ROOT / "result" / args.experiment
    records = json.loads((SHAPES_ROOT / "queries_seed_0_9.json").read_text())["queries"]
    rows: list[dict[str, object]] = []

    for probe_number, seed in enumerate(seeds, start=1):
        suffix = f"timestamp_shared_individual_seed{seed}_own_trajectory"
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
            target_indices = None
            incidence = None
            true_values = None
            for reduction, base in REDUCTIONS.items():
                namespace = f"traj_tracin_{base}_{suffix}"
                for variant, score_dir_name in VARIANTS.items():
                    indices, scores = load_scores(score_root / namespace / score_dir_name)
                    if target_indices is None:
                        target_indices = indices
                        incidence, true_values = load_target_data(
                            cache_group(eval_root), indices
                        )
                    elif not np.array_equal(target_indices, indices):
                        raise ValueError(
                            f"probe {probe_number} Q{query}: score indices differ"
                        )
                    assert incidence is not None and true_values is not None
                    prediction = scores @ incidence.T
                    for sign, multiplier in (("p1", 1.0), ("m1", -1.0)):
                        for target in TARGETS:
                            rows.append(
                                {
                                    "probe": probe_number,
                                    "probe_seed": seed,
                                    "query": query,
                                    "reduction": reduction,
                                    "variant": variant,
                                    "sign": sign,
                                    "target": target,
                                    "lds_percent": 100.0 * spearman(
                                        multiplier * prediction,
                                        true_values[target],
                                    ),
                                }
                            )
        print(f"[probe {probe_number}/8] seed={seed} loaded", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "per_query.csv", rows)
    grouped: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            row["probe"], row["probe_seed"], row["reduction"], row["sign"],
            row["target"], row["variant"],
        )
        grouped[key].append(float(row["lds_percent"]))
    means = [
        {
            "probe": key[0], "probe_seed": key[1], "reduction": key[2],
            "sign": key[3], "target": key[4], "variant": key[5],
            "mean_lds_percent": float(np.mean(values)),
        }
        for key, values in grouped.items()
    ]
    write_csv(args.out_dir / "ten_query_means.csv", means)

    lookup = {
        (row["probe"], row["reduction"], row["sign"], row["target"], row["variant"]):
        row["mean_lds_percent"]
        for row in means
    }
    for probe_number, seed in enumerate(seeds, start=1):
        for reduction in REDUCTIONS:
            for sign in ("p1", "m1"):
                print(f"\nPROBE {probe_number} seed={seed} {reduction.upper()} sign={sign}")
                print(
                    f"{'TARGET':24s} {'RAW':>9s} {'QUERY-L2':>9s} "
                    f"{'TRAIN-L2':>9s} {'BOTH-L2':>9s}"
                )
                print("-" * 68)
                for target in TARGETS:
                    values = [
                        lookup[(probe_number, reduction, sign, target, variant)]
                        for variant in VARIANTS
                    ]
                    print(f"{target:24s} " + " ".join(f"{v:8.3f}%" for v in values))
    print(f"\n[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
