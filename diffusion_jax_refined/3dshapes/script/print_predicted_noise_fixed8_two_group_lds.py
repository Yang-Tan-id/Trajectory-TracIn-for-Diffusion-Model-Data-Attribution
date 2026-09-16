#!/usr/bin/env python3
"""Compare fixed orthogonal probes 1-4 against probes 5-8 per query."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_predicted_noise_probe8_choose4 import (
    TARGETS,
    VARIANTS,
    cache_group,
    load_probe_scores,
    load_target_data,
    spearman,
    write_csv,
)


SHAPES_ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "fixed_1_4": (0, 1, 2, 3),
    "fixed_5_8": (4, 5, 6, 7),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prediction-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument(
        "--reduction",
        choices=("linear", "termwise_square"),
        default="linear",
    )
    parser.add_argument(
        "--namespace-suffix",
        default="orthogonal_extended_groupaudit",
    )
    args = parser.parse_args()

    result_root = SHAPES_ROOT / "result" / args.experiment
    staging_namespace = (
        "traj_tracin_predicted_noise_jvp_final_post_square_probe8"
        if args.reduction == "linear"
        else "traj_tracin_predicted_noise_jvp_termwise_squared_per_probe_probe8"
    )
    shard_dir = (
        result_root
        / "stream_score"
        / f"{staging_namespace}_{args.namespace_suffix}"
        / f"train_seed_{args.train_seed}"
        / f"run_{args.run_id}"
        / "shards"
    )
    probe_scores, score_indices = load_probe_scores(shard_dir, num_probes=8)
    records = json.loads((SHAPES_ROOT / "queries_seed_0_9.json").read_text())["queries"]
    rows: list[dict[str, object]] = []

    for query_id, record in enumerate(records):
        prompt = str(record["prompt"])
        prompt_tag = prompt.replace(",", "_")
        eval_root = (
            result_root
            / "eval"
            / "prompted_solo"
            / f"query_{prompt_tag}"
            / f"initial_seed_{int(record['initial_seed'])}"
        )
        incidence, true_values = load_target_data(cache_group(eval_root), score_indices)
        for variant, values in probe_scores.items():
            per_probe_predictions = (
                args.prediction_sign * values[:, query_id, :] @ incidence.T
            )
            for group, indices in GROUPS.items():
                prediction = per_probe_predictions[list(indices)].mean(axis=0)
                for target in TARGETS:
                    rows.append(
                        {
                            "group": group,
                            "query": query_id,
                            "target": target,
                            "variant": variant,
                            "lds_percent": 100.0
                            * spearman(prediction, true_values[target]),
                            "prompt": prompt_tag,
                        }
                    )

    lookup = {
        (str(row["group"]), int(row["query"]), str(row["target"]), str(row["variant"])):
        float(row["lds_percent"])
        for row in rows
    }
    title = "LINEAR" if args.reduction == "linear" else "PRODUCT SQUARE"
    print(f"FIXED ORTHOGONAL 4-PROBE GROUPS — {title} sign {args.prediction_sign:+g}")
    print(
        f"{'TARGET':24s} {'Q':>2s} | "
        f"{'1-4 RAW':>8s} {'1-4 QL2':>8s} {'1-4 TL2':>8s} {'1-4 BL2':>8s} | "
        f"{'5-8 RAW':>8s} {'5-8 QL2':>8s} {'5-8 TL2':>8s} {'5-8 BL2':>8s}  PROMPT"
    )
    print("-" * 151)
    variant_order = ("raw", "query_l2", "train_l2", "query_train_l2")
    for query_id, record in enumerate(records):
        prompt_tag = str(record["prompt"]).replace(",", "_")
        for target in TARGETS:
            left = [lookup[("fixed_1_4", query_id, target, variant)] for variant in variant_order]
            right = [lookup[("fixed_5_8", query_id, target, variant)] for variant in variant_order]
            print(
                f"{target:24s} {query_id:2d} | "
                + " ".join(f"{value:7.3f}%" for value in left)
                + " | "
                + " ".join(f"{value:7.3f}%" for value in right)
                + f"  {prompt_tag}"
            )

    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["group"]), str(row["target"]), str(row["variant"]))].append(
            float(row["lds_percent"])
        )
    print("\n10-query mean")
    print(
        f"{'TARGET':24s} | "
        f"{'1-4 RAW':>8s} {'1-4 QL2':>8s} {'1-4 TL2':>8s} {'1-4 BL2':>8s} | "
        f"{'5-8 RAW':>8s} {'5-8 QL2':>8s} {'5-8 TL2':>8s} {'5-8 BL2':>8s}"
    )
    print("-" * 111)
    for target in TARGETS:
        left = [statistics.mean(grouped[("fixed_1_4", target, variant)]) for variant in variant_order]
        right = [statistics.mean(grouped[("fixed_5_8", target, variant)]) for variant in variant_order]
        print(
            f"{target:24s} | "
            + " ".join(f"{value:7.3f}%" for value in left)
            + " | "
            + " ".join(f"{value:7.3f}%" for value in right)
        )

    output_dir = (
        result_root
        / "eval"
        / f"predicted_noise_fixed8_two_group_{args.reduction}"
        / f"run_{args.run_id}"
    )
    write_csv(output_dir / "per_query.csv", rows)
    print(f"[saved] {output_dir / 'per_query.csv'}")


if __name__ == "__main__":
    main()
