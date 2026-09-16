#!/usr/bin/env python3
"""Print per-query and mean LDS for final probe-score reductions."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


SHAPES_ROOT = Path(__file__).resolve().parents[1]
TARGETS = (
    "endpoint_contarfactual",
    "noise_trajectory",
    "simple_loss",
    "traj_contarfactual",
)
VARIANTS = ("raw", "query_l2", "train_l2", "query_train_l2")
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--prediction-sign", choices=("p1", "m1"), default="m1")
    parser.add_argument("--num-probes", type=int, choices=(4, 8, 12), default=4)
    parser.add_argument(
        "--namespace-suffix",
        default="",
        help="Optional score namespace suffix, without the leading underscore.",
    )
    parser.add_argument(
        "--method",
        choices=("linear", "square", "all"),
        default="square",
        help="Print the no-square linear mean, the two square reductions, or all three.",
    )
    args = parser.parse_args()
    suffix = f"_{args.namespace_suffix}" if args.namespace_suffix else ""

    result_root = SHAPES_ROOT / "result" / args.experiment
    records = json.loads((SHAPES_ROOT / "queries_seed_0_9.json").read_text())["queries"]
    square_methods = (
        (
            "SQUARE EACH, THEN MEAN",
            f"traj_tracin_predicted_noise_jvp_final_square_then_mean_probe{args.num_probes}{suffix}",
            args.prediction_sign,
        ),
        (
            "MEAN, THEN SQUARE",
            f"traj_tracin_predicted_noise_jvp_final_mean_then_square_probe{args.num_probes}{suffix}",
            args.prediction_sign,
        ),
    )
    linear_method = (
        "LINEAR MEAN (NO SQUARE)",
        f"traj_tracin_predicted_noise_jvp_final_linear_mean_probe{args.num_probes}{suffix}",
        "p1",
    )
    methods = (
        (linear_method,)
        if args.method == "linear"
        else square_methods
        if args.method == "square"
        else (linear_method, *square_methods)
    )

    for title, namespace, prediction_sign in methods:
        values_by_target_variant: dict[tuple[str, str], list[float]] = {}
        print(f"\n{title}")
        print(
            f"{'TARGET':24s} {'Q':>2s} {'RAW':>9s} {'QUERY-L2':>9s} "
            f"{'TRAIN-L2':>9s} {'BOTH-L2':>9s}  PROMPT"
        )
        print("-" * 132)
        for query_id, record in enumerate(records):
            prompt = str(record["prompt"])
            prompt_tag = prompt.replace(",", "_")
            seed = int(record["initial_seed"])
            eval_root = (
                result_root
                / "eval"
                / "prompted_solo"
                / f"query_{prompt_tag}"
                / f"initial_seed_{seed}"
            )
            for target in TARGETS:
                row = []
                for variant in VARIANTS:
                    matches = list(
                        (
                            eval_root
                            / "lds"
                            / f"{namespace}_{variant}"
                            / target
                            / f"pred_kept_sign_{prediction_sign}"
                        ).glob("*/lds_summary.json")
                    )
                    if len(matches) != 1:
                        raise RuntimeError(
                            f"expected one LDS summary for query={query_id}, target={target}, "
                            f"variant={variant}; found {len(matches)} under {eval_root}"
                        )
                    value = float(json.loads(matches[0].read_text())["lds_percent"])
                    row.append(value)
                    values_by_target_variant.setdefault((target, variant), []).append(value)
                print(
                    f"{target:24s} {query_id:2d} "
                    + " ".join(f"{value:8.3f}%" for value in row)
                    + f"  {prompt_tag}"
                )

        print("\n10-query mean")
        print(
            f"{'TARGET':24s} {'RAW':>9s} {'QUERY-L2':>9s} "
            f"{'TRAIN-L2':>9s} {'BOTH-L2':>9s}"
        )
        print("-" * 68)
        for target in TARGETS:
            means = [
                statistics.mean(values_by_target_variant[(target, variant)])
                for variant in VARIANTS
            ]
            print(
                f"{target:24s} "
                + " ".join(f"{value:8.3f}%" for value in means)
            )


if __name__ == "__main__":
    main()
