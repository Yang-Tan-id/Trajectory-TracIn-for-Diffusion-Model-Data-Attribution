#!/usr/bin/env python3
"""Print all LDS results for the shared orthogonal four-probe experiment."""

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
METHODS = (
    (
        "LINEAR MEAN (sign +1)",
        "traj_tracin_predicted_noise_jvp_final_linear_mean_probe4_orthogonal_shared",
        "p1",
    ),
    (
        "MEAN SQUARE (sign -1)",
        "traj_tracin_predicted_noise_jvp_final_square_then_mean_probe4_orthogonal_shared",
        "m1",
    ),
    (
        "SQUARE MEAN (sign -1)",
        "traj_tracin_predicted_noise_jvp_final_mean_then_square_probe4_orthogonal_shared",
        "m1",
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    args = parser.parse_args()

    result_root = SHAPES_ROOT / "result" / args.experiment
    records = json.loads((SHAPES_ROOT / "queries_seed_0_9.json").read_text())["queries"]

    for title, namespace, sign_tag in METHODS:
        grouped: dict[tuple[str, str], list[float]] = {}
        print(f"\n{title}")
        print(
            f"{'TARGET':24s} {'Q':>2s} {'RAW':>9s} {'QUERY-L2':>9s} "
            f"{'TRAIN-L2':>9s} {'BOTH-L2':>9s}  PROMPT"
        )
        print("-" * 132)
        for query_id, record in enumerate(records):
            prompt_tag = str(record["prompt"]).replace(",", "_")
            eval_root = (
                result_root
                / "eval"
                / "prompted_solo"
                / f"query_{prompt_tag}"
                / f"initial_seed_{int(record['initial_seed'])}"
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
                            / f"pred_kept_sign_{sign_tag}"
                        ).glob("*/lds_summary.json")
                    )
                    if len(matches) != 1:
                        raise RuntimeError(
                            f"expected one summary for query={query_id}, target={target}, "
                            f"variant={variant}; found {len(matches)}"
                        )
                    value = float(json.loads(matches[0].read_text())["lds_percent"])
                    row.append(value)
                    grouped.setdefault((target, variant), []).append(value)
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
                statistics.mean(grouped[(target, variant)])
                for variant in VARIANTS
            ]
            print(
                f"{target:24s} "
                + " ".join(f"{value:8.3f}%" for value in means)
            )


if __name__ == "__main__":
    main()
