#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT))

from dataset_config import DAS_DAMPING_SWEEP_VALUES, _prompt_tag


TARGETS = (
    "endpoint_contarfactual",
    "traj_contarfactual",
    "simple_loss",
    "noise_trajectory",
)


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(value) for value in text.replace(",", " ").split() if value.strip()]


def lambda_tag(value: float) -> str:
    return f"{float(value):g}".replace("+", "").replace("-", "neg_").replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print query-mean LDS for mixed local-Gram DAS scores.")
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--query-file", type=Path, default=SHAPES_ROOT / "queries_in_distribution_plus_zero_seed_100_219.json")
    parser.add_argument("--query-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--group-counts", default="1,2,4,8,10,20")
    parser.add_argument("--partition-seed", type=int, default=0)
    parser.add_argument("--namespace-prefix", default="factorized_mc4_indist100q_original100x1_localgram_mix")
    parser.add_argument("--baseline-namespace", default="factorized_mc4_indist100q_original100x1")
    parser.add_argument("--prediction-sign", choices=("p1", "m1"), default="m1")
    parser.add_argument("--lambdas", default=",".join(f"{float(value):g}" for value in DAS_DAMPING_SWEEP_VALUES))
    args = parser.parse_args()

    records = json.loads(args.query_file.read_text())["queries"]
    query_ids = parse_ints(args.query_ids)
    group_counts = parse_ints(args.group_counts)
    lambdas = parse_floats(args.lambdas)
    eval_base = SHAPES_ROOT / "result" / args.experiment / "eval" / "prompted_solo"

    print("DAS MIXED LOCAL-GRAM LDS: values are Q-mean ± population SD")
    print(f"{'GROUPS':>6s} {'SIZE':>6s} {'LAMBDA':>9s} {'ENDPOINT':>16s} {'TRAJ':>16s} {'SIMPLE':>16s} {'NOISE':>16s} {'N':>5s}")
    print("-" * 96)
    for group_count in group_counts:
        namespace = (
            args.baseline_namespace
            if group_count == 1
            else f"{args.namespace_prefix}_g{group_count}_seed{args.partition_seed}"
        )
        for damping in lambdas:
            cells = []
            counts = []
            for target in TARGETS:
                values = []
                for query_id in query_ids:
                    record = records[query_id]
                    root = (
                        eval_base / f"query_{_prompt_tag(str(record['prompt']))}"
                        / f"initial_seed_{int(record['initial_seed'])}" / "lds"
                        / f"das_{namespace}_lambda_{lambda_tag(damping)}" / target
                        / f"pred_kept_sign_{args.prediction_sign}"
                    )
                    matches = list(root.glob("*/lds_summary.json"))
                    if len(matches) == 1:
                        value = float(json.loads(matches[0].read_text())["lds_percent"])
                        if math.isfinite(value):
                            values.append(value)
                counts.append(len(values))
                if values:
                    cells.append(f"{statistics.fmean(values):+6.2f}±{statistics.pstdev(values):5.2f}%")
                else:
                    cells.append(f"{'MISSING':>16s}")
            size = 5000 // group_count
            print(
                f"{group_count:6d} {size:6d} {damping:9g} "
                + " ".join(f"{cell:>16s}" for cell in cells)
                + f" {min(counts):2d}/{len(query_ids):<2d}"
            )


if __name__ == "__main__":
    main()
