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

from dataset_config import _prompt_tag


TARGETS = (
    "endpoint_contarfactual",
    "traj_contarfactual",
    "simple_loss",
    "noise_trajectory",
)
DISPLAY_NAMES = {
    "endpoint_contarfactual": "ENDPOINT-CF",
    "traj_contarfactual": "TRAJ-CF",
    "simple_loss": "SIMPLE-LOSS",
    "noise_trajectory": "NOISE-TRAJ",
}


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print requested-query LDS summaries for every DAS lambda and target."
    )
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--artifact-namespace", default="aligned10x10")
    parser.add_argument(
        "--query-file", type=Path, default=SHAPES_ROOT / "queries_seed_0_9.json",
    )
    parser.add_argument("--query-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--prediction-sign", choices=("p1", "m1"), default="m1")
    parser.add_argument(
        "--show-std", action="store_true",
        help="print population standard deviation across requested queries",
    )
    parser.add_argument(
        "--per-query-best",
        action="store_true",
        help="print the independently best endpoint and trajectory lambda for every query",
    )
    parser.add_argument(
        "--best-lambda-per-target",
        action="store_true",
        help="select one lambda by ten-query mean for each target, then print every query",
    )
    args = parser.parse_args()

    records = json.loads(args.query_file.read_text())["queries"]
    query_ids = parse_ints(args.query_ids)
    namespace = args.artifact_namespace.strip().strip("_/")
    das_name = "das" if not namespace else f"das_{namespace}"
    result_root = SHAPES_ROOT / "result" / args.experiment

    # values[lambda][target][query_id] = LDS percent
    values: dict[float, dict[str, dict[int, float]]] = {}
    for query_id in query_ids:
        if query_id < 0 or query_id >= len(records):
            raise ValueError(f"query id {query_id} is outside [0, {len(records) - 1}]")
        record = records[query_id]
        lds_root = (
            result_root
            / "eval"
            / "prompted_solo"
            / f"query_{_prompt_tag(str(record['prompt']))}"
            / f"initial_seed_{int(record['initial_seed'])}"
            / "lds"
        )
        for target in TARGETS:
            pattern = (
                f"{das_name}_lambda_*/{target}/"
                f"pred_kept_sign_{args.prediction_sign}/*/lds_summary.json"
            )
            matches = list(lds_root.glob(pattern))
            for path in matches:
                payload = json.loads(path.read_text())
                damping = float(payload["damping"])
                lds = float(payload["lds_percent"])
                if math.isfinite(lds):
                    values.setdefault(damping, {}).setdefault(target, {})[query_id] = lds

    if not values:
        raise RuntimeError(
            f"No LDS summaries found for {das_name}, sign={args.prediction_sign}, "
            f"experiment={args.experiment}."
        )

    if args.per_query_best:
        print(f"DAS PER-QUERY BEST LAMBDA: {das_name}, sign={args.prediction_sign}")
        print(
            f"{'Q':>2s} {'END-LAMBDA':>11s} {'END-LDS':>10s} "
            f"{'TRAJ-LAMBDA':>12s} {'TRAJ-LDS':>10s}"
        )
        print("-" * 54)
        for query_id in query_ids:
            best = {}
            for target in ("endpoint_contarfactual", "traj_contarfactual"):
                candidates = [
                    (target_values[target][query_id], damping)
                    for damping, target_values in values.items()
                    if target in target_values and query_id in target_values[target]
                ]
                if not candidates:
                    raise RuntimeError(f"No {target} LDS values found for query {query_id}")
                best[target] = max(candidates, key=lambda item: (item[0], -item[1]))
            endpoint_lds, endpoint_lambda = best["endpoint_contarfactual"]
            trajectory_lds, trajectory_lambda = best["traj_contarfactual"]
            print(
                f"{query_id:2d} {endpoint_lambda:11g} {endpoint_lds:+9.3f}% "
                f"{trajectory_lambda:12g} {trajectory_lds:+9.3f}%"
            )
        return

    if args.best_lambda_per_target:
        selected_targets = (
            "endpoint_contarfactual",
            "traj_contarfactual",
            "simple_loss",
        )
        selected = {}
        for target in selected_targets:
            candidates = []
            for damping, target_values in values.items():
                query_values = target_values.get(target, {})
                if all(query_id in query_values for query_id in query_ids):
                    mean = statistics.fmean(query_values[query_id] for query_id in query_ids)
                    candidates.append((mean, damping))
            if not candidates:
                raise RuntimeError(f"No complete ten-query lambda candidate for {target}")
            selected[target] = max(candidates, key=lambda item: (item[0], -item[1]))

        endpoint_mean, endpoint_lambda = selected["endpoint_contarfactual"]
        trajectory_mean, trajectory_lambda = selected["traj_contarfactual"]
        simple_mean, simple_lambda = selected["simple_loss"]
        print(f"DAS FIXED BEST LAMBDA BY TARGET: {das_name}, sign={args.prediction_sign}")
        print(
            f"endpoint lambda={endpoint_lambda:g}, 10-query mean={endpoint_mean:+.3f}% | "
            f"trajectory lambda={trajectory_lambda:g}, 10-query mean={trajectory_mean:+.3f}% | "
            f"simple lambda={simple_lambda:g}, 10-query mean={simple_mean:+.3f}%"
        )
        print(f"{'Q':>2s} {'END-LDS':>10s} {'TRAJ-LDS':>10s} {'SIMPLE-LDS':>11s}")
        print("-" * 39)
        for query_id in query_ids:
            endpoint_lds = values[endpoint_lambda]["endpoint_contarfactual"][query_id]
            trajectory_lds = values[trajectory_lambda]["traj_contarfactual"][query_id]
            simple_lds = values[simple_lambda]["simple_loss"][query_id]
            print(
                f"{query_id:2d} {endpoint_lds:+9.3f}% {trajectory_lds:+9.3f}% "
                f"{simple_lds:+10.3f}%"
            )
        print(
            f"MEAN {endpoint_mean:+7.3f}% {trajectory_mean:+9.3f}% "
            f"{simple_mean:+10.3f}%"
        )
        return

    expected_n = len(query_ids)
    print(
        f"DAS ALL LAMBDAS: {das_name}, sign={args.prediction_sign}, "
        f"values are mean{' ± population SD' if args.show_std else ''} over requested queries"
    )
    print(
        f"{'LAMBDA':>10s} "
        + " ".join(f"{DISPLAY_NAMES[target]:>14s}" for target in TARGETS)
        + f" {'JOINT':>12s}  {'N PER TARGET':>14s}"
    )
    print("-" * 101)

    for damping in sorted(values):
        target_means: list[float] = []
        counts: list[int] = []
        cells: list[str] = []
        for target in TARGETS:
            query_values = values[damping].get(target, {})
            present = [query_values[q] for q in query_ids if q in query_values]
            counts.append(len(present))
            if present:
                mean = statistics.fmean(present)
                target_means.append(mean)
                if args.show_std:
                    std = statistics.pstdev(present)
                    cells.append(f"{mean:+6.3f}±{std:5.3f}%")
                else:
                    cells.append(f"{mean:13.3f}%")
            else:
                cells.append(f"{'MISSING':>14s}")
        joint = statistics.fmean(target_means) if len(target_means) == len(TARGETS) else math.nan
        count_text = "/".join(str(count) for count in counts)
        warning = "" if all(count == expected_n for count in counts) else "  INCOMPLETE"
        print(
            f"{damping:10g} "
            + " ".join(cells)
            + f" {joint:11.3f}%  {count_text:>14s}{warning}"
        )


if __name__ == "__main__":
    main()
