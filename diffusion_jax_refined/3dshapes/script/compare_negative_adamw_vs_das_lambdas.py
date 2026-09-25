#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path
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
DEFAULT_LAMBDAS = "0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000,2000,5000,10000"


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value]


def parse_floats(text: str) -> list[float]:
    return [float(value) for value in text.replace(",", " ").split() if value]


def load_one_summary(pattern: Path, description: str) -> dict:
    # The wildcard is in an ancestor component (the LDS subset-group directory),
    # so Path.glob(pattern.name) from pattern.parent would treat that ``*`` as a
    # literal directory. Expand the complete path instead.
    matches = [Path(value) for value in sorted(glob.glob(str(pattern)))]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {description}; found {len(matches)} for {pattern}"
        )
    return json.loads(matches[0].read_text())


def relation(negative_adamw: float, das: float, tolerance: float) -> str:
    if math.isclose(negative_adamw, das, rel_tol=0.0, abs_tol=tolerance):
        return "equal"
    if negative_adamw > das:
        return "negative_adamw_gt_das"
    return "negative_adamw_lt_das"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Negate per-query AdamW LDS values, compare them with DAS at every lambda, "
            "and report above/below/equal proportions."
        )
    )
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument(
        "--query-file",
        type=Path,
        default=SHAPES_ROOT / "queries_in_distribution_plus_zero_seed_100_219.json",
    )
    parser.add_argument("--query-ids", default=",".join(str(i) for i in range(100)))
    parser.add_argument(
        "--adamw-lds-namespace",
        default=(
            "traj_tracin_adamw_full_aligned10x10_timestamp_sum_squared_"
            "previous_lr_ref100q_query_train_l2"
        ),
        help="exact directory name below each query's lds directory",
    )
    parser.add_argument("--adamw-prediction-sign", choices=("p1", "m1"), default="p1")
    parser.add_argument(
        "--das-artifact-namespace",
        default="factorized_mc4_indist100q_original100x1",
    )
    parser.add_argument("--das-prediction-sign", choices=("p1", "m1"), default="m1")
    parser.add_argument("--lambdas", default=DEFAULT_LAMBDAS)
    parser.add_argument("--equal-tolerance", type=float, default=1e-12)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="default: result/EXPERIMENT/eval/negative_adamw_vs_das_q0_99",
    )
    args = parser.parse_args()

    records = json.loads(args.query_file.read_text())["queries"]
    query_ids = parse_ints(args.query_ids)
    lambdas = parse_floats(args.lambdas)
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("--query-ids contains duplicates")
    if len(set(lambdas)) != len(lambdas):
        raise ValueError("--lambdas contains duplicates")

    result_root = SHAPES_ROOT / "result" / args.experiment
    output_dir = args.output_dir or (
        result_root / "eval" / "negative_adamw_vs_das_q0_99"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    das_base = f"das_{args.das_artifact_namespace.strip().strip('_/')}"
    adamw_values: dict[tuple[int, str], float] = {}
    das_values: dict[tuple[float, int, str], float] = {}

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
            adamw_pattern = (
                lds_root
                / args.adamw_lds_namespace
                / target
                / f"pred_kept_sign_{args.adamw_prediction_sign}"
                / "*"
                / "lds_summary.json"
            )
            adamw_payload = load_one_summary(
                adamw_pattern, f"AdamW LDS summary for Q{query_id}/{target}"
            )
            adamw_lds = float(adamw_payload["lds_percent"])
            if not math.isfinite(adamw_lds):
                raise RuntimeError(f"Non-finite AdamW LDS for Q{query_id}/{target}")
            adamw_values[(query_id, target)] = adamw_lds

            for damping in lambdas:
                lambda_text = f"{damping:g}"
                das_pattern = (
                    lds_root
                    / f"{das_base}_lambda_{lambda_text}"
                    / target
                    / f"pred_kept_sign_{args.das_prediction_sign}"
                    / "*"
                    / "lds_summary.json"
                )
                das_payload = load_one_summary(
                    das_pattern,
                    f"DAS LDS summary for lambda={lambda_text}/Q{query_id}/{target}",
                )
                das_lds = float(das_payload["lds_percent"])
                if not math.isfinite(das_lds):
                    raise RuntimeError(
                        f"Non-finite DAS LDS for lambda={lambda_text}/Q{query_id}/{target}"
                    )
                das_values[(damping, query_id, target)] = das_lds

    detail_path = output_dir / "per_query_comparison.csv"
    detail_rows: list[dict[str, object]] = []
    counts: dict[tuple[float, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for damping in lambdas:
        for query_id in query_ids:
            for target in TARGETS:
                adamw_original = adamw_values[(query_id, target)]
                negative_adamw = -adamw_original
                das_lds = das_values[(damping, query_id, target)]
                outcome = relation(negative_adamw, das_lds, args.equal_tolerance)
                counts[(damping, target)][outcome] += 1
                counts[(damping, "ALL")][outcome] += 1
                detail_rows.append(
                    {
                        "lambda": f"{damping:g}",
                        "query": query_id,
                        "target": target,
                        "adamw_original_lds_percent": adamw_original,
                        "negative_adamw_lds_percent": negative_adamw,
                        "das_lds_percent": das_lds,
                        "negative_adamw_minus_das": negative_adamw - das_lds,
                        "relation": outcome,
                    }
                )

    with detail_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)

    summary_path = output_dir / "proportions.csv"
    summary_rows: list[dict[str, object]] = []
    for damping in lambdas:
        for target in (*TARGETS, "ALL"):
            cell = counts[(damping, target)]
            total = sum(cell.values())
            gt = cell["negative_adamw_gt_das"]
            lt = cell["negative_adamw_lt_das"]
            eq = cell["equal"]
            summary_rows.append(
                {
                    "lambda": f"{damping:g}",
                    "target": target,
                    "n": total,
                    "negative_adamw_gt_das_count": gt,
                    "negative_adamw_gt_das_percent": 100.0 * gt / total,
                    "negative_adamw_lt_das_count": lt,
                    "negative_adamw_lt_das_percent": 100.0 * lt / total,
                    "equal_count": eq,
                    "equal_percent": 100.0 * eq / total,
                }
            )

    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(
        "Comparison: NEG-AdamW = -1 * AdamW LDS; "
        f"AdamW sign={args.adamw_prediction_sign}; DAS sign={args.das_prediction_sign}"
    )
    print(
        f"{'LAMBDA':>8s} {'TARGET':>13s} {'N':>4s} "
        f"{'NEG-AW>DAS':>12s} {'NEG-AW<DAS':>12s} {'EQUAL':>9s}"
    )
    print("-" * 66)
    for row in summary_rows:
        target = str(row["target"])
        display = "ALL" if target == "ALL" else DISPLAY_NAMES[target]
        print(
            f"{str(row['lambda']):>8s} {display:>13s} {int(row['n']):4d} "
            f"{float(row['negative_adamw_gt_das_percent']):11.1f}% "
            f"{float(row['negative_adamw_lt_das_percent']):11.1f}% "
            f"{float(row['equal_percent']):8.1f}%"
        )
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved per-query details: {detail_path}")


if __name__ == "__main__":
    main()
