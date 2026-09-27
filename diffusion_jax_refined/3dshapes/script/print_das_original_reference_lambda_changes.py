#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_config import _prompt_tag


TARGETS = (
    "endpoint_contarfactual",
    "traj_contarfactual",
    "simple_loss",
    "noise_trajectory",
)
DISPLAY = {
    "endpoint_contarfactual": "ENDPOINT",
    "traj_contarfactual": "TRAJ-CF",
    "simple_loss": "SIMPLE",
    "noise_trajectory": "NOISE",
}


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value.strip()]


def load_values(experiment: str, namespace: str, sign: str, records, query_ids):
    name = "das" if not namespace else f"das_{namespace}"
    values = {}
    for q in query_ids:
        record = records[q]
        lds_root = (
            ROOT / "result" / experiment / "eval" / "prompted_solo"
            / f"query_{_prompt_tag(str(record['prompt']))}"
            / f"initial_seed_{int(record['initial_seed'])}" / "lds"
        )
        for target in TARGETS:
            pattern = f"{name}_lambda_*/{target}/pred_kept_sign_{sign}/*/lds_summary.json"
            for path in lds_root.glob(pattern):
                payload = json.loads(path.read_text())
                damping = float(payload["damping"])
                lds = float(payload["lds_percent"])
                if math.isfinite(lds):
                    values[(damping, q, target)] = lds
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--query-file", type=Path, default=ROOT / "queries_seed_0_9.json")
    parser.add_argument("--query-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--original-namespace", default="factorized_mc4_original100x1")
    parser.add_argument(
        "--reference-namespace",
        default="factorized_mc4_generation_reference100x1",
    )
    parser.add_argument("--prediction-sign", choices=("p1", "m1"), default="m1")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="print one aggregate comparison row per lambda instead of every query",
    )
    args = parser.parse_args()

    records = json.loads(args.query_file.read_text())["queries"]
    query_ids = parse_ints(args.query_ids)
    bad = [q for q in query_ids if q < 0 or q >= len(records)]
    if bad:
        raise ValueError(f"query ids outside manifest range: {bad}")
    original = load_values(
        args.experiment, args.original_namespace, args.prediction_sign, records, query_ids
    )
    reference = load_values(
        args.experiment, args.reference_namespace, args.prediction_sign, records, query_ids
    )
    lambdas = sorted(
        set(key[0] for key in original) & set(key[0] for key in reference)
    )
    if not lambdas:
        raise RuntimeError("No common original/reference DAS lambdas found")

    print(
        f"DAS GENERATION-TRAJECTORY MINUS ORIGINAL | queries={len(query_ids)} | "
        f"sign={args.prediction_sign}"
    )
    print(
        f"{'LAMBDA':>8s} {'TARGET':>9s} {'ORIGINAL':>11s} {'TRAJ-XT':>11s} "
        f"{'DELTA':>11s} {'XT-WINS':>9s}"
    )
    print("-" * 66)
    for damping in lambdas:
        target_values = {}
        for target in TARGETS:
            original_values = []
            reference_values = []
            for q in query_ids:
                key = (damping, q, target)
                if key not in original or key not in reference:
                    raise RuntimeError(
                        f"Incomplete LDS at lambda={damping:g}, query={q}, target={target}"
                    )
                original_values.append(original[key])
                reference_values.append(reference[key])
            deltas = [new - old for old, new in zip(original_values, reference_values)]
            target_values[target] = (original_values, reference_values, deltas)
            print(
                f"{damping:8g} {DISPLAY[target]:>9s} "
                f"{statistics.fmean(original_values):+10.3f}% "
                f"{statistics.fmean(reference_values):+10.3f}% "
                f"{statistics.fmean(deltas):+10.3f}% "
                f"{sum(delta > 0 for delta in deltas):3d}/{len(deltas):<3d}"
            )

        if args.summary_only:
            continue
        print(f"\nLAMBDA={damping:g} — PER QUERY")
        print(
            f"{'Q':>3s} "
            + " ".join(
                f"{DISPLAY[target] + '-O':>11s} {DISPLAY[target] + '-XT':>11s} {DISPLAY[target] + '-D':>11s}"
                for target in TARGETS
            )
        )
        print("-" * 151)
        for position, q in enumerate(query_ids):
            cells = []
            for target in TARGETS:
                original_values, reference_values, deltas = target_values[target]
                cells.append(
                    f"{original_values[position]:+10.3f}% "
                    f"{reference_values[position]:+10.3f}% "
                    f"{deltas[position]:+10.3f}%"
                )
            print(f"{q:3d} " + " ".join(cells))


if __name__ == "__main__":
    main()
