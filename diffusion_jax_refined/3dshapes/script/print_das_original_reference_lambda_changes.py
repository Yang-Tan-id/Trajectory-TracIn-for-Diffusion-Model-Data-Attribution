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


TARGETS = ("endpoint_contarfactual", "traj_contarfactual")


def load_values(experiment: str, namespace: str, sign: str):
    records = json.loads((ROOT / "queries_seed_0_9.json").read_text())["queries"]
    name = "das" if not namespace else f"das_{namespace}"
    values = {}
    for q, record in enumerate(records):
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
    parser.add_argument("--original-namespace", default="factorized_mc4_original100x1")
    parser.add_argument(
        "--reference-namespace",
        default="factorized_mc4_generation_reference100x1",
    )
    parser.add_argument("--prediction-sign", choices=("p1", "m1"), default="m1")
    args = parser.parse_args()

    original = load_values(args.experiment, args.original_namespace, args.prediction_sign)
    reference = load_values(args.experiment, args.reference_namespace, args.prediction_sign)
    lambdas = sorted(
        set(key[0] for key in original) & set(key[0] for key in reference)
    )
    if not lambdas:
        raise RuntimeError("No common original/reference DAS lambdas found")

    for damping in lambdas:
        print(f"\nLAMBDA={damping:g} — REFERENCE MINUS ORIGINAL")
        print(
            f"{'Q':>2s} {'END-ORIG':>10s} {'END-REF':>10s} {'END-DELTA':>11s} "
            f"{'TRAJ-ORIG':>11s} {'TRAJ-REF':>10s} {'TRAJ-DELTA':>12s}"
        )
        print("-" * 83)
        end_delta = []
        traj_delta = []
        for q in range(10):
            keys = [
                (damping, q, "endpoint_contarfactual"),
                (damping, q, "traj_contarfactual"),
            ]
            if not all(key in original and key in reference for key in keys):
                raise RuntimeError(f"Incomplete LDS at lambda={damping:g}, query={q}")
            eo, to = (original[key] for key in keys)
            er, tr = (reference[key] for key in keys)
            de, dt = er - eo, tr - to
            end_delta.append(de)
            traj_delta.append(dt)
            print(
                f"{q:2d} {eo:+9.3f}% {er:+9.3f}% {de:+10.3f}% "
                f"{to:+10.3f}% {tr:+9.3f}% {dt:+11.3f}%"
            )
        print(
            f"MEAN {'':>19s}{statistics.fmean(end_delta):+10.3f}% "
            f"{'':>22s}{statistics.fmean(traj_delta):+11.3f}%"
        )


if __name__ == "__main__":
    main()
