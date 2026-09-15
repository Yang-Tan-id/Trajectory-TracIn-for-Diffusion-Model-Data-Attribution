#!/usr/bin/env python3
"""Compare original squared DAS with eight-probe linear Both-L2 LDS."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


SHAPES_ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("endpoint_contarfactual", "traj_contarfactual")
PROBE8_ALGORITHM = (
    "traj_tracin_predicted_noise_jvp_final_linear_mean_probe8_query_train_l2"
)


def lambda_tag(value: float) -> str:
    return f"{float(value):g}".replace("+", "").replace("-", "neg_").replace(".", "p")


def read_one(root: Path, pattern: str) -> float:
    matches = list(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one LDS summary for {pattern}; found {len(matches)} under {root}")
    return float(json.loads(matches[0].read_text())["lds_percent"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--das-lambda", type=float, default=200.0)
    args = parser.parse_args()

    result_root = SHAPES_ROOT / "result" / args.experiment
    records = json.loads((SHAPES_ROOT / "queries_seed_0_9.json").read_text())["queries"]
    das_algorithm = f"das_lambda_{lambda_tag(args.das_lambda)}"

    print(
        f"DAS squared (lambda={args.das_lambda:g}, sign=-1)  vs  "
        "Probe8 linear Both-L2 (sign=+1)"
    )
    for target in TARGETS:
        das_values = []
        probe_values = []
        print(f"\nTARGET: {target.replace('contarfactual', 'counterfactual')}")
        print(f"{'Q':>2s} {'DAS-SQ':>10s} {'P8-LIN-BL2':>12s} {'DELTA':>10s}  PROMPT")
        print("-" * 112)
        for query_id, record in enumerate(records):
            prompt = str(record["prompt"])
            prompt_tag = prompt.replace(",", "_")
            lds_root = (
                result_root
                / "eval"
                / "prompted_solo"
                / f"query_{prompt_tag}"
                / f"initial_seed_{int(record['initial_seed'])}"
                / "lds"
            )
            das = read_one(
                lds_root,
                f"{das_algorithm}/{target}/pred_kept_sign_m1/*/lds_summary.json",
            )
            probe = read_one(
                lds_root,
                f"{PROBE8_ALGORITHM}/{target}/pred_kept_sign_p1/*/lds_summary.json",
            )
            das_values.append(das)
            probe_values.append(probe)
            print(
                f"{query_id:2d} {das:9.3f}% {probe:11.3f}% {probe - das:+9.3f}%  {prompt_tag}"
            )

        deltas = [probe - das for probe, das in zip(probe_values, das_values)]
        wins = sum(delta > 0 for delta in deltas)
        print("-" * 112)
        print(
            f"MEAN {statistics.mean(das_values):7.3f}% "
            f"{statistics.mean(probe_values):11.3f}% "
            f"{statistics.mean(deltas):+9.3f}%  Probe8 wins {wins}/10"
        )
        print(
            f"STD  {statistics.pstdev(das_values):7.3f}% "
            f"{statistics.pstdev(probe_values):11.3f}%"
        )

    all_das = []
    all_probe = []
    for target in TARGETS:
        for record in records:
            prompt_tag = str(record["prompt"]).replace(",", "_")
            lds_root = (
                result_root
                / "eval"
                / "prompted_solo"
                / f"query_{prompt_tag}"
                / f"initial_seed_{int(record['initial_seed'])}"
                / "lds"
            )
            all_das.append(
                read_one(
                    lds_root,
                    f"{das_algorithm}/{target}/pred_kept_sign_m1/*/lds_summary.json",
                )
            )
            all_probe.append(
                read_one(
                    lds_root,
                    f"{PROBE8_ALGORITHM}/{target}/pred_kept_sign_p1/*/lds_summary.json",
                )
            )
    print("\nJOINT ENDPOINT + TRAJECTORY")
    print(f"DAS squared             : {statistics.mean(all_das):8.3f}%")
    print(f"Probe8 linear Both-L2   : {statistics.mean(all_probe):8.3f}%")
    print(f"Probe8 - DAS            : {statistics.mean(all_probe) - statistics.mean(all_das):+8.3f}%")


if __name__ == "__main__":
    main()
