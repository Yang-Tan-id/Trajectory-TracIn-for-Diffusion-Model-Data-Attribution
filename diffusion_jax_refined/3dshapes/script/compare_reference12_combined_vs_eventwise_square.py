#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
VARIANTS = ("raw", "query_l2", "train_l2", "query_train_l2")
SCHEMES = {
    "raw": (
        "traj_tracin_predicted_noise_jvp_l2_squared_probe12_adamw4_four_reference12",
        "traj_tracin_predicted_noise_jvp_l2_squared_probe12_adamw4_eventwise_square_sum_reference12",
    ),
    "residual": (
        "traj_tracin_predicted_noise_jvp_l2_squared_probe12_adamw4_four_residual_reference12",
        "traj_tracin_predicted_noise_jvp_l2_squared_probe12_adamw4_eventwise_residual_square_sum_reference12",
    ),
}


def read_lds(root: Path, scheme: str, variant: str, target: str, sign: str) -> float:
    matches = list((root / "lds" / f"{scheme}_{variant}" / target / f"pred_kept_sign_{sign}").glob("*/lds_summary.json"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one result for {scheme}/{variant}/{target}; found {len(matches)}")
    return float(json.loads(matches[0].read_text())["lds_percent"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--prediction-sign", choices=("p1", "m1"), default="p1")
    args = parser.parse_args()
    records = json.loads((ROOT / "queries_seed_0_9.json").read_text())["queries"]
    result = ROOT / "result" / args.experiment / "eval" / "prompted_solo"

    for family, (old_scheme, new_scheme) in SCHEMES.items():
        for variant in VARIANTS:
            print(f"\n{family.upper()} — {variant.upper()} — NEW=SUM_e(E_e^2), OLD=(SUM_e E_e)^2")
            print(f"{'TARGET':24s} {'Q':>2s} {'OLD':>10s} {'NEW':>10s} {'DELTA':>10s}")
            print("-" * 62)
            grouped = {target: ([], [], []) for target in TARGETS}
            for query, record in enumerate(records):
                query_root = result / f"query_{_prompt_tag(str(record['prompt']))}" / f"initial_seed_{int(record['initial_seed'])}"
                for target in TARGETS:
                    old = read_lds(query_root, old_scheme, variant, target, args.prediction_sign)
                    new = read_lds(query_root, new_scheme, variant, target, args.prediction_sign)
                    delta = new - old
                    grouped[target][0].append(old)
                    grouped[target][1].append(new)
                    grouped[target][2].append(delta)
                    print(f"{target:24s} {query:2d} {old:+9.3f}% {new:+9.3f}% {delta:+9.3f}%")
            print("MEAN BY TARGET")
            for target in TARGETS:
                old, new, delta = (statistics.fmean(values) for values in grouped[target])
                print(f"{target:24s}    {old:+9.3f}% {new:+9.3f}% {delta:+9.3f}%")


if __name__ == "__main__":
    main()
