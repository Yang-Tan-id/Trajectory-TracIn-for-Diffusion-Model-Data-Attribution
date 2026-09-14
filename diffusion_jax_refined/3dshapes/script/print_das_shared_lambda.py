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


TARGETS = ("endpoint_contarfactual", "traj_contarfactual")


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value.strip()]


def display_target(target: str) -> str:
    return target.replace("contarfactual", "counterfactual")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select one DAS lambda shared by endpoint and trajectory counterfactual "
            "using their joint mean LDS, then print every query."
        )
    )
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--query-ids", default="0,1,2,3,4,5,6,7,8,9")
    args = parser.parse_args()

    records = json.loads((SHAPES_ROOT / "queries_seed_0_9.json").read_text())["queries"]
    query_ids = parse_ints(args.query_ids)
    result_root = SHAPES_ROOT / "result" / args.experiment
    values: dict[float, dict[str, dict[int, float]]] = {}

    for query_id in query_ids:
        if query_id < 0 or query_id >= len(records):
            raise ValueError(f"query id {query_id} is outside [0, {len(records) - 1}]")
        record = records[query_id]
        prompt = str(record["prompt"])
        seed = int(record["initial_seed"])
        lds_root = (
            result_root
            / "eval"
            / "prompted_solo"
            / f"query_{_prompt_tag(prompt)}"
            / f"initial_seed_{seed}"
            / "lds"
        )
        for target in TARGETS:
            pattern = f"das_lambda_*/{target}/pred_kept_sign_m1/*/lds_summary.json"
            for summary_path in lds_root.glob(pattern):
                payload = json.loads(summary_path.read_text())
                damping = float(payload["damping"])
                lds = float(payload["lds_percent"])
                values.setdefault(damping, {}).setdefault(target, {})[query_id] = lds

    complete: dict[float, dict[str, list[float]]] = {}
    for damping, target_values in values.items():
        if all(
            target in target_values and all(query_id in target_values[target] for query_id in query_ids)
            for target in TARGETS
        ):
            complete[damping] = {
                target: [target_values[target][query_id] for query_id in query_ids]
                for target in TARGETS
            }
    if not complete:
        raise RuntimeError(
            "No lambda has complete endpoint and trajectory LDS summaries for all selected queries."
        )

    def joint_mean(damping: float) -> float:
        numbers = [value for target in TARGETS for value in complete[damping][target]]
        return statistics.fmean(numbers)

    usable = [damping for damping in complete if not math.isnan(joint_mean(damping))]
    if not usable:
        raise RuntimeError("Every complete shared-lambda candidate has a NaN joint mean.")
    best = max(usable, key=lambda damping: (joint_mean(damping), -damping))

    print("=" * 110)
    print("DAS SHARED LAMBDA: endpoint_counterfactual + traj_counterfactual")
    print(f"BEST SHARED LAMBDA : {best:g}")
    print(f"JOINT MEAN LDS     : {joint_mean(best):.4f}%  (20 query-target values)")
    print("=" * 110)

    for target in TARGETS:
        target_numbers = complete[best][target]
        print()
        print(f"TARGET      : {display_target(target)}")
        print(f"SHARED LAMBDA: {best:g}")
        print(f"MEAN LDS    : {statistics.fmean(target_numbers):.4f}%")
        print(f"STD LDS     : {statistics.pstdev(target_numbers):.4f}%")
        print("-" * 110)
        print(f"{'Query':<7} {'Seed':<7} {'LDS':>12}  Prompt")
        print("-" * 110)
        for query_id, lds in zip(query_ids, target_numbers):
            record = records[query_id]
            print(
                f"{query_id:<7} {int(record['initial_seed']):<7} "
                f"{lds:>11.4f}%  {record['prompt']}"
            )


if __name__ == "__main__":
    main()
