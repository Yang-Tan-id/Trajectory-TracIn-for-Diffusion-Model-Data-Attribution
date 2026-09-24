#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


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
SCHEME_GROUPS = {
    "original": (
        ("residual", "traj_tracin_adamw_residual_aligned10x10"),
        ("full", "traj_tracin_adamw_full_aligned10x10"),
    ),
    "addon": (
        ("addon_residual", "traj_tracin_adamw_residual_aligned10x10_addon10"),
        ("addon_full", "traj_tracin_adamw_full_aligned10x10_addon10"),
    ),
    "combined": (
        ("combined_residual", "traj_tracin_adamw_residual_aligned20x10_combined"),
        ("combined_full", "traj_tracin_adamw_full_aligned20x10_combined"),
    ),
    "halves": (
        ("early_high_t_residual", "traj_tracin_adamw_residual_aligned10x10_early_high_t"),
        ("early_high_t_full", "traj_tracin_adamw_full_aligned10x10_early_high_t"),
        ("late_low_t_residual", "traj_tracin_adamw_residual_aligned10x10_late_low_t"),
        ("late_low_t_full", "traj_tracin_adamw_full_aligned10x10_late_low_t"),
    ),
    "endpoint_linear": (
        (
            "endpoint_linear_residual",
            "traj_tracin_adamw_residual_aligned20x10_endpoint_linear",
        ),
        (
            "endpoint_linear_full",
            "traj_tracin_adamw_full_aligned20x10_endpoint_linear",
        ),
    ),
    "corrected": (
        (
            "addon_residual_corrected",
            "traj_tracin_adamw_residual_aligned10x10_addon10_corrected",
        ),
        (
            "addon_full_corrected",
            "traj_tracin_adamw_full_aligned10x10_addon10_corrected",
        ),
        (
            "combined_residual_corrected",
            "traj_tracin_adamw_residual_aligned20x10_combined_corrected",
        ),
        (
            "combined_full_corrected",
            "traj_tracin_adamw_full_aligned20x10_combined_corrected",
        ),
    ),
}
VARIANTS = ("raw", "query_l2", "train_l2", "query_train_l2")


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print per-query residual/full aligned10x10 LDS as four-target tables."
    )
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument(
        "--query-file",
        type=Path,
        default=SHAPES_ROOT / "queries_seed_0_9.json",
    )
    parser.add_argument("--query-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument(
        "--scheme-group",
        choices=(
            "original",
            "addon",
            "combined",
            "addon_combined",
            "halves",
            "endpoint_linear",
            "corrected",
            "all",
        ),
        default="original",
    )
    args = parser.parse_args()

    records = json.loads(args.query_file.read_text())["queries"]
    query_ids = parse_ints(args.query_ids)
    if args.scheme_group == "addon_combined":
        schemes = SCHEME_GROUPS["addon"] + SCHEME_GROUPS["combined"]
    elif args.scheme_group == "all":
        schemes = (
            SCHEME_GROUPS["original"]
            + SCHEME_GROUPS["addon"]
            + SCHEME_GROUPS["combined"]
            + SCHEME_GROUPS["halves"]
            + SCHEME_GROUPS["endpoint_linear"]
            + SCHEME_GROUPS["corrected"]
        )
    else:
        schemes = SCHEME_GROUPS[args.scheme_group]
    result_root = SHAPES_ROOT / "result" / args.experiment
    values: dict[tuple[str, str, int, str], float] = {}

    for query_id in query_ids:
        if query_id < 0 or query_id >= len(records):
            raise ValueError(f"query id {query_id} is outside the manifest")
        record = records[query_id]
        lds_root = (
            result_root
            / "eval"
            / "prompted_solo"
            / f"query_{_prompt_tag(str(record['prompt']))}"
            / f"initial_seed_{int(record['initial_seed'])}"
            / "lds"
        )
        for scheme, base in schemes:
            for variant in VARIANTS:
                for target in TARGETS:
                    target_root = (
                        lds_root
                        / f"{base}_{variant}"
                        / target
                        / "pred_kept_sign_p1"
                    )
                    matches = list(target_root.glob("*/lds_summary.json"))
                    if len(matches) != 1:
                        raise RuntimeError(
                            f"Expected one result for Q{query_id} {scheme}/{variant}/{target}; "
                            f"found {len(matches)} under {target_root}"
                        )
                    payload = json.loads(matches[0].read_text())
                    values[(scheme, variant, query_id, target)] = float(
                        payload["lds_percent"]
                    )

    for scheme, _ in schemes:
        for variant in VARIANTS:
            print()
            print(f"{scheme.upper()} — {variant.upper()}")
            print(
                f"{'QUERY':6s} {'ENDPOINT':>10s} {'TRAJ-CF':>10s} "
                f"{'SIMPLE':>10s} {'NOISE':>10s}"
            )
            print("-" * 54)
            columns: dict[str, list[float]] = {target: [] for target in TARGETS}
            for query_id in query_ids:
                row = [values[(scheme, variant, query_id, target)] for target in TARGETS]
                for target, value in zip(TARGETS, row):
                    columns[target].append(value)
                print(
                    f"Q{query_id:<5d} "
                    + " ".join(f"{value:+9.3f}%" for value in row)
                )
            means = [statistics.fmean(columns[target]) for target in TARGETS]
            print("MEAN   " + " ".join(f"{value:+9.3f}%" for value in means))


if __name__ == "__main__":
    main()
