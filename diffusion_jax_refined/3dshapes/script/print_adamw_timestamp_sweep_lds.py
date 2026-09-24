#!/usr/bin/env python3
"""Print LDS as a function of the 20 aligned diffusion timestamps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT))

from dataset_config import _prompt_tag


TIMESTAMPS = (
    0, 49, 111, 149, 222, 249, 333, 349, 444, 449,
    549, 555, 649, 666, 749, 777, 849, 888, 949, 999,
)
TARGETS = (
    "endpoint_contarfactual",
    "traj_contarfactual",
    "simple_loss",
    "noise_trajectory",
)
VARIANTS = ("raw", "query_l2", "train_l2", "query_train_l2")


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument(
        "--query-file",
        type=Path,
        default=SHAPES_ROOT / "queries_in_distribution_plus_zero_seed_100_219.json",
    )
    parser.add_argument("--query-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--variant", choices=VARIANTS, default="raw")
    parser.add_argument("--kind", choices=("residual", "full", "both"), default="both")
    parser.add_argument("--per-query-target", choices=TARGETS)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()

    records = json.loads(args.query_file.read_text())["queries"]
    query_ids = parse_ints(args.query_ids)
    kinds = ("residual", "full") if args.kind == "both" else (args.kind,)
    rows: list[dict[str, object]] = []
    eval_root = SHAPES_ROOT / "result" / args.experiment / "eval" / "prompted_solo"
    for kind in kinds:
        for timestep in TIMESTAMPS:
            for query_id in query_ids:
                record = records[query_id]
                method = (
                    f"traj_tracin_adamw_{kind}_single_timestamp_t{timestep:03d}_"
                    f"{args.variant}"
                )
                for target in TARGETS:
                    root = (
                        eval_root
                        / f"query_{_prompt_tag(str(record['prompt']))}"
                        / f"initial_seed_{int(record['initial_seed'])}"
                        / "lds"
                        / method
                        / target
                        / "pred_kept_sign_p1"
                    )
                    matches = list(root.glob("*/lds_summary.json"))
                    if len(matches) != 1:
                        raise RuntimeError(
                            f"expected one LDS summary for {kind=} {timestep=} "
                            f"Q{query_id} {target}; found {len(matches)} under {root}"
                        )
                    payload = json.loads(matches[0].read_text())
                    rows.append(
                        {
                            "kind": kind,
                            "variant": args.variant,
                            "timestep": timestep,
                            "query": query_id,
                            "initial_seed": int(record["initial_seed"]),
                            "target": target,
                            "lds_percent": float(payload["lds_percent"]),
                        }
                    )

    output_csv = args.output_csv or (
        SHAPES_ROOT
        / "result"
        / args.experiment
        / "eval"
        / f"adamw_single_timestamp_{args.variant}_per_query.csv"
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for kind in kinds:
        print(f"\n{kind.upper()} — {args.variant.upper()} — MEAN OVER {len(query_ids)} QUERIES")
        print(f"{'t':>4s} {'ENDPOINT':>10s} {'TRAJ-CF':>10s} {'SIMPLE':>10s} {'NOISE':>10s}")
        print("-" * 52)
        for timestep in TIMESTAMPS:
            values = []
            for target in TARGETS:
                selected = [
                    float(row["lds_percent"])
                    for row in rows
                    if row["kind"] == kind
                    and row["timestep"] == timestep
                    and row["target"] == target
                ]
                values.append(statistics.fmean(selected))
            print(f"{timestep:4d} " + " ".join(f"{value:+9.3f}%" for value in values))

        if args.per_query_target:
            print(f"\n{kind.upper()} — {args.variant.upper()} — {args.per_query_target}")
            print(f"{'t':>4s} " + " ".join(f"Q{query_id:<8d}" for query_id in query_ids) + " MEAN")
            print("-" * (6 + 10 * len(query_ids) + 10))
            for timestep in TIMESTAMPS:
                values = [
                    next(
                        float(row["lds_percent"])
                        for row in rows
                        if row["kind"] == kind
                        and row["timestep"] == timestep
                        and row["query"] == query_id
                        and row["target"] == args.per_query_target
                    )
                    for query_id in query_ids
                ]
                print(
                    f"{timestep:4d} "
                    + " ".join(f"{value:+9.3f}%" for value in values)
                    + f" {statistics.fmean(values):+9.3f}%"
                )
    print(f"\nSaved per-query LDS: {output_csv}")


if __name__ == "__main__":
    main()
