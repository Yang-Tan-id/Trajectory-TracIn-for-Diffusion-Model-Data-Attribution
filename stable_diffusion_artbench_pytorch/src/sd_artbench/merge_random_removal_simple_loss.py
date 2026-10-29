"""Merge ArtBench simple-loss shards and produce Q0-Q9 reports."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from .analyze_random_removal_simple_loss_correlation import (
    FIELDS,
    completed_pairs,
    parse_csv_ints,
    summarize,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--query-indices", type=parse_csv_ints, default=list(range(10)))
    parser.add_argument("--model-indices", type=parse_csv_ints, default=list(range(60)))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or (args.root / "combined_q0_9_m0_59")
    output_dir.mkdir(parents=True, exist_ok=True)

    lookup: dict[tuple[int, int, int], dict[str, Any]] = {}
    for path in sorted(args.root.glob("shard_*/simple_loss_per_query_model_timestep.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (
                    int(row["evaluation_query_index"]),
                    int(row["random_model_index"]), int(row["step"]),
                )
                if key[0] not in args.query_indices or key[1] not in args.model_indices:
                    continue
                previous = lookup.get(key)
                if previous is not None and previous != row:
                    raise RuntimeError(f"conflicting duplicate {key}")
                lookup[key] = row
    if not lookup:
        raise SystemExit(f"no shard simple-loss CSV files below {args.root}")
    rows = list(lookup.values())
    expected_pairs = {
        (query, model) for query in args.query_indices for model in args.model_indices
    }
    missing = sorted(expected_pairs - completed_pairs(rows, args.steps))
    if missing:
        raise RuntimeError(f"incomplete query-model pairs: {missing}")
    timesteps = []
    for step in range(args.steps):
        values = {
            int(row["timestep"]) for row in rows if int(row["step"]) == step
        }
        if len(values) != 1:
            raise RuntimeError(f"step {step} has timestep values {sorted(values)}")
        timesteps.append(values.pop())

    combined = output_dir / "simple_loss_per_query_model_timestep.csv"
    with combined.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(
            rows, key=lambda row: (
                int(row["evaluation_query_index"]),
                int(row["random_model_index"]), int(row["step"]),
            )
        ))
    summarize(output_dir, rows, args.query_indices, args.model_indices, timesteps)
    print((output_dir / "simple_loss_correlation_report.txt").read_text())
    print(f"[saved] {output_dir}")


if __name__ == "__main__":
    main()
