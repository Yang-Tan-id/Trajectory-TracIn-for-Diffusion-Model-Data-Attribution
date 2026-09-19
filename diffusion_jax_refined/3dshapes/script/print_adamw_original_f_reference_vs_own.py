#!/usr/bin/env python3
"""Print matched reference- and own-trajectory LDS values side by side."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

KEYS = ("method", "variant", "target", "query")


def load(path: Path) -> dict[tuple[str, str, str, int], float]:
    csv_path = path / "per_query.csv" if path.is_dir() else path
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {
        (row["method"], row["variant"], row["target"], int(row["query"])): float(
            row["lds_percent"]
        )
        for row in rows
    }
    if len(result) != len(rows):
        raise ValueError(f"duplicate comparison keys in {csv_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--own", type=Path, required=True)
    parser.add_argument(
        "--compact-raw-residual",
        action="store_true",
        help="print full AdamW FOUR and history-subtracted FOUR_RESIDUAL for the three requested targets",
    )
    args = parser.parse_args()

    reference = load(args.reference)
    own = load(args.own)
    if reference.keys() != own.keys():
        only_reference = sorted(reference.keys() - own.keys())
        only_own = sorted(own.keys() - reference.keys())
        raise ValueError(
            "reference/own rows do not match: "
            f"only_reference={only_reference[:5]} only_own={only_own[:5]}"
        )

    groups = sorted({key[:3] for key in reference})
    if args.compact_raw_residual:
        groups = [
            group
            for group in groups
            if group[0] in ("four", "four_residual")
            and group[1] == "query_train_l2"
            and group[2]
            in ("endpoint_contarfactual", "traj_contarfactual", "simple_loss")
        ]
    for method, variant, target in groups:
        print(f"\n{method.upper()} — {variant.upper()} — {target}")
        print(f"{'Q':>4} {'REFERENCE':>12} {'OWN':>12}")
        print("-" * 32)
        reference_values = []
        own_values = []
        for query in range(10):
            key = (method, variant, target, query)
            reference_value = reference[key]
            own_value = own[key]
            reference_values.append(reference_value)
            own_values.append(own_value)
            print(f"Q{query:<2d} {reference_value:+11.3f}% {own_value:+11.3f}%")
        print(
            f"MEAN {sum(reference_values) / len(reference_values):+11.3f}% "
            f"{sum(own_values) / len(own_values):+11.3f}%"
        )


if __name__ == "__main__":
    main()
