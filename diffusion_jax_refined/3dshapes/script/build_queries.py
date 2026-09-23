#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


LABEL_GROUPS = (
    tuple(f"shape_{name}" for name in ("cube", "cylinder", "sphere", "capsule")),
    tuple(f"object_hue_{i}" for i in range(10)),
    tuple(f"wall_hue_{i}" for i in range(10)),
    tuple(f"floor_hue_{i}" for i in range(10)),
)
LABELS = tuple(label for group in LABEL_GROUPS for label in group)
ZERO_CONDITION_PROMPT = "__zero_condition__"


def queries(
    seeds: range = range(10),
    labels_per_query: int = 4,
    selection: str = "uniform_tokens",
) -> list[dict[str, object]]:
    if selection not in {"uniform_tokens", "one_per_category"}:
        raise ValueError(f"unsupported query selection: {selection}")
    if selection == "one_per_category" and labels_per_query != len(LABEL_GROUPS):
        raise ValueError("one_per_category requires exactly four labels per query")
    records = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        if selection == "one_per_category":
            picked = [str(rng.choice(group)) for group in LABEL_GROUPS]
        else:
            picked = rng.choice(LABELS, size=labels_per_query, replace=False).tolist()
        records.append({"query_seed": seed, "initial_seed": seed, "labels": picked, "prompt": ",".join(picked)})
    return records


def zero_condition_queries(seeds: range) -> list[dict[str, object]]:
    return [
        {
            "query_seed": seed,
            "initial_seed": seed,
            "labels": [],
            "prompt": ZERO_CONDITION_PROMPT,
            "conditioning": "zero_multi_hot",
        }
        for seed in seeds
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic unordered 3D Shapes queries.")
    parser.add_argument("--num-queries", type=int, default=10)
    parser.add_argument("--num-zero-queries", type=int, default=0)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--selection",
        choices=("uniform_tokens", "one_per_category"),
        default="uniform_tokens",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "queries_seed_0_9.json",
    )
    args = parser.parse_args()
    if args.num_queries <= 0:
        parser.error("--num-queries must be positive")
    if args.num_zero_queries < 0:
        parser.error("--num-zero-queries must be nonnegative")
    if args.seed_start < 0:
        parser.error("--seed-start must be nonnegative")
    selection_description = {
        "uniform_tokens": "four unique tokens uniformly sampled from all 34 non-scale/non-orientation tokens",
        "one_per_category": "one token from each category: shape, object hue, wall hue, and floor hue",
    }[args.selection]
    records = queries(
        range(args.seed_start, args.seed_start + args.num_queries),
        selection=args.selection,
    )
    records.extend(
        zero_condition_queries(
            range(
                args.seed_start + args.num_queries,
                args.seed_start + args.num_queries + args.num_zero_queries,
            )
        )
    )
    payload = {
        "format_version": 1,
        "selection": selection_description,
        "zero_condition_queries": args.num_zero_queries,
        "order_semantics": "none; prompts are converted to multi-hot vectors",
        "queries": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"saved {len(payload['queries'])} queries to {args.output}")


if __name__ == "__main__":
    main()
