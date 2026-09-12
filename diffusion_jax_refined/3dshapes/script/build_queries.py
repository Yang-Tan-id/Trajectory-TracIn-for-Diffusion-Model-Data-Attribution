#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


LABELS = (
    tuple(f"shape_{name}" for name in ("cube", "cylinder", "sphere", "capsule"))
    + tuple(f"object_hue_{i}" for i in range(10))
    + tuple(f"wall_hue_{i}" for i in range(10))
    + tuple(f"floor_hue_{i}" for i in range(10))
)


def queries(seeds: range = range(10), labels_per_query: int = 4) -> list[dict[str, object]]:
    records = []
    for seed in seeds:
        picked = np.random.default_rng(seed).choice(LABELS, size=labels_per_query, replace=False).tolist()
        records.append({"query_seed": seed, "initial_seed": seed, "labels": picked, "prompt": ",".join(picked)})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the ten deterministic unordered 3D Shapes queries.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "queries_seed_0_9.json",
    )
    args = parser.parse_args()
    payload = {
        "format_version": 1,
        "selection": "four unique tokens uniformly sampled from all 34 non-scale/non-orientation tokens",
        "order_semantics": "none; prompts are converted to multi-hot vectors",
        "queries": queries(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"saved {len(payload['queries'])} queries to {args.output}")


if __name__ == "__main__":
    main()
