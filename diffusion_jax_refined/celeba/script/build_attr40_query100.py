#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from prepare_celeba import read_attributes, read_partition


def parse_ints(text: str) -> list[int]:
    values = [int(value) for value in text.replace(",", " ").split() if value]
    if not values or len(set(values)) != len(values):
        raise ValueError("--initial-seeds must contain distinct integers")
    return values


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Select 25 CelebA validation labels and expand each over four initial seeds."
    )
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--attributes", type=Path, required=True)
    parser.add_argument("--num-label-points", type=int, default=25)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--initial-seeds", default="0,1,2,3")
    parser.add_argument("--min-positive", type=int, default=4)
    parser.add_argument("--max-positive", type=int, default=14)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "queries_attr40_val25_x4.json",
    )
    args = parser.parse_args()
    initial_seeds = parse_ints(args.initial_seeds)
    if args.num_label_points <= 0:
        parser.error("--num-label-points must be positive")

    partitions = read_partition(args.partition)
    names, attributes = read_attributes(args.attributes)
    blurry_index = names.index("Blurry")
    candidates = []
    for filename in sorted(attributes):
        if partitions.get(filename) != 1:
            continue
        raw = attributes[filename]
        labels = (raw > 0).astype(np.uint8)
        positives = int(labels.sum())
        if labels[blurry_index] or not args.min_positive <= positives <= args.max_positive:
            continue
        candidates.append((filename, labels))
    if args.num_label_points > len(candidates):
        raise ValueError(
            f"requested {args.num_label_points} points but only {len(candidates)} are eligible"
        )

    rng = np.random.default_rng(args.selection_seed)
    selected_positions = rng.choice(
        len(candidates), size=args.num_label_points, replace=False
    )
    queries = []
    for prompt_id, position in enumerate(selected_positions.tolist()):
        filename, labels = candidates[position]
        positive_names = [name for name, value in zip(names, labels.tolist()) if value]
        prompt = ",".join(positive_names)
        for initial_seed in initial_seeds:
            queries.append(
                {
                    "query_id": len(queries),
                    "prompt_id": prompt_id,
                    "source_partition": 1,
                    "source_filename": filename,
                    "selection_seed": args.selection_seed,
                    "initial_seed": initial_seed,
                    "labels": labels.astype(int).tolist(),
                    "positive_attributes": positive_names,
                    "prompt": prompt,
                }
            )

    payload = {
        "format_version": 1,
        "selection": (
            "uniform without replacement from official validation split after "
            f"Blurry=0 and {args.min_positive}<=positive_count<={args.max_positive}"
        ),
        "selection_seed": args.selection_seed,
        "attribute_names": list(names),
        "num_label_points": args.num_label_points,
        "initial_seeds": initial_seeds,
        "queries": queries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "query_id", "prompt_id", "source_partition", "source_filename",
                "selection_seed", "initial_seed", "positive_count", "prompt",
                "label_bits",
            ),
        )
        writer.writeheader()
        for record in queries:
            writer.writerow(
                {
                    **{key: record[key] for key in writer.fieldnames if key in record},
                    "positive_count": len(record["positive_attributes"]),
                    "label_bits": "".join(str(value) for value in record["labels"]),
                }
            )
    print(f"saved {len(queries)} queries to {args.output}")
    print(f"saved query manifest CSV to {csv_path}")


if __name__ == "__main__":
    main()
