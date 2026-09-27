"""Merge timestamp-sharded projected Traj scores without approximation."""

import argparse
import json
import re

import numpy as np

from exp_config import ATTR_DIR, QUERY_DIR, TRACIN_CONTRACTIONS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("prompted", "unprompted"), required=True)
    parser.add_argument("--timestamp-shard-count", type=int, required=True)
    parser.add_argument(
        "--checkpoint-direction",
        choices=("forward", "backward"),
        default="forward",
    )
    parser.add_argument("--checkpoint-pair-indices")
    parser.add_argument("--first-order-only", action="store_true")
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()
    if args.output_suffix and not re.fullmatch(
        r"[A-Za-z0-9_]+", args.output_suffix
    ):
        raise ValueError("--output-suffix may contain only letters, digits, and underscores")
    expected_pair_indices = (
        [
            int(item.strip())
            for item in args.checkpoint_pair_indices.split(",")
            if item.strip()
        ]
        if args.checkpoint_pair_indices is not None
        else None
    )

    with open(QUERY_DIR / "manifest.json") as handle:
        records = [r for r in json.load(handle) if r["family"] == args.family]
    shard_namespace = (
        "_projected_traj_shards"
        if args.checkpoint_direction == "forward"
        else "_projected_backward_traj_shards"
    )
    if args.output_suffix:
        shard_namespace += f"_{args.output_suffix}"
    root = ATTR_DIR / shard_namespace / args.family
    orders = (
        ("first",)
        if args.first_order_only or args.checkpoint_direction == "backward"
        else ("first", "second")
    )
    arrays = {}
    covered_timestamps = []
    for shard_i in range(args.timestamp_shard_count):
        shard = root / f"shard_{shard_i:02d}_of_{args.timestamp_shard_count:02d}"
        with open(shard / "done.json") as handle:
            metadata = json.load(handle)
        expected_ids = [int(record["query_id"]) for record in records]
        if metadata["query_ids"] != expected_ids:
            raise ValueError(f"query IDs mismatch in {shard}")
        if metadata.get("checkpoint_direction", "forward") != args.checkpoint_direction:
            raise ValueError(f"checkpoint direction mismatch in {shard}")
        if metadata.get("first_order_only", False) != args.first_order_only:
            raise ValueError(f"first-order mode mismatch in {shard}")
        if metadata.get("output_suffix", "") != args.output_suffix:
            raise ValueError(f"output suffix mismatch in {shard}")
        if expected_pair_indices is not None and metadata.get(
            "checkpoint_pair_indices"
        ) != expected_pair_indices:
            raise ValueError(f"checkpoint pair indices mismatch in {shard}")
        covered_timestamps.extend(int(value) for value in metadata["timestamp_indices"])
        for order in orders:
            for contraction in TRACIN_CONTRACTIONS:
                key = (order, contraction)
                value = np.load(shard / f"{order}_{contraction}.npy")
                arrays[key] = value if key not in arrays else arrays[key] + value
    if sorted(covered_timestamps) != list(range(100)):
        raise ValueError(f"timestamp shards do not cover 0..99 exactly: {covered_timestamps}")

    for qi, record in enumerate(records):
        for (order, contraction), values in arrays.items():
            prefix = (
                "traj_projected"
                if args.checkpoint_direction == "forward"
                else "traj_projected_backward"
            )
            suffix = f"_{args.output_suffix}" if args.output_suffix else ""
            method = f"{prefix}_{order}_raw_{contraction}{suffix}"
            out = ATTR_DIR / method / f"q{int(record['query_id']):02d}"
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / "scores.npy", values[qi])
            with open(out / "info.json", "w") as handle:
                json.dump(
                    {
                        "query": record,
                        "order": order,
                        "target": (
                            "next" if args.checkpoint_direction == "forward" else "previous"
                        ),
                        "checkpoint_direction": args.checkpoint_direction,
                        "checkpoint_pair_indices": metadata.get(
                            "checkpoint_pair_indices"
                        ),
                        "checkpoint_pairs": metadata.get("checkpoint_pairs"),
                        "first_order_only": args.first_order_only,
                        "output_suffix": args.output_suffix,
                        "param_source": "raw",
                        "projection": "countsketch",
                        "contraction": contraction,
                        "timestamp_shards": args.timestamp_shard_count,
                        "learning_rate_source": (
                            "current_checkpoint"
                            if args.checkpoint_direction == "forward"
                            else "previous_checkpoint"
                        ),
                        "merged_timestamp_indices": sorted(covered_timestamps),
                    },
                    handle,
                    indent=2,
                )
    print(
        f"[done] merged projected Traj {args.checkpoint_direction} "
        f"{args.family} from {args.timestamp_shard_count} shards"
    )


if __name__ == "__main__":
    main()
