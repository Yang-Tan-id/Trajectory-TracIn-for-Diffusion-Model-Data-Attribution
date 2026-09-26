"""Merge timestamp shards of subset-level unlearning parameter updates."""

import argparse
import os

import torch

from checkpoint_counterfactual_config import CF_UPDATE_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("prompted", "unprompted"), required=True)
    parser.add_argument("--timestamp-shard-count", type=int, required=True)
    args = parser.parse_args()
    merged = None
    metadata = None
    covered = []
    for shard in range(args.timestamp_shard_count):
        path = (
            CF_UPDATE_ROOT / "shards" / args.family
            / f"shard_{shard:02d}_of_{args.timestamp_shard_count:02d}.pt"
        )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if metadata is None:
            metadata = payload
            merged = payload["updates"].to(torch.float32)
        else:
            if payload["parameter_names"] != metadata["parameter_names"]:
                raise ValueError(f"parameter names differ in {path}")
            merged += payload["updates"].to(torch.float32)
        covered.extend(payload["timestamp_indices"])
    if sorted(covered) != list(range(100)):
        raise ValueError("timestamp shards do not cover 0..99")
    output = CF_UPDATE_ROOT / f"{args.family}.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".pt.tmp")
    torch.save(
        {
            **{key: value for key, value in metadata.items() if key != "updates"},
            "updates": merged,
            "timestamp_indices": sorted(covered),
        },
        temporary,
    )
    os.replace(temporary, output)
    print(f"[done] {output} shape={tuple(merged.shape)}", flush=True)


if __name__ == "__main__":
    main()
