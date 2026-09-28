"""Merge reference-forward-10 aligned-loss timestamp shards."""

import argparse
import json

import numpy as np

from reference_forward10_config import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--query-scope", choices=("ten", "all"), default="all")
    parser.add_argument("--timestamp-shard-count", type=int, default=2)
    args = parser.parse_args()
    if args.query_scope == "ten":
        if args.family != "prompted":
            raise ValueError("ten-query scope is q00-q09 prompted only")
        query_ids = list(range(10))
    else:
        query_ids = (
            list(range(75))
            if args.family == "prompted"
            else list(range(75, 100))
        )
    totals = {
        name: np.zeros((len(query_ids), N_TRAIN), dtype=np.float64)
        for name in REF_FORWARD10_METHODS
    }
    covered = []
    for shard_index in range(args.timestamp_shard_count):
        root = ref_forward10_shard_root(
            args.family,
            shard_index,
            args.timestamp_shard_count,
            args.query_scope,
        )
        with open(root / "done.json") as handle:
            info = json.load(handle)
        if info["query_ids"] != query_ids or info["family"] != args.family:
            raise ValueError(f"query/family mismatch in {root}")
        if info.get("query_scope", "all") != args.query_scope:
            raise ValueError(f"query-scope mismatch in {root}")
        covered.extend(int(value) for value in info["timestamp_indices"])
        for contraction in REF_FORWARD10_METHODS:
            values = np.load(root / f"{contraction}.npy")
            if values.shape != totals[contraction].shape:
                raise ValueError(f"shape mismatch in {root}/{contraction}.npy")
            totals[contraction] += values
    if sorted(covered) != list(range(TRAJ_SNAPSHOTS)):
        raise ValueError("timestamp shards do not cover 0..99 exactly")

    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    by_id = {int(record["query_id"]): record for record in manifest}
    for contraction, method in REF_FORWARD10_METHODS.items():
        for position, query_id in enumerate(query_ids):
            output = ATTR_DIR / method / f"q{query_id:02d}"
            output.mkdir(parents=True, exist_ok=True)
            np.save(output / "scores.npy", totals[contraction][position])
            with open(output / "info.json", "w") as handle:
                json.dump(
                    {
                        "query": by_id[query_id],
                        "query_scope": args.query_scope,
                        "method": method,
                        "contraction": contraction,
                        "checkpoint_count": 50,
                        "checkpoint_transition_count": 49,
                        "reference_timestamp_count": 100,
                        "delta_t": REF_FORWARD10_DELTA_T,
                        "loss_timestep": "reference timestep + 10",
                        "maximum_zero_based_timestep": T - 1 + REF_FORWARD10_DELTA_T,
                        "maximum_one_based_timestep": T + REF_FORWARD10_DELTA_T,
                        "query_train_noise_aligned": True,
                        "query_scalar": (
                            "dot(epsilon_current, normalize(epsilon_next-"
                            "epsilon_current)) at the same reference-forward-10 state"
                        ),
                        "checkpoint_target": "next",
                        "query_uses_loss": False,
                        "train_uses_diffusion_loss": True,
                        "parameter_source": "raw",
                        "query_state_source": "cached final-EMA reference trajectory",
                        "projection": "countsketch",
                        "projection_dim": TRACIN_PROJ_DIM,
                        "lr_weighted": TRACIN_USE_LR_WEIGHTS,
                    },
                    handle,
                    indent=2,
                )
        print(f"[saved] {method} q{query_ids[0]:02d}-q{query_ids[-1]:02d}")


if __name__ == "__main__":
    main()
