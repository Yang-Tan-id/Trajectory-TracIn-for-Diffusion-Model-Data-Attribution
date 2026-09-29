"""Merge higher-noise-aligned fully-unrolled trajectory DAS shards."""

import argparse
import json

import numpy as np

from unrolled_traj_das_config import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-count", type=int, default=4)
    args = parser.parse_args()
    totals = {
        float(lam): np.zeros(
            (len(UNROLLED_TRAJ_DAS_QUERY_IDS), N_TRAIN), dtype=np.float64
        )
        for lam in DAS_LAMBDAS
    }
    covered = []
    metadata = []
    for shard_index in range(args.shard_count):
        root = unrolled_traj_das_shard_root(shard_index, args.shard_count)
        with open(root / "done.json") as handle:
            info = json.load(handle)
        metadata.append(info)
        if info["method"] != UNROLLED_TRAJ_DAS_METHOD:
            raise ValueError(f"method differs in {root}")
        if info["query_ids"] != list(UNROLLED_TRAJ_DAS_QUERY_IDS):
            raise ValueError(f"query IDs differ in {root}")
        covered.extend(int(value) for value in info["selected_term_indices"])
        for lam in DAS_LAMBDAS:
            values = np.load(root / f"lambda_{lambda_tag(lam)}.npy")
            if values.shape != totals[float(lam)].shape:
                raise ValueError(f"score shape differs in {root}: {values.shape}")
            totals[float(lam)] += values
    expected_terms = len(DAS_TIMESTEPS) * int(DAS_NUM_MC)
    if sorted(covered) != list(range(expected_terms)):
        raise ValueError("shards do not cover all timestamp/MC terms exactly")
    trajectory_timesteps = metadata[0]["trajectory_timesteps"]
    higher_noise_counts = metadata[0][
        "higher_noise_training_timestamp_counts"
    ]
    for info in metadata[1:]:
        if info["trajectory_timesteps"] != trajectory_timesteps:
            raise ValueError("trajectory timesteps differ across shards")
        if (
            info["higher_noise_training_timestamp_counts"]
            != higher_noise_counts
        ):
            raise ValueError("higher-noise timestamp counts differ across shards")

    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    by_id = {int(record["query_id"]): record for record in manifest}
    for query_position, query_id in enumerate(UNROLLED_TRAJ_DAS_QUERY_IDS):
        for lam, values in totals.items():
            output = (
                ATTR_DIR
                / UNROLLED_TRAJ_DAS_METHOD
                / f"q{query_id:02d}"
                / f"lambda_{lambda_tag(lam)}"
            )
            output.mkdir(parents=True, exist_ok=True)
            np.save(output / "scores.npy", values[query_position])
            with open(output / "info.json", "w") as handle:
                json.dump(
                    {
                        "query": by_id[query_id],
                        "method": UNROLLED_TRAJ_DAS_METHOD,
                        "lambda": lam,
                        "parameter_source": "final EMA",
                        "trajectory_definition": (
                            "fully differentiable deterministic DDIM unroll from "
                            "the cached initial noise"
                        ),
                        "trajectory_snapshots": int(TRAJ_SNAPSHOTS),
                        "query_output_dimension": int(
                            UNROLLED_TRAJ_DAS_OUTPUT_DIM
                        ),
                        "shared_output_probe": True,
                        "output_probe_role": (
                            "the same Gaussian output probe is used for the "
                            "training gradient, residual, and aligned query VJP"
                        ),
                        "trajectory_response": (
                            "per-state shared-probe squared response; each state t is "
                            "averaged over training-loss timestamps s>=t, then "
                            "the 100 state responses are averaged"
                        ),
                        "trajectory_timesteps": trajectory_timesteps,
                        "higher_noise_training_timestamp_counts": (
                            higher_noise_counts
                        ),
                        "timestamp_alignment": "training loss timestamp s >= trajectory state timestamp t",
                        "das_timestamp_count": len(DAS_TIMESTEPS),
                        "das_outer_mc": int(DAS_NUM_MC),
                        "train_gradient_mc": int(DAS_TRAIN_GRAD_MC),
                        "projection_dim": int(UNROLLED_TRAJ_DAS_PROJECTION_DIM),
                        "global_projection": True,
                        "normalize_train_features": bool(
                            DAS_NORMALIZE_PROJECTED_GRADS
                        ),
                        "normalize_query_features": False,
                        "shard_count": args.shard_count,
                    },
                    handle,
                    indent=2,
                )
    print(f"[saved] {UNROLLED_TRAJ_DAS_METHOD} q00-q09 all lambdas", flush=True)


if __name__ == "__main__":
    main()
