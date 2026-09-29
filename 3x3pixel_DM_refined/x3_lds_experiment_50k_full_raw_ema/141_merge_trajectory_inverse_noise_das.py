"""Merge query-dependent trajectory inverse-noise DAS shards."""

import argparse
import json

import numpy as np

from trajectory_inverse_noise_das_config import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp-shard-count", type=int, default=4)
    args = parser.parse_args()
    totals = {
        float(lam): np.zeros(
            (len(TRAJECTORY_INVERSE_DAS_QUERY_IDS), N_TRAIN), dtype=np.float64
        )
        for lam in DAS_LAMBDAS
    }
    covered = []
    metadata = []
    for shard_index in range(args.timestamp_shard_count):
        root = trajectory_inverse_das_shard_root(
            shard_index, args.timestamp_shard_count
        )
        with open(root / "done.json") as handle:
            info = json.load(handle)
        metadata.append(info)
        if info["method"] != TRAJECTORY_INVERSE_DAS_METHOD:
            raise ValueError(f"method differs in {root}")
        if info["query_ids"] != list(TRAJECTORY_INVERSE_DAS_QUERY_IDS):
            raise ValueError(f"query IDs differ in {root}")
        covered.extend(int(value) for value in info["timestamp_indices"])
        for lam in DAS_LAMBDAS:
            value = np.load(root / f"lambda_{lambda_tag(lam)}.npy")
            if value.shape != totals[float(lam)].shape:
                raise ValueError(f"score shape differs in {root}: {value.shape}")
            totals[float(lam)] += value
    if sorted(covered) != list(range(99)):
        raise ValueError(f"timestamp shards do not cover indices 0..98: {covered}")

    with open(QUERY_DIR / "manifest.json") as handle:
        by_id = {int(record["query_id"]): record for record in json.load(handle)}
    for query_position, query_id in enumerate(TRAJECTORY_INVERSE_DAS_QUERY_IDS):
        for lam, values in totals.items():
            output = (
                ATTR_DIR
                / TRAJECTORY_INVERSE_DAS_METHOD
                / f"q{query_id:02d}"
                / f"lambda_{lambda_tag(lam)}"
            )
            output.mkdir(parents=True, exist_ok=True)
            np.save(output / "scores.npy", values[query_position])
            with open(output / "info.json", "w") as handle:
                json.dump(
                    {
                        "query": by_id[query_id],
                        "method": TRAJECTORY_INVERSE_DAS_METHOD,
                        "lambda": lam,
                        "parameter_source": "final EMA",
                        "projection_dim": TRAJECTORY_INVERSE_DAS_PROJ_DIM,
                        "outer_probe_count": int(DAS_NUM_MC),
                        "included_timestamp_indices": list(range(99)),
                        "endpoint_excluded": True,
                        "term_weight": 1.0 / (99.0 * float(DAS_NUM_MC)),
                        "train_loss_is_query_dependent": True,
                        "loss_definition": metadata[0]["loss_definition"],
                    },
                    handle,
                    indent=2,
                )
    print("[done] merged trajectory inverse-noise DAS q00-q09")


if __name__ == "__main__":
    main()
