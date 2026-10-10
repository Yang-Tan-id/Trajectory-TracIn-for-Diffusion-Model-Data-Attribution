"""Merge four timestamp shards of trajectory-predicted-noise aligned DAS."""

import argparse
import json

import numpy as np

from trajectory_predicted_noise_aligned_das_config import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp-shard-count", type=int, default=4)
    args = parser.parse_args()
    totals = {
        float(lam): np.zeros(
            (len(TPNA_DAS_QUERY_IDS), N_TRAIN), dtype=np.float64
        )
        for lam in DAS_LAMBDAS
    }
    covered = []
    metadata = []
    for shard_index in range(args.timestamp_shard_count):
        root = tpna_das_shard_root(shard_index, args.timestamp_shard_count)
        with open(root / "done.json") as handle:
            info = json.load(handle)
        metadata.append(info)
        if info["method"] != TPNA_DAS_METHOD:
            raise ValueError(f"method differs in {root}")
        if info["query_ids"] != list(TPNA_DAS_QUERY_IDS):
            raise ValueError(f"query IDs differ in {root}")
        covered.extend(int(value) for value in info["timestamp_indices"])
        for lam in DAS_LAMBDAS:
            values = np.load(root / f"lambda_{lambda_tag(lam)}.npy")
            if values.shape != totals[float(lam)].shape:
                raise ValueError(f"score shape differs in {root}: {values.shape}")
            totals[float(lam)] += values
    if sorted(covered) != list(range(99)):
        raise ValueError("timestamp shards do not cover indices 0..98")
    with open(QUERY_DIR / "manifest.json") as handle:
        by_id = {int(record["query_id"]): record for record in json.load(handle)}
    for query_position, query_id in enumerate(TPNA_DAS_QUERY_IDS):
        for lam, values in totals.items():
            output = (
                ATTR_DIR
                / TPNA_DAS_METHOD
                / f"q{query_id:02d}"
                / f"lambda_{lambda_tag(lam)}"
            )
            output.mkdir(parents=True, exist_ok=True)
            np.save(output / "scores.npy", values[query_position])
            with open(output / "info.json", "w") as handle:
                json.dump(
                    {
                        "query": by_id[query_id],
                        "method": TPNA_DAS_METHOD,
                        "lambda": lam,
                        "parameter_source": "final EMA",
                        "projection_dim": TPNA_DAS_PROJ_DIM,
                        "outer_probe_count": int(DAS_NUM_MC),
                        "included_timestamp_indices": list(range(99)),
                        "endpoint_excluded": True,
                        "term_weight": 1.0 / (99.0 * float(DAS_NUM_MC)),
                        "train_loss_is_query_dependent": True,
                        "alignment_definition": metadata[0][
                            "alignment_definition"
                        ],
                    },
                    handle,
                    indent=2,
                )
    print("[done] merged trajectory-predicted-noise aligned DAS q00-q09")


if __name__ == "__main__":
    main()
