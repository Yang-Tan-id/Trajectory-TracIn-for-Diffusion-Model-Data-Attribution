"""Merge four balanced family/query shards for 100-query inverse-noise DAS."""

import json

import numpy as np

from trajectory_inverse_noise_das_config import *


ASSIGNMENTS = (
    ("prompted", 0, 3),
    ("prompted", 1, 3),
    ("prompted", 2, 3),
    ("unprompted", 0, 1),
)


def main():
    totals = {
        float(lam): np.empty((100, N_TRAIN), dtype=np.float64)
        for lam in DAS_LAMBDAS
    }
    covered = []
    metadata_by_query = {}
    for family, shard_index, shard_count in ASSIGNMENTS:
        root = trajectory_inverse_das_100q_shard_root(
            family, shard_index, shard_count
        )
        with open(root / "done.json") as handle:
            info = json.load(handle)
        if info["method"] != TRAJECTORY_INVERSE_DAS_100Q_METHOD:
            raise ValueError(f"method differs in {root}")
        query_ids = [int(value) for value in info["query_ids"]]
        expected_ids = list(
            trajectory_inverse_das_family_query_ids(family)[
                shard_index::shard_count
            ]
        )
        if query_ids != expected_ids:
            raise ValueError(f"query IDs differ in {root}: {query_ids}")
        covered.extend(query_ids)
        for query_id in query_ids:
            metadata_by_query[query_id] = info
        for lam in DAS_LAMBDAS:
            values = np.load(root / f"lambda_{lambda_tag(lam)}.npy")
            if values.shape != (len(query_ids), N_TRAIN):
                raise ValueError(f"score shape differs in {root}: {values.shape}")
            totals[float(lam)][query_ids] = values
    if sorted(covered) != list(TRAJECTORY_INVERSE_DAS_ALL_QUERY_IDS):
        raise ValueError(f"query shards do not cover q00-q99 exactly: {covered}")

    with open(QUERY_DIR / "manifest.json") as handle:
        by_id = {int(record["query_id"]): record for record in json.load(handle)}
    for query_id in TRAJECTORY_INVERSE_DAS_ALL_QUERY_IDS:
        info = metadata_by_query[query_id]
        for lam, values in totals.items():
            output = (
                ATTR_DIR
                / TRAJECTORY_INVERSE_DAS_100Q_METHOD
                / f"q{query_id:02d}"
                / f"lambda_{lambda_tag(lam)}"
            )
            output.mkdir(parents=True, exist_ok=True)
            np.save(output / "scores.npy", values[query_id])
            with open(output / "info.json", "w") as handle:
                json.dump(
                    {
                        "query": by_id[query_id],
                        "method": TRAJECTORY_INVERSE_DAS_100Q_METHOD,
                        "lambda": lam,
                        "parameter_source": "final EMA",
                        "family": info["family"],
                        "projection_dim": TRAJECTORY_INVERSE_DAS_PROJ_DIM,
                        "outer_probe_count": int(DAS_NUM_MC),
                        "included_timestamp_indices": list(range(99)),
                        "endpoint_excluded": True,
                        "term_weight": 1.0 / (99.0 * float(DAS_NUM_MC)),
                        "train_loss_is_query_dependent": True,
                        "loss_definition": info["loss_definition"],
                    },
                    handle,
                    indent=2,
                )
    print("[done] merged trajectory inverse-noise DAS q00-q99")


if __name__ == "__main__":
    main()
