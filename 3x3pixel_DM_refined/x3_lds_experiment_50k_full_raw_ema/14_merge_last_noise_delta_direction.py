"""Merge delta-direction timestamp shards into per-query score artifacts."""

import argparse
import json

import numpy as np

from checkpoint_counterfactual_config import CF_DIRECTION_SCORE_METHOD
from exp_config import ATTR_DIR, N_TRAIN
from run_exact_traj_next_bank import atomic_json_save, atomic_numpy_save


SHARD_NAMESPACE = "_last_noise_delta_direction_shards"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("prompted", "unprompted"), required=True)
    parser.add_argument("--timestamp-shard-count", type=int, required=True)
    args = parser.parse_args()
    merged = None
    query_ids = None
    covered = []
    metadata = []
    for shard in range(args.timestamp_shard_count):
        root = (
            ATTR_DIR / SHARD_NAMESPACE / args.family
            / f"shard_{shard:02d}_of_{args.timestamp_shard_count:02d}"
        )
        with open(root / "done.json") as handle:
            done = json.load(handle)
        values = np.load(root / "linear.npy").astype(np.float64)
        if values.shape[1] != N_TRAIN:
            raise ValueError(f"unexpected score shape {values.shape} in {root}")
        if query_ids is None:
            query_ids = done["query_ids"]
            merged = values
        else:
            if done["query_ids"] != query_ids:
                raise ValueError("query IDs differ between shards")
            merged += values
        covered.extend(done["timestamp_indices"])
        metadata.append(done)
    if sorted(covered) != list(range(100)):
        raise ValueError(f"timestamp coverage is not 0..99: {sorted(covered)}")
    for row, query_id in enumerate(query_ids):
        output = ATTR_DIR / CF_DIRECTION_SCORE_METHOD / f"q{int(query_id):02d}"
        output.mkdir(parents=True, exist_ok=True)
        atomic_numpy_save(output / "scores.npy", merged[row])
        atomic_json_save(
            output / "meta.json",
            {
                "method": CF_DIRECTION_SCORE_METHOD,
                "query_id": int(query_id),
                "family": args.family,
                "score_sign": "saved_as_positive_lr_times_train_gradient_dot_query_direction_gradient",
                "lds_evaluates_both_saved_score_and_negated_score": True,
                "shards": metadata,
            },
        )
    print(
        f"[done] merged {args.family}: queries={len(query_ids)} "
        f"shape={merged.shape} method={CF_DIRECTION_SCORE_METHOD}",
        flush=True,
    )


if __name__ == "__main__":
    main()
