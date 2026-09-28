"""Merge aligned-noise DAS timestamp shards into q00-q98 score files."""

import argparse
import json

import numpy as np

from exp_config import *
from tracin_das_config import TRACIN_DAS_FIRST99_QUERY_IDS


METHOD = "das_ema_aligned_noise"
SHARD_NAMESPACE = "_das_ema_aligned_noise_99q_shards"


def tag(value):
    return str(float(value)).replace(".", "p")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--timestamp-shard-count", type=int, default=2)
    args = parser.parse_args()
    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    by_id = {int(record["query_id"]): record for record in manifest}
    query_ids = [
        query_id
        for query_id in TRACIN_DAS_FIRST99_QUERY_IDS
        if by_id[query_id]["family"] == args.family
    ]
    totals = {
        float(lam): np.zeros((len(query_ids), N_TRAIN), dtype=np.float64)
        for lam in DAS_LAMBDAS
    }
    covered = []
    term_count = 0
    metadata = []
    for shard_index in range(args.timestamp_shard_count):
        root = (
            ATTR_DIR
            / SHARD_NAMESPACE
            / args.family
            / f"shard_{shard_index:02d}_of_{args.timestamp_shard_count:02d}"
        )
        with open(root / "done.json") as handle:
            info = json.load(handle)
        metadata.append(info)
        if info["query_ids"] != query_ids or info["family"] != args.family:
            raise ValueError(f"query/family mismatch in {root}")
        covered.extend(int(value) for value in info["timestamp_indices"])
        term_count += int(info["term_count"])
        for lam in DAS_LAMBDAS:
            totals[float(lam)] += np.load(root / f"lambda_{tag(lam)}.npy")
    if sorted(covered) != list(range(len(DAS_TIMESTEPS))):
        raise ValueError("timestamp shards do not cover 0..99 exactly")
    expected_terms = len(DAS_TIMESTEPS) * int(DAS_NUM_MC)
    if term_count != expected_terms:
        raise ValueError(f"merged term_count={term_count}, expected={expected_terms}")

    for query_position, query_id in enumerate(query_ids):
        for lam, values in totals.items():
            output = ATTR_DIR / METHOD / f"q{query_id:02d}" / f"lambda_{tag(lam)}"
            output.mkdir(parents=True, exist_ok=True)
            np.save(output / "scores.npy", values[query_position] / expected_terms)
            with open(output / "info.json", "w") as handle:
                json.dump(
                    {
                        "query": by_id[query_id],
                        "method": METHOD,
                        "lambda": lam,
                        "parameter_source": "ema",
                        "projection_dim": int(DAS_PROJ_DIM),
                        "timestamps": [int(value) for value in DAS_TIMESTEPS],
                        "num_mc": int(DAS_NUM_MC),
                        "train_gradient_mc_per_term": 1,
                        "effective_aligned_mc_per_timestamp": int(DAS_NUM_MC),
                        "normalize_projected_grads": bool(
                            DAS_NORMALIZE_PROJECTED_GRADS
                        ),
                        "noise_alignment": metadata[0]["noise_alignment"],
                        "timestamp_shards": args.timestamp_shard_count,
                    },
                    handle,
                    indent=2,
                )
    print(
        f"[saved] {METHOD} family={args.family} "
        f"q{query_ids[0]:02d}-q{query_ids[-1]:02d}",
        flush=True,
    )


if __name__ == "__main__":
    main()
