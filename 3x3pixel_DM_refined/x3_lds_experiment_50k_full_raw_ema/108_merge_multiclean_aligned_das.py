"""Merge ten-anchor predicted-clean aligned-DAS timestamp shards."""

import argparse
import json

import numpy as np

from multiclean_das_config import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp-shard-count", type=int, default=4)
    args = parser.parse_args()
    totals = {
        float(lam): np.zeros(
            (len(MULTICLEAN_QUERY_IDS), N_TRAIN), dtype=np.float64
        )
        for lam in DAS_LAMBDAS
    }
    covered = []
    term_count = 0
    metadata = []
    for shard_index in range(args.timestamp_shard_count):
        root = multiclean_shard_root(shard_index, args.timestamp_shard_count)
        with open(root / "done.json") as handle:
            info = json.load(handle)
        metadata.append(info)
        if info["query_ids"] != list(MULTICLEAN_QUERY_IDS):
            raise ValueError(f"query IDs differ in {root}")
        if info["anchor_indices"] != list(MULTICLEAN_ANCHOR_INDICES):
            raise ValueError(f"anchor indices differ in {root}")
        covered.extend(int(value) for value in info["timestamp_indices"])
        term_count += int(info["term_count"])
        for lam in DAS_LAMBDAS:
            totals[float(lam)] += np.load(
                root / f"lambda_{lambda_tag(lam)}.npy"
            )
    if sorted(covered) != list(range(len(DAS_TIMESTEPS))):
        raise ValueError("timestamp shards do not cover 0..99 exactly")
    expected_terms = len(DAS_TIMESTEPS) * int(DAS_NUM_MC)
    if term_count != expected_terms:
        raise ValueError(f"term_count={term_count}, expected={expected_terms}")

    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    by_id = {int(record["query_id"]): record for record in manifest}
    for position, query_id in enumerate(MULTICLEAN_QUERY_IDS):
        for lam, values in totals.items():
            output = (
                ATTR_DIR
                / MULTICLEAN_METHOD
                / f"q{query_id:02d}"
                / f"lambda_{lambda_tag(lam)}"
            )
            output.mkdir(parents=True, exist_ok=True)
            np.save(output / "scores.npy", values[position] / expected_terms)
            with open(output / "info.json", "w") as handle:
                json.dump(
                    {
                        "query": by_id[query_id],
                        "method": MULTICLEAN_METHOD,
                        "lambda": lam,
                        "anchor_snapshot_indices": list(MULTICLEAN_ANCHOR_INDICES),
                        "anchor_count": MULTICLEAN_ANCHOR_COUNT,
                        "anchor_reduction": "sum of per-anchor squared DAS scores",
                        "clean_definition": (
                            "x0_hat=(x_k-sqrt(1-alpha_bar_k)*eps_ema(x_k,k))"
                            "/sqrt(alpha_bar_k)"
                        ),
                        "parameter_source": "final EMA",
                        "projection_dim": int(DAS_PROJ_DIM),
                        "das_timestamps": [int(value) for value in DAS_TIMESTEPS],
                        "num_mc": int(DAS_NUM_MC),
                        "normalize_projected_grads": bool(
                            DAS_NORMALIZE_PROJECTED_GRADS
                        ),
                        "noise_alignment": metadata[0]["noise_alignment"],
                        "train_feature_reuse": metadata[0]["train_feature_reuse"],
                        "timestamp_shards": args.timestamp_shard_count,
                    },
                    handle,
                    indent=2,
                )
    print(f"[saved] {MULTICLEAN_METHOD} q00-q09 all lambdas", flush=True)


if __name__ == "__main__":
    main()
