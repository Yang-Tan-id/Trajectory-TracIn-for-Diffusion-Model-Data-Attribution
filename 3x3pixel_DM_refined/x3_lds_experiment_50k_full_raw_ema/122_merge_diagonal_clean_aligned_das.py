"""Merge timestamp-diagonal predicted-clean DAS shards."""

import argparse
import json

import numpy as np

from diagonal_clean_das_config import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp-shard-count", type=int, default=4)
    args = parser.parse_args()
    totals = {
        float(lam): np.zeros(
            (len(DIAGONAL_CLEAN_QUERY_IDS), N_TRAIN), dtype=np.float64
        )
        for lam in DAS_LAMBDAS
    }
    covered = []
    term_count = 0
    metadata = []
    for shard_index in range(args.timestamp_shard_count):
        root = diagonal_clean_shard_root(shard_index, args.timestamp_shard_count)
        with open(root / "done.json") as handle:
            info = json.load(handle)
        metadata.append(info)
        if info["query_ids"] != list(DIAGONAL_CLEAN_QUERY_IDS):
            raise ValueError(f"query IDs differ in {root}")
        covered.extend(int(value) for value in info["snapshot_indices"])
        term_count += int(info["term_count"])
        for lam in DAS_LAMBDAS:
            totals[float(lam)] += np.load(root / f"lambda_{lambda_tag(lam)}.npy")
    if sorted(covered) != list(range(100)):
        raise ValueError("timestamp shards do not cover snapshot indices 0..99")
    expected_terms = 100 * int(DAS_NUM_MC)
    if term_count != expected_terms:
        raise ValueError(f"term_count={term_count}, expected={expected_terms}")

    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    by_id = {int(record["query_id"]): record for record in manifest}
    timestamps = np.load(DIAGONAL_CLEAN_CACHE_DIR / "trajectory_t.npy")
    for position, query_id in enumerate(DIAGONAL_CLEAN_QUERY_IDS):
        for lam, values in totals.items():
            output = (
                ATTR_DIR
                / DIAGONAL_CLEAN_METHOD
                / f"q{query_id:02d}"
                / f"lambda_{lambda_tag(lam)}"
            )
            output.mkdir(parents=True, exist_ok=True)
            np.save(output / "scores.npy", values[position] / expected_terms)
            with open(output / "info.json", "w") as handle:
                json.dump(
                    {
                        "query": by_id[query_id],
                        "method": DIAGONAL_CLEAN_METHOD,
                        "lambda": lam,
                        "snapshot_indices": list(range(100)),
                        "trajectory_timesteps": [int(value) for value in timestamps],
                        "pairing": (
                            "predicted clean from reference snapshot k is scored "
                            "only at the same diffusion noise level k"
                        ),
                        "clean_definition": (
                            "x0_hat_k=(x_k-sqrt(1-alpha_bar_k)*eps_ema(x_k,k))"
                            "/sqrt(alpha_bar_k)"
                        ),
                        "reduction": "mean over 100 timestamps and MC10 of squared DAS score",
                        "parameter_source": "final EMA",
                        "projection_dim": int(DAS_PROJ_DIM),
                        "num_mc": int(DAS_NUM_MC),
                        "normalize_projected_grads": bool(DAS_NORMALIZE_PROJECTED_GRADS),
                        "noise_alignment": metadata[0]["noise_alignment"],
                        "timestamp_shards": args.timestamp_shard_count,
                    },
                    handle,
                    indent=2,
                )
    print(f"[saved] {DIAGONAL_CLEAN_METHOD} q00-q09 all lambdas", flush=True)


if __name__ == "__main__":
    main()
