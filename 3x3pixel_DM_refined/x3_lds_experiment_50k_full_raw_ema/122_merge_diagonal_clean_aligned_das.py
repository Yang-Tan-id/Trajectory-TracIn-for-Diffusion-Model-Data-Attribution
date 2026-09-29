"""Merge timestamp-diagonal predicted-clean DAS shards."""

import argparse
import json

import numpy as np

from diagonal_clean_das_config import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp-shard-count", type=int, default=4)
    parser.add_argument("--timestamp-count", type=int, choices=(10, 100), default=100)
    args = parser.parse_args()
    selected_indices = diagonal_clean_indices(args.timestamp_count)
    method = diagonal_clean_method(args.timestamp_count)
    totals = {
        float(lam): np.zeros((100, N_TRAIN), dtype=np.float64)
        for lam in DAS_LAMBDAS
    }
    term_count = 0
    metadata = []
    for family in DIAGONAL_CLEAN_FAMILIES:
        covered = []
        query_ids = list(diagonal_clean_query_ids(family))
        family_terms = 0
        for shard_index in range(args.timestamp_shard_count):
            root = diagonal_clean_shard_root(
                family,
                shard_index,
                args.timestamp_shard_count,
                args.timestamp_count,
            )
            with open(root / "done.json") as handle:
                info = json.load(handle)
            metadata.append(info)
            if info["query_ids"] != query_ids or info["family"] != family:
                raise ValueError(f"query/family mismatch in {root}")
            covered.extend(int(value) for value in info["snapshot_indices"])
            family_terms += int(info["term_count"])
            for lam in DAS_LAMBDAS:
                totals[float(lam)][query_ids] += np.load(
                    root / f"lambda_{lambda_tag(lam)}.npy"
                )
        if sorted(covered) != sorted(selected_indices):
            raise ValueError(
                f"{family} shards do not cover expected indices {selected_indices}"
            )
        expected_family_terms = args.timestamp_count * int(DAS_NUM_MC)
        if family_terms != expected_family_terms:
            raise ValueError(
                f"{family} term_count={family_terms}, expected={expected_family_terms}"
            )
        term_count += family_terms
    expected_terms = (
        len(DIAGONAL_CLEAN_FAMILIES) * args.timestamp_count * int(DAS_NUM_MC)
    )
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
                / method
                / f"q{query_id:02d}"
                / f"lambda_{lambda_tag(lam)}"
            )
            output.mkdir(parents=True, exist_ok=True)
            np.save(
                output / "scores.npy",
                values[position] / (args.timestamp_count * int(DAS_NUM_MC)),
            )
            with open(output / "info.json", "w") as handle:
                json.dump(
                    {
                        "query": by_id[query_id],
                        "method": method,
                        "lambda": lam,
                        "snapshot_indices": list(selected_indices),
                        "trajectory_timesteps": [
                            int(timestamps[index]) for index in selected_indices
                        ],
                        "pairing": (
                            "predicted clean from reference snapshot k is scored "
                            "only at the same diffusion noise level k"
                        ),
                        "clean_definition": (
                            "x0_hat_k=(x_k-sqrt(1-alpha_bar_k)*eps_ema(x_k,k))"
                            "/sqrt(alpha_bar_k)"
                        ),
                        "reduction": (
                            f"mean over {args.timestamp_count} timestamps and MC10 "
                            "of squared DAS score"
                        ),
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
    print(f"[saved] {method} q00-q99 all lambdas", flush=True)


if __name__ == "__main__":
    main()
