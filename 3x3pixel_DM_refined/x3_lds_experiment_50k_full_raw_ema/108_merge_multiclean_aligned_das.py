"""Merge trajectory-state relative-forward aligned-DAS shards."""

import argparse
import json

import numpy as np

from multiclean_das_config import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp-shard-count", type=int, default=2)
    args = parser.parse_args()
    totals = {
        float(lam): np.zeros(
            (len(MULTICLEAN_QUERY_IDS), N_TRAIN), dtype=np.float64
        )
        for lam in DAS_LAMBDAS
    }
    covered_by_family = {family: [] for family in MULTICLEAN_FAMILIES}
    term_count_by_family = {family: 0 for family in MULTICLEAN_FAMILIES}
    metadata = []
    for family in MULTICLEAN_FAMILIES:
        query_ids = multiclean_query_ids(family)
        for shard_index in range(args.timestamp_shard_count):
            root = multiclean_shard_root(
                family, shard_index, args.timestamp_shard_count
            )
            with open(root / "done.json") as handle:
                info = json.load(handle)
            metadata.append(info)
            if info["family"] != family:
                raise ValueError(f"family differs in {root}")
            if info["query_ids"] != list(query_ids):
                raise ValueError(f"query IDs differ in {root}")
            if info["anchor_indices"] != list(MULTICLEAN_ANCHOR_INDICES):
                raise ValueError(f"anchor indices differ in {root}")
            if info["anchor_das_timestamp_counts"] != list(
                MULTICLEAN_ANCHOR_DAS_COUNTS
            ):
                raise ValueError(f"anchor DAS timestamp counts differ in {root}")
            covered_by_family[family].extend(
                int(value) for value in info["selected_pair_indices"]
            )
            term_count_by_family[family] += int(info["term_count"])
            for lam in DAS_LAMBDAS:
                values = np.load(root / f"lambda_{lambda_tag(lam)}.npy")
                expected_shape = (len(query_ids), N_TRAIN)
                if values.shape != expected_shape:
                    raise ValueError(
                        f"score shape={values.shape}, expected={expected_shape} in {root}"
                    )
                totals[float(lam)][list(query_ids)] += values
    expected_pairs = sum(MULTICLEAN_ANCHOR_DAS_COUNTS)
    expected_terms = expected_pairs * int(DAS_NUM_MC)
    for family in MULTICLEAN_FAMILIES:
        if sorted(covered_by_family[family]) != list(range(expected_pairs)):
            raise ValueError(
                f"{family} shards do not cover relative-forward pairs exactly"
            )
        if term_count_by_family[family] != expected_terms:
            raise ValueError(
                f"{family} term_count={term_count_by_family[family]}, "
                f"expected={expected_terms}"
            )

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
            np.save(output / "scores.npy", values[position])
            with open(output / "info.json", "w") as handle:
                json.dump(
                    {
                        "query": by_id[query_id],
                        "method": MULTICLEAN_METHOD,
                        "lambda": lam,
                        "anchor_snapshot_indices": list(MULTICLEAN_ANCHOR_INDICES),
                        "anchor_das_timestamp_counts": list(
                            MULTICLEAN_ANCHOR_DAS_COUNTS
                        ),
                        "anchor_count": MULTICLEAN_ANCHOR_COUNT,
                        "anchor_reduction": (
                            "sum of independently target-and-MC-averaged "
                            "per-anchor squared DAS scores"
                        ),
                        "anchor_timesteps": metadata[0]["anchor_timesteps"],
                        "anchor_target_timesteps": metadata[0][
                            "anchor_target_timesteps"
                        ],
                        "query_definition": "cached reference trajectory state x_t",
                        "initial_t999_anchor": "skipped",
                        "parameter_source": "final EMA",
                        "projection_dim": int(DAS_PROJ_DIM),
                        "anchor_timestamp_rule": (
                            "anchor x_t uses its configured number of evenly "
                            "spaced targets s in [t,999]"
                        ),
                        "num_mc": int(DAS_NUM_MC),
                        "normalize_projected_grads": bool(
                            DAS_NORMALIZE_PROJECTED_GRADS
                        ),
                        "noise_alignment": metadata[0]["noise_alignment"],
                        "relative_forward_formula": metadata[0][
                            "relative_forward_formula"
                        ],
                        "family": by_id[query_id]["family"],
                        "timestamp_shards_per_family": args.timestamp_shard_count,
                    },
                    handle,
                    indent=2,
                )
    print(f"[saved] {MULTICLEAN_METHOD} q00-q99 all lambdas", flush=True)


if __name__ == "__main__":
    main()
