"""Post-hoc included/complement variants for vector trajectory DAS Bundle."""

import argparse
import json

import numpy as np
from scipy.stats import spearmanr

from exp_config import *


METHOD = "vector_trajectory_das_ema_mc10_projected4096_lambda100_bundle"


def energy(vectors):
    return np.mean(np.sum(np.square(vectors), axis=-1), axis=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-ids", default="0-9")
    args = parser.parse_args()
    query_ids = []
    for token in args.query_ids.split(","):
        if "-" in token:
            left, right = map(int, token.split("-", 1))
            query_ids.extend(range(left, right + 1))
        elif token.strip():
            query_ids.append(int(token))

    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    inclusion_rate = membership.mean(axis=1)
    if np.any(inclusion_rate <= 0.0) or not np.allclose(inclusion_rate, inclusion_rate[0]):
        raise ValueError("this evaluator expects equal-size nonempty LDS subsets")
    p = float(inclusion_rate[0])
    scores = {name: {} for name in ("included", "complement", "removal_marginal", "subset_centered")}
    for qid in query_ids:
        vectors = np.load(
            ATTR_DIR / METHOD / f"q{qid:02d}" / "subset_vectors.npy"
        ).astype(np.float64)
        # E_S[sum_{i in S} v_i] = p * sum_i v_i for uniform random subsets.
        full = vectors.mean(axis=0) / p
        complement = full[None] - vectors
        full_energy = float(energy(full[None])[0])
        variants = {
            "included": energy(vectors),
            "complement": energy(complement),
            "removal_marginal": full_energy - energy(complement),
            "subset_centered": energy(vectors - p * full[None]),
        }
        for name, value in variants.items():
            scores[name][qid] = value
        print(
            f"[q{qid:02d}] p={p:.6f} full={full_energy:.6e} "
            f"included=[{variants['included'].min():.3e},{variants['included'].max():.3e}]",
            flush=True,
        )

    output = {"source_method": METHOD, "query_ids": query_ids, "subset_fraction": p, "variants": {}}
    for variant, by_query in scores.items():
        output["variants"][variant] = {}
        print(f"\nVARIANT: {variant}")
        for metric in LDS_METRICS:
            observed = np.load(LDS_DIR / f"observed_{metric}.npy").astype(np.float64)
            values = np.array([
                spearmanr(by_query[qid], observed[qid]).statistic for qid in query_ids
            ])
            result = {
                "positive": {"mean": float(np.nanmean(values)), "std": float(np.nanstd(values)), "per_query": values.tolist()},
                "negative": {"mean": float(np.nanmean(-values)), "std": float(np.nanstd(-values)), "per_query": (-values).tolist()},
            }
            output["variants"][variant][metric] = result
            print(
                f"{metric:30s} sign=+1 {np.nanmean(values):+.6f}±{np.nanstd(values):.6f} | "
                f"sign=-1 {np.nanmean(-values):+.6f}±{np.nanstd(values):.6f}"
            )
    path = LDS_DIR / f"{METHOD}_bundle_variants_q{query_ids[0]:02d}_q{query_ids[-1]:02d}.json"
    with open(path, "w") as handle:
        json.dump(output, handle, indent=2)
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
