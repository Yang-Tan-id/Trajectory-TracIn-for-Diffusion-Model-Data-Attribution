"""Evaluate Bundle TracIn in the full-model-centered coordinate system.

The existing Bundle run stores A_S for each of the 192 random fixed-size LDS
subsets.  Because every datapoint is included with probability p, the subset
mean is an unbiased Monte Carlo estimate of p A_D.  This script estimates the
full-data response as mean_S(A_S) / p and scores each subset by

    mean_t ||A_D - A_S||_2^2,

which matches observed full-model-versus-subset deviations more directly than
the original mean_t ||A_S||_2^2 contraction.
"""

import argparse
import json

import numpy as np
from scipy.stats import spearmanr

from bundle_tracin_config import *


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
        "per_query": values.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-ids", default="0-9")
    args = parser.parse_args()
    query_ids = parse_query_ids(args.query_ids)

    membership = np.load(MASK_DIR / "membership.npy", mmap_mode="r")
    inclusion_fraction = float(membership.sum(axis=1).mean() / membership.shape[1])
    if not 0.0 < inclusion_fraction < 1.0:
        raise ValueError(f"invalid inclusion fraction: {inclusion_fraction}")

    method_root = ATTR_DIR / BUNDLE_TRACIN_METHOD
    scores_by_query = {}
    for query_id in query_ids:
        query_root = method_root / f"q{query_id:02d}"
        subset_vectors = np.load(
            query_root / "bundle_vectors.npy", mmap_mode="r"
        ).astype(np.float64)
        expected_shape = (
            membership.shape[0],
            TRAJ_SNAPSHOTS,
            BUNDLE_TRACIN_OUTPUT_DIM,
        )
        if subset_vectors.shape != expected_shape:
            raise ValueError(
                f"q{query_id:02d}: vectors={subset_vectors.shape}, "
                f"expected={expected_shape}"
            )

        full_vector = subset_vectors.mean(axis=0) / inclusion_fraction
        complement_vectors = full_vector[None, :, :] - subset_vectors
        complement_scores = np.mean(
            np.sum(np.square(complement_vectors), axis=-1), axis=-1
        )
        np.save(
            query_root / "estimated_full_bundle_vector.npy",
            full_vector.astype(np.float32),
        )
        np.save(
            query_root / "complement_bundle_scores.npy",
            complement_scores.astype(np.float64),
        )
        scores_by_query[query_id] = complement_scores
        print(
            f"[complement] q{query_id:02d} p={inclusion_fraction:.6f} "
            f"score=[{complement_scores.min():.6e},{complement_scores.max():.6e}]",
            flush=True,
        )

    output = {
        "method": BUNDLE_TRACIN_METHOD + "_estimated_full_complement",
        "source_method": BUNDLE_TRACIN_METHOD,
        "query_ids": query_ids,
        "query_count": len(query_ids),
        "inclusion_fraction": inclusion_fraction,
        "full_vector_estimator": "mean_subset_vector / inclusion_fraction",
        "score_definition": "mean_t ||estimated_A_D_t - A_S_t||_2^2",
        "metrics": {},
    }

    print("\nFULL-MODEL-CENTERED BUNDLE LDS", flush=True)
    for metric in LDS_METRICS:
        observed = np.load(LDS_DIR / f"observed_{metric}.npy").astype(np.float64)
        positive = np.asarray(
            [
                spearmanr(
                    scores_by_query[query_id], observed[query_id]
                ).statistic
                for query_id in query_ids
            ],
            dtype=np.float64,
        )
        output["metrics"][metric] = {
            "positive": summarize(positive),
            "negative": summarize(-positive),
        }
        print(
            f"{metric:30s} sign=+1 {np.nanmean(positive):+.6f}"
            f"±{np.nanstd(positive):.6f} | sign=-1 "
            f"{np.nanmean(-positive):+.6f}±{np.nanstd(positive):.6f}",
            flush=True,
        )

    output_path = LDS_DIR / (
        f"{BUNDLE_TRACIN_METHOD}_estimated_full_complement_"
        f"q{query_ids[0]:02d}_q{query_ids[-1]:02d}.json"
    )
    with open(output_path, "w") as handle:
        json.dump(output, handle, indent=2)
    print(f"[saved] {output_path}", flush=True)


if __name__ == "__main__":
    main()
