"""Merge checkpoint shards, form Bundle scores, and evaluate every LDS target."""

import argparse
import json

import numpy as np
from scipy.stats import spearmanr

from bundle_tracin_config import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-ids", default="0-9")
    parser.add_argument(
        "--checkpoint-shard-count",
        type=int,
        default=BUNDLE_TRACIN_CHECKPOINT_SHARDS,
    )
    args = parser.parse_args()
    query_ids = parse_query_ids(args.query_ids)
    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    by_id = {int(record["query_id"]): record for record in manifest}
    membership = np.load(MASK_DIR / "membership.npy", mmap_mode="r")
    mask_count = membership.shape[0]
    method_root = ATTR_DIR / BUNDLE_TRACIN_METHOD
    method_root.mkdir(parents=True, exist_ok=True)

    scores_by_query = {}
    for query_id in query_ids:
        family = by_id[query_id]["family"]
        merged = None
        covered = []
        for shard_index in range(args.checkpoint_shard_count):
            shard_root = (
                ATTR_DIR
                / BUNDLE_TRACIN_SHARD_NAMESPACE
                / family
                / f"shard_{shard_index:02d}_of_{args.checkpoint_shard_count:02d}"
            )
            with open(shard_root / "done.json") as handle:
                metadata = json.load(handle)
            if query_id not in metadata["query_ids"]:
                raise ValueError(f"q{query_id:02d} missing from {shard_root}")
            covered.extend(int(value) for value in metadata["checkpoint_indices"])
            value = np.load(
                shard_root / f"q{query_id:02d}_bundle_vectors.npy"
            ).astype(np.float64)
            expected_shape = (mask_count, TRAJ_SNAPSHOTS, BUNDLE_TRACIN_OUTPUT_DIM)
            if value.shape != expected_shape:
                raise ValueError(
                    f"{shard_root} q{query_id:02d} shape={value.shape}, "
                    f"expected={expected_shape}"
                )
            merged = value if merged is None else merged + value
        if sorted(covered) != list(range(50)):
            raise ValueError(f"checkpoint shards do not cover 0..49 exactly: {covered}")

        # Required Bundle contraction: add datapoint/checkpoint response vectors
        # first, then take the output-vector norm, then average over timestamps.
        bundle_scores = np.mean(np.sum(np.square(merged), axis=-1), axis=-1)
        query_root = method_root / f"q{query_id:02d}"
        query_root.mkdir(parents=True, exist_ok=True)
        np.save(query_root / "bundle_vectors.npy", merged.astype(np.float32))
        np.save(query_root / "bundle_scores.npy", bundle_scores.astype(np.float64))
        with open(query_root / "info.json", "w") as handle:
            json.dump(
                {
                    "method": BUNDLE_TRACIN_METHOD,
                    "query": by_id[query_id],
                    "score_definition": "mean_t ||sum_i sum_c -eta_c J_q,c,t g_i,c||_2^2",
                    "vector_shape": list(merged.shape),
                    "subset_count": mask_count,
                    "timestamp_count": TRAJ_SNAPSHOTS,
                    "output_dimension": BUNDLE_TRACIN_OUTPUT_DIM,
                    "train_mc": BUNDLE_TRACIN_TRAIN_MC,
                    "train_t_noise_alignment": "independent_of_query",
                    "parameter_source": BUNDLE_TRACIN_PARAM_SOURCE,
                    "projection_dimension": BUNDLE_TRACIN_PROJ_DIM,
                    "checkpoint_count": 50,
                },
                handle,
                indent=2,
            )
        scores_by_query[query_id] = bundle_scores
        print(
            f"[merged] q{query_id:02d} vectors={merged.shape} "
            f"score=[{bundle_scores.min():.6e},{bundle_scores.max():.6e}]",
            flush=True,
        )

    output = {
        "method": BUNDLE_TRACIN_METHOD,
        "query_ids": query_ids,
        "query_count": len(query_ids),
        "score_definition": "mean_t ||sum_i sum_c -eta_c J_q,c,t g_i,c||_2^2",
        "metrics": {},
    }
    print(f"\nMETHOD: {BUNDLE_TRACIN_METHOD}", flush=True)
    for metric in LDS_METRICS:
        observed = np.load(LDS_DIR / f"observed_{metric}.npy").astype(np.float64)
        positive = np.asarray(
            [
                spearmanr(scores_by_query[query_id], observed[query_id]).statistic
                for query_id in query_ids
            ],
            dtype=np.float64,
        )
        signs = {
            "positive": {
                "mean": float(np.nanmean(positive)),
                "std": float(np.nanstd(positive)),
                "per_query": positive.tolist(),
            },
            "negative": {
                "mean": float(np.nanmean(-positive)),
                "std": float(np.nanstd(-positive)),
                "per_query": (-positive).tolist(),
            },
        }
        output["metrics"][metric] = signs
        print(
            f"{metric:30s} sign=+1 {signs['positive']['mean']:+.6f}"
            f"±{signs['positive']['std']:.6f} | sign=-1 "
            f"{signs['negative']['mean']:+.6f}±{signs['negative']['std']:.6f}",
            flush=True,
        )

    output_path = LDS_DIR / f"{BUNDLE_TRACIN_METHOD}_q{query_ids[0]:02d}_q{query_ids[-1]:02d}.json"
    with open(output_path, "w") as handle:
        json.dump(output, handle, indent=2)
    print(f"[saved] {output_path}", flush=True)


if __name__ == "__main__":
    main()
