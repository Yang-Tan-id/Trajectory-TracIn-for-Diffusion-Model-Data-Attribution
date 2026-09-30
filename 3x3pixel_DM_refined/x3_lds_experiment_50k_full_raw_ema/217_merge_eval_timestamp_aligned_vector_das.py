"""Merge ten aligned vector-DAS timestamps and evaluate the lambda sweep."""

import argparse
import json

import numpy as np
from scipy.stats import spearmanr

from timestamp_aligned_vector_das_config import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=FAMILIES, default="prompted")
    parser.add_argument("--query-ids", default="0-9")
    args = parser.parse_args()
    query_ids = []
    for token in args.query_ids.split(","):
        if "-" in token:
            a, b = map(int, token.split("-", 1)); query_ids.extend(range(a, b + 1))
        elif token.strip(): query_ids.append(int(token))
    scores = {lam: {} for lam in TAVD_LAMBDAS}
    output_root = ATTR_DIR / TAVD_METHOD
    for qid in query_ids:
        for damping in TAVD_LAMBDAS:
            vectors = np.stack([
                np.load(TAVD_ROOT / args.family / f"task_{task:02d}" / f"q{qid:02d}_lambda_{lambda_tag(damping)}.npy")
                for task in range(len(TAVD_POSITIONS))
            ], axis=1).astype(np.float32)
            values = np.mean(np.sum(np.square(vectors.astype(np.float64)), axis=-1), axis=-1)
            root = output_root / f"lambda_{lambda_tag(damping)}" / f"q{qid:02d}"
            root.mkdir(parents=True, exist_ok=True)
            np.save(root / "subset_vectors.npy", vectors)
            np.save(root / "bundle_scores.npy", values)
            scores[damping][qid] = values
    result = {"method": TAVD_METHOD, "query_ids": query_ids, "timestamp_positions": list(TAVD_POSITIONS), "results": {}}
    for damping in TAVD_LAMBDAS:
        tag = lambda_tag(damping)
        result["results"][tag] = {}
        print(f"\nLAMBDA={damping:g}")
        for metric in LDS_METRICS:
            observed = np.load(LDS_DIR / f"observed_{metric}.npy").astype(np.float64)
            values = np.array([spearmanr(scores[damping][qid], observed[qid]).statistic for qid in query_ids])
            result["results"][tag][metric] = {"positive": {"mean": float(np.nanmean(values)), "std": float(np.nanstd(values)), "per_query": values.tolist()}, "negative": {"mean": float(np.nanmean(-values)), "std": float(np.nanstd(-values)), "per_query": (-values).tolist()}}
            print(f"{metric:30s} sign=+1 {np.nanmean(values):+.6f}±{np.nanstd(values):.6f} | sign=-1 {np.nanmean(-values):+.6f}±{np.nanstd(values):.6f}")
    path = LDS_DIR / f"{TAVD_METHOD}_lambda_sweep_q{query_ids[0]:02d}_q{query_ids[-1]:02d}.json"
    with open(path, "w") as handle: json.dump(result, handle, indent=2)
    print(f"[saved] {path}")


if __name__ == "__main__": main()
