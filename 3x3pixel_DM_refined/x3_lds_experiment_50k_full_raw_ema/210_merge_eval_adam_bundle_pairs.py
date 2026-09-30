"""Merge 49 frozen-start AdamW pair tangents and evaluate Bundle LDS."""

import argparse
import json

import numpy as np
from scipy.stats import spearmanr

from adam_bundle_pair_config import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-ids", default="0-9")
    args = parser.parse_args()
    query_ids = parse_query_ids(args.query_ids)
    with open(QUERY_DIR / "manifest.json") as handle:
        by_id = {int(r["query_id"]): r for r in json.load(handle)}
    scores = {}
    method_root = ATTR_DIR / ADAM_BUNDLE_METHOD
    for qid in query_ids:
        family = by_id[qid]["family"]
        merged = None
        alphas = []
        for pair_index in range(49):
            root = ADAM_BUNDLE_ROOT / family / f"pair_{pair_index:02d}"
            with open(root / "done.json") as handle:
                metadata = json.load(handle)
            alphas.append(float(metadata["alpha"]))
            value = np.load(root / f"q{qid:02d}_vectors.npy").astype(np.float64)
            merged = value if merged is None else merged + value
        bundle = np.mean(np.sum(np.square(merged), axis=-1), axis=-1)
        output_dir = method_root / f"q{qid:02d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "bundle_vectors.npy", merged.astype(np.float32))
        np.save(output_dir / "bundle_scores.npy", bundle)
        with open(output_dir / "info.json", "w") as handle:
            json.dump({"method": ADAM_BUNDLE_METHOD, "query": by_id[qid], "pair_alphas": alphas, "score_definition": "mean_t ||sum_c J_start,c,t (-alpha_c dtheta_subset,c)||^2"}, handle, indent=2)
        scores[qid] = bundle
        print(f"[merged] q{qid:02d} score=[{bundle.min():.6e},{bundle.max():.6e}]", flush=True)
    result = {"method": ADAM_BUNDLE_METHOD, "query_ids": query_ids, "metrics": {}}
    print(f"\nMETHOD: {ADAM_BUNDLE_METHOD}")
    for metric in LDS_METRICS:
        observed = np.load(LDS_DIR / f"observed_{metric}.npy").astype(np.float64)
        positive = np.array([spearmanr(scores[qid], observed[qid]).statistic for qid in query_ids])
        result["metrics"][metric] = {
            "positive": {"mean": float(np.nanmean(positive)), "std": float(np.nanstd(positive)), "per_query": positive.tolist()},
            "negative": {"mean": float(np.nanmean(-positive)), "std": float(np.nanstd(-positive)), "per_query": (-positive).tolist()},
        }
        print(f"{metric:30s} sign=+1 {np.nanmean(positive):+.6f}±{np.nanstd(positive):.6f} | sign=-1 {np.nanmean(-positive):+.6f}±{np.nanstd(positive):.6f}")
    output = LDS_DIR / f"{ADAM_BUNDLE_METHOD}_q{query_ids[0]:02d}_q{query_ids[-1]:02d}.json"
    with open(output, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"[saved] {output}")


if __name__ == "__main__":
    main()
