"""Evaluate all three reference-trajectory MC4 contractions on q00-q09."""

import argparse
import json

import numpy as np
from scipy.stats import spearmanr

from reference_traj_mc4_config import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, default=REF_MC4_DEFAULT_EPSILON)
    parser.add_argument("--train-mc", type=int, default=REF_MC4_DEFAULT_TRAIN_MC)
    args = parser.parse_args()
    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    methods = ref_mc4_methods(args.epsilon, args.train_mc)
    result = {"epsilon": args.epsilon, "train_mc": args.train_mc, "methods": {}}
    for contraction, method in methods.items():
        scores = [
            np.load(ATTR_DIR / method / f"q{qid:02d}" / "scores.npy").astype(np.float64)
            for qid in REF_MC4_QUERY_IDS
        ]
        method_result = {"contraction": contraction, "metrics": {}}
        print(f"\nMETHOD: {method}", flush=True)
        for metric in LDS_METRICS:
            observed = np.load(LDS_DIR / f"observed_{metric}.npy").astype(np.float64)
            signs = {}
            for sign_name, sign in (("negative", -1.0), ("positive", 1.0)):
                per_query = [
                    float(
                        spearmanr(
                            sign * (membership @ scores[position]), observed[qid]
                        ).statistic
                    )
                    for position, qid in enumerate(REF_MC4_QUERY_IDS)
                ]
                signs[sign_name] = {
                    "mean": float(np.nanmean(per_query)),
                    "per_query": per_query,
                }
            method_result["metrics"][metric] = signs
            print(
                f"{metric:30s} sign=-1 {signs['negative']['mean']:+.6f} | "
                f"sign=+1 {signs['positive']['mean']:+.6f}",
                flush=True,
            )
        result["methods"][method] = method_result
    path = LDS_DIR / (
        f"reference_traj_mc4_eps_{epsilon_tag(args.epsilon)}_"
        f"train_mc{args.train_mc}_q00_q09.json"
    )
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
