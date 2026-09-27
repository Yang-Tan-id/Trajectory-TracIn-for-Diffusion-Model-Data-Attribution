"""Evaluate staged Traj and final-EMA/all-50k DAS against staged LDS models."""

import json

import numpy as np
from scipy.stats import spearmanr

from staged_lds_config import *


METRICS = (
    "simple_loss_ema", "simple_loss_raw", "traj_ref_ema", "traj_ref_raw",
    "endpoint_deviation_ema", "endpoint_deviation_raw",
    "trajectory_state_mse_ema", "trajectory_state_mse_raw",
)


def tag(value):
    return str(float(value)).replace(".", "p")


def evaluate(method, score_paths, membership, observed):
    result = {"method": method, "metrics": {}}
    for metric in METRICS:
        actual = observed[metric]
        signs = {}
        for sign_name, sign in (("negative", -1.0), ("positive", 1.0)):
            per_query = []
            for position, path in enumerate(score_paths):
                score = np.load(path).astype(np.float64)
                prediction = sign * (membership @ score)
                per_query.append(float(spearmanr(prediction, actual[position]).statistic))
            signs[sign_name] = {"mean": float(np.nanmean(per_query)), "per_query": per_query}
        result["metrics"][metric] = signs
    return result


def main():
    membership = np.load(STAGED_MASK_DIR / "membership.npy").astype(np.float64)
    observed = {metric: np.load(STAGED_LDS_DIR / f"observed_{metric}.npy").astype(np.float64) for metric in METRICS}
    results = []
    traj_paths = [STAGED_ATTR_DIR / STAGED_TRAJ_METHOD / f"q{qid:02d}" / "scores.npy" for qid in STAGED_QUERY_IDS]
    results.append(evaluate(STAGED_TRAJ_METHOD, traj_paths, membership, observed))
    for lam in DAS_LAMBDAS:
        paths = [STAGED_ATTR_DIR / STAGED_DAS_METHOD / f"q{qid:02d}" / f"lambda_{tag(lam)}" / "scores.npy" for qid in STAGED_QUERY_IDS]
        entry = evaluate(f"{STAGED_DAS_METHOD}_lambda_{tag(lam)}", paths, membership, observed)
        entry["lambda"] = float(lam)
        results.append(entry)
    STAGED_LDS_DIR.mkdir(parents=True, exist_ok=True)
    out = STAGED_LDS_DIR / "staged_traj_vs_das.json"
    with open(out, "w") as handle:
        json.dump({"results": results}, handle, indent=2)
    primary = "traj_ref_raw"
    for entry in results:
        negative = entry["metrics"][primary]["negative"]["mean"]
        positive = entry["metrics"][primary]["positive"]["mean"]
        print(f"{entry['method']}: {primary} sign=-1 {negative:+.6f} | sign=+1 {positive:+.6f}", flush=True)
    print(f"[saved] {out}", flush=True)


if __name__ == "__main__":
    main()
