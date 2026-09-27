"""Evaluate q00-q09 endpoint-unlearning MUCS scores against existing LDS."""

import json

import numpy as np

from checkpoint_counterfactual_metrics import spearman_correlation
from mucs_endpoint_unlearning_config import *


def main():
    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    output = {"method": MUCS_METHOD, "query_ids": list(MUCS_QUERY_IDS), "results": {}}
    for metric in MUCS_LDS_METRICS:
        observed = np.load(LDS_DIR / f"observed_{metric}.npy").astype(np.float64)
        by_sign = {}
        for sign_name, multiplier in (("saved_score", 1.0), ("negated_score", -1.0)):
            queries = []
            for qid in MUCS_QUERY_IDS:
                score_path = ATTR_DIR / MUCS_METHOD / f"q{qid:02d}" / "scores.npy"
                if not score_path.is_file():
                    raise FileNotFoundError(score_path)
                score = np.load(score_path).astype(np.float64).reshape(-1)
                prediction = multiplier * (membership @ score)
                rho = spearman_correlation(prediction, observed[qid])
                queries.append({"query_id": qid, "spearman": rho})
            mean = float(np.nanmean([item["spearman"] for item in queries]))
            by_sign[sign_name] = {
                "multiplier": multiplier,
                "mean": mean,
                "queries": queries,
            }
            print(f"[LDS] {metric} {sign_name}: mean={mean:.6f}", flush=True)
        output["results"][metric] = by_sign
    output_path = LDS_DIR / f"{MUCS_METHOD}_both_signs_q00_q09.json"
    with open(output_path, "w") as handle:
        json.dump(output, handle, indent=2)
    print(f"[saved] {output_path}", flush=True)


if __name__ == "__main__":
    main()
