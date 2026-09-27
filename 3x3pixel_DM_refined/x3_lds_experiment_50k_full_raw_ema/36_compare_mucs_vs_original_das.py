"""Compare q00-q09 joint MUCS with original DAS EMA lambda=10."""

import csv
import json

import numpy as np

from checkpoint_counterfactual_metrics import spearman_correlation
from mucs_endpoint_unlearning_config import *


DAS_METHOD = "das_ema"
DAS_LAMBDA = 10.0
DAS_LAMBDA_TAG = "10p0"
TOPK = 1000


def score_path(method, qid):
    if method == MUCS_METHOD:
        return ATTR_DIR / method / f"q{qid:02d}" / "scores.npy"
    return (
        ATTR_DIR / DAS_METHOD / f"q{qid:02d}"
        / f"lambda_{DAS_LAMBDA_TAG}" / "scores.npy"
    )


def main():
    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    scores = {MUCS_METHOD: {}, DAS_METHOD: {}}
    score_comparison = []
    for qid in MUCS_QUERY_IDS:
        mucs = np.load(score_path(MUCS_METHOD, qid)).astype(np.float64).reshape(-1)
        das = np.load(score_path(DAS_METHOD, qid)).astype(np.float64).reshape(-1)
        if mucs.shape != (N_TRAIN,) or das.shape != (N_TRAIN,):
            raise ValueError(f"q{qid:02d}: score shapes MUCS={mucs.shape}, DAS={das.shape}")
        scores[MUCS_METHOD][qid] = mucs
        scores[DAS_METHOD][qid] = das
        mucs_top = set(np.argsort(-mucs, kind="stable")[:TOPK].tolist())
        das_top = set(np.argsort(-das, kind="stable")[:TOPK].tolist())
        overlap = len(mucs_top & das_top)
        score_comparison.append(
            {
                "query_id": qid,
                "score_spearman": spearman_correlation(mucs, das),
                "top1000_overlap": overlap,
                "top1000_overlap_fraction": overlap / TOPK,
                "mucs_min": float(mucs.min()),
                "mucs_max": float(mucs.max()),
                "das_min": float(das.min()),
                "das_max": float(das.max()),
            }
        )

    rows = []
    metric_results = {}
    methods = (("joint_mucs", MUCS_METHOD), ("das_ema_lambda_10", DAS_METHOD))
    for metric in MUCS_LDS_METRICS:
        observed = np.load(LDS_DIR / f"observed_{metric}.npy").astype(np.float64)
        metric_payload = {}
        for label, method in methods:
            sign_results = {}
            for sign_name, multiplier in (("saved_score", 1.0), ("negated_score", -1.0)):
                query_values = []
                for qid in MUCS_QUERY_IDS:
                    prediction = multiplier * (membership @ scores[method][qid])
                    rho = spearman_correlation(prediction, observed[qid])
                    query_values.append({"query_id": qid, "spearman": rho})
                    rows.append(
                        {
                            "metric": metric,
                            "method": label,
                            "sign": sign_name,
                            "query_id": qid,
                            "spearman": rho,
                        }
                    )
                sign_results[sign_name] = {
                    "multiplier": multiplier,
                    "mean": float(np.nanmean([item["spearman"] for item in query_values])),
                    "queries": query_values,
                }
            metric_payload[label] = sign_results
        metric_payload["canonical_das_sign"] = "negated_score"
        metric_payload["canonical_das_mean"] = metric_payload[
            "das_ema_lambda_10"
        ]["negated_score"]["mean"]
        metric_results[metric] = metric_payload

    score_rhos = np.asarray(
        [item["score_spearman"] for item in score_comparison], dtype=np.float64
    )
    overlaps = np.asarray(
        [item["top1000_overlap"] for item in score_comparison], dtype=np.float64
    )
    payload = {
        "query_ids": list(MUCS_QUERY_IDS),
        "mucs_method": MUCS_METHOD,
        "das_method": DAS_METHOD,
        "das_lambda": DAS_LAMBDA,
        "das_canonical_lds_prediction": "-(membership @ score)",
        "score_comparison": score_comparison,
        "score_comparison_summary": {
            "mean_score_spearman": float(score_rhos.mean()),
            "median_score_spearman": float(np.median(score_rhos)),
            "mean_top1000_overlap": float(overlaps.mean()),
            "median_top1000_overlap": float(np.median(overlaps)),
            "mean_top1000_overlap_fraction": float(overlaps.mean() / TOPK),
        },
        "lds": metric_results,
    }
    json_path = LDS_DIR / "joint_mucs_vs_original_das_lambda10_q00_q09.json"
    csv_path = LDS_DIR / "joint_mucs_vs_original_das_lambda10_q00_q09.csv"
    with open(json_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("metric", "method", "sign", "query_id", "spearman"),
        )
        writer.writeheader()
        writer.writerows(rows)

    print("[score agreement]", flush=True)
    print(
        f"  mean Spearman={score_rhos.mean():+.6f} | "
        f"top1000 overlap={overlaps.mean():.1f}/{TOPK} "
        f"({overlaps.mean()/TOPK:.2%})",
        flush=True,
    )
    for metric, result in metric_results.items():
        print(f"[{metric}]", flush=True)
        for label, _ in methods:
            saved = result[label]["saved_score"]["mean"]
            negated = result[label]["negated_score"]["mean"]
            print(
                f"  {label:20s} saved={saved:+.6f} negated={negated:+.6f}",
                flush=True,
            )
    print(f"[saved] {json_path}", flush=True)
    print(f"[saved] {csv_path}", flush=True)


if __name__ == "__main__":
    main()
