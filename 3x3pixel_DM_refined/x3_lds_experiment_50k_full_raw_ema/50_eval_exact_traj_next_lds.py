"""Evaluate all three true full-gradient next-checkpoint Traj contractions."""

import json

import numpy as np

from checkpoint_counterfactual_metrics import spearman_correlation
from exp_config import ATTR_DIR, LDS_DIR, LDS_METRICS, MASK_DIR, N_TRAIN
from run_exact_traj_next_bank import EXACT_METHODS


def main():
    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    observed = {
        metric: np.load(LDS_DIR / f"observed_{metric}.npy").astype(np.float64)
        for metric in LDS_METRICS
    }
    query_count = next(iter(observed.values())).shape[0]
    summary = {
        "description": "50 checkpoints / 49 next transitions / exact full dot",
        "prediction": "-(membership @ score)",
        "query_count": query_count,
        "methods": {},
    }

    for contraction, method in EXACT_METHODS.items():
        predictions = np.empty(
            (query_count, membership.shape[0]), dtype=np.float64
        )
        for query_id in range(query_count):
            path = ATTR_DIR / method / f"q{query_id:02d}" / "scores.npy"
            if not path.is_file():
                raise FileNotFoundError(path)
            score = np.load(path).astype(np.float64).reshape(-1)
            if score.shape != (N_TRAIN,):
                raise ValueError(f"{path} has shape {score.shape}")
            if not np.isfinite(score).all():
                raise ValueError(f"non-finite score in {path}")
            predictions[query_id] = -(membership @ score)

        method_results = {}
        for metric, target in observed.items():
            query_rows = [
                {
                    "query_id": query_id,
                    "spearman": spearman_correlation(
                        predictions[query_id], target[query_id]
                    ),
                }
                for query_id in range(query_count)
            ]
            payload = {
                "method": method,
                "contraction": contraction,
                "metric": metric,
                "prediction": "-(membership @ score)",
                "mean": float(
                    np.nanmean([row["spearman"] for row in query_rows])
                ),
                "queries": query_rows,
            }
            output = LDS_DIR / f"{method}_{metric}.json"
            with open(output, "w") as handle:
                json.dump(payload, handle, indent=2)
            method_results[metric] = {
                "mean": payload["mean"],
                "output": str(output),
            }
        summary["methods"][method] = method_results
        print(
            f"[LDS] {contraction}: "
            f"traj_ref_raw={method_results['traj_ref_raw']['mean']:+.6f}",
            flush=True,
        )

    summary_path = LDS_DIR / "traj_exact_first_raw_three_contractions_lds.json"
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[saved] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
