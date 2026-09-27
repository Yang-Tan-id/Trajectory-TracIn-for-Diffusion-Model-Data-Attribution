"""Evaluate all four q00-q19 probe12 Traj variants on existing LDS targets."""

import csv
import json

import numpy as np

from checkpoint_counterfactual_metrics import spearman_correlation
from traj_probe12_config import *


def main():
    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    observed = {
        metric: np.load(LDS_DIR / f"observed_{metric}.npy").astype(np.float64)
        for metric in LDS_METRICS
    }
    rows = []
    payload = {
        "query_ids": list(TRAJ_PROBE12_QUERY_IDS),
        "num_probes": TRAJ_PROBE12_NUM_PROBES,
        "methods": {},
    }
    for variant in TRAJ_PROBE12_VARIANTS:
        method = TRAJ_PROBE12_METHODS[variant]
        method_result = {
            "contraction": variant[0],
            "query_normalization": variant[1],
            "metrics": {},
        }
        predictions = {}
        for query_id in TRAJ_PROBE12_QUERY_IDS:
            path = ATTR_DIR / method / f"q{query_id:02d}" / "scores.npy"
            if not path.is_file():
                raise FileNotFoundError(path)
            scores = np.load(path).astype(np.float64).reshape(-1)
            if scores.shape != (N_TRAIN,):
                raise ValueError(f"{path}: shape={scores.shape}")
            predictions[query_id] = membership @ scores

        for metric, target in observed.items():
            signs = {}
            for sign_name, multiplier in (
                ("saved_score", 1.0),
                ("negated_score", -1.0),
            ):
                query_rows = []
                for query_id in TRAJ_PROBE12_QUERY_IDS:
                    rho = spearman_correlation(
                        multiplier * predictions[query_id], target[query_id]
                    )
                    query_rows.append({"query_id": query_id, "spearman": rho})
                    rows.append(
                        {
                            "method": method,
                            "contraction": variant[0],
                            "query_normalization": variant[1],
                            "metric": metric,
                            "sign": sign_name,
                            "query_id": query_id,
                            "spearman": rho,
                        }
                    )
                mean = float(
                    np.nanmean([item["spearman"] for item in query_rows])
                )
                signs[sign_name] = {
                    "multiplier": multiplier,
                    "mean": mean,
                    "queries": query_rows,
                }
            method_result["metrics"][metric] = signs
            print(
                f"[LDS] {method} {metric}: "
                f"saved={signs['saved_score']['mean']:+.6f} "
                f"negated={signs['negated_score']['mean']:+.6f}",
                flush=True,
            )
        payload["methods"][method] = method_result

    json_path = LDS_DIR / "traj_probe12_q00_q19_both_signs.json"
    csv_path = LDS_DIR / "traj_probe12_q00_q19_both_signs.csv"
    with open(json_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "method",
                "contraction",
                "query_normalization",
                "metric",
                "sign",
                "query_id",
                "spearman",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {json_path}", flush=True)
    print(f"[saved] {csv_path}", flush=True)


if __name__ == "__main__":
    main()
