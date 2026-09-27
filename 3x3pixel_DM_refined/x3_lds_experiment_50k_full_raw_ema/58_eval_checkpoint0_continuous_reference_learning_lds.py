"""Evaluate raw and difference-over-sum checkpoint-0 continuous scores."""

import argparse
import json

import numpy as np

from checkpoint0_continuous_reference_learning_config import *
from checkpoint_counterfactual_metrics import spearman_correlation


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    all_methods = {}
    for method in C0RL_METHODS:
        predictions = np.empty(
            (len(CARL_QUERY_IDS), membership.shape[0]), dtype=np.float64
        )
        for position, query_id in enumerate(CARL_QUERY_IDS):
            path = ATTR_DIR / method / f"q{query_id:02d}" / "scores.npy"
            if not path.is_file():
                raise FileNotFoundError(path)
            score = np.load(path).astype(np.float64).reshape(-1)
            if score.shape != (N_TRAIN,):
                raise ValueError(f"{path} has shape {score.shape}")
            predictions[position] = membership @ score
        method_output = {"method": method, "results": {}}
        for metric in C0RL_LDS_METRICS:
            observed = np.load(LDS_DIR / f"observed_{metric}.npy").astype(
                np.float64
            )
            signs = {}
            for sign_name, multiplier in (
                ("saved_score", 1.0),
                ("negated_score", -1.0),
            ):
                rows = [
                    {
                        "query_id": query_id,
                        "spearman": spearman_correlation(
                            multiplier * predictions[position],
                            observed[query_id],
                        ),
                    }
                    for position, query_id in enumerate(CARL_QUERY_IDS)
                ]
                signs[sign_name] = {
                    "multiplier": multiplier,
                    "mean": float(
                        np.nanmean([row["spearman"] for row in rows])
                    ),
                    "queries": rows,
                }
            method_output["results"][metric] = signs
            best_name, best = max(
                signs.items(), key=lambda item: item[1]["mean"]
            )
            print(
                f"[LDS] {method} {metric}: {best_name}={best['mean']:+.6f}",
                flush=True,
            )
        output = LDS_DIR / f"{method}_both_signs.json"
        with open(output, "w") as handle:
            json.dump(method_output, handle, indent=2)
        all_methods[method] = method_output

    combined = (
        LDS_DIR
        / "checkpoint0_continuous_reference_learning_two_scores_both_signs_q00_q09.json"
    )
    with open(combined, "w") as handle:
        json.dump(
            {"query_ids": list(CARL_QUERY_IDS), "methods": all_methods},
            handle,
            indent=2,
        )
    print(f"[saved] {combined}", flush=True)


if __name__ == "__main__":
    main()
