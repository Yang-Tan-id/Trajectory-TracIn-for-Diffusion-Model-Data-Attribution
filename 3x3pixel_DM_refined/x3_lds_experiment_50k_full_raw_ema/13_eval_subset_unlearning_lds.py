"""Compute LDS for direct subset-level counterfactual predictions."""

import argparse
import json

import numpy as np

from checkpoint_counterfactual_config import *
from checkpoint_counterfactual_metrics import spearman_correlation
from exp_config import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-scale", type=float, default=CF_UPDATE_SCALE)
    parser.add_argument(
        "--final-source", choices=("raw", "ema"), default=CF_FINAL_PARAM_SOURCE
    )
    args = parser.parse_args()
    response_dir = (
        CF_RESPONSE_ROOT / f"source_{args.final_source}_scale_{args.update_scale:g}"
    )
    results = {}
    for response_name, observed_name in (
        ("endpoint_deviation", f"endpoint_deviation_{args.final_source}"),
        ("trajectory_state_mse", f"trajectory_state_mse_{args.final_source}"),
    ):
        observed = np.load(LDS_DIR / f"observed_{observed_name}.npy").astype(np.float64)
        query_results = []
        for query_id in range(observed.shape[0]):
            values = np.load(response_dir / f"q{query_id:02d}.npz")
            predicted = values[response_name].astype(np.float64)
            if predicted.shape != (observed.shape[1],):
                raise ValueError(f"q{query_id:02d} prediction shape {predicted.shape}")
            rho = spearman_correlation(predicted, observed[query_id])
            query_results.append({"query_id": query_id, "spearman": rho})
        mean = float(np.nanmean([item["spearman"] for item in query_results]))
        results[response_name] = {
            "observed_metric": observed_name,
            "mean": mean,
            "queries": query_results,
        }
        print(f"[LDS] {response_name}: mean={mean:.6f}", flush=True)

    output = response_dir / "lds.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as handle:
        json.dump(
            {
                "experiment": "subset_parameter_unlearning_counterfactual",
                "gradient_param_source": "raw",
                "final_param_source": args.final_source,
                "unlearn_set": CF_UNLEARN_SET,
                "update_scale": args.update_scale,
                "parameter_rule": "theta_cf = theta_final - mean_removed(sum_checkpoint,t[-lr*grad_loss]/100)",
                "results": results,
            },
            handle,
            indent=2,
        )
    print(f"[saved] {output}", flush=True)


if __name__ == "__main__":
    main()
