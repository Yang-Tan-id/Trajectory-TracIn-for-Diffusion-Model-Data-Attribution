"""Print per-query LDS for aligned DAS λ=100 versus legacy aligned MC1."""

import json

import numpy as np
from scipy.stats import spearmanr

from tracin_das_config import *


QUERY_IDS = tuple(range(75))
DAS_METHOD = "das_ema_aligned_noise"
DAS_LAMBDA = 100.0
TARGETS = {
    "simple_loss": ("simple_loss_ema", "simple_loss_raw"),
    "traj_ref": ("traj_ref_ema", "traj_ref_raw"),
    "endpoint_deviation": ("endpoint_deviation_ema", "endpoint_deviation_raw"),
    "trajectory_state_mse": ("trajectory_state_mse_ema", "trajectory_state_mse_raw"),
}


def lambda_tag(value):
    return str(float(value)).replace(".", "p")


def load_method_scores(method):
    return np.stack(
        [
            np.load(ATTR_DIR / method / f"q{query_id:02d}" / "scores.npy").astype(
                np.float64
            )
            for query_id in QUERY_IDS
        ],
        axis=0,
    )


def correlations(scores, membership, observed):
    predictions = membership @ scores.T
    positive = np.asarray(
        [
            spearmanr(predictions[:, position], observed[query_id]).statistic
            for position, query_id in enumerate(QUERY_IDS)
        ],
        dtype=np.float64,
    )
    return {"negative": -positive, "positive": positive}


def describe(values):
    return {
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
        "per_query": values.tolist(),
    }


def main():
    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    das_scores = np.stack(
        [
            np.load(
                ATTR_DIR
                / DAS_METHOD
                / f"q{query_id:02d}"
                / f"lambda_{lambda_tag(DAS_LAMBDA)}"
                / "scores.npy"
            ).astype(np.float64)
            for query_id in QUERY_IDS
        ],
        axis=0,
    )
    old_methods = tracin_das_methods(
        "checkpoint", "projected4096", "aligned", 1
    )
    output = {
        "query_ids": list(QUERY_IDS),
        "query_count": len(QUERY_IDS),
        "das": {"method": DAS_METHOD, "lambda": DAS_LAMBDA},
        "old_mc1": {
            "query_mc": 1,
            "train_mc": 1,
            "query_train_noise_aligned": True,
            "checkpoint_count": 50,
            "transition_count": 49,
            "timestamp_count": 100,
            "parameter_projection": "projected4096",
        },
        "results": {},
    }
    lines = [
        "q00-q74: aligned DAS lambda=100 versus legacy aligned TracIn-DAS MC1",
        "main per-query table uses sign=-1",
        "delta = old_MC1 - DAS_lambda100",
        "STD = population std across q00-q74 (ddof=0)",
    ]

    for contraction in ("linear", "termwise_squared", "timestamp_sum_squared"):
        old_method = old_methods[contraction]
        old_scores = load_method_scores(old_method)
        contraction_result = {"old_method": old_method, "targets": {}}
        lines.extend(["", "=" * 104, f"CONTRACTION: {contraction}"])
        for target_name, metrics in TARGETS.items():
            target_result = {}
            for metric in metrics:
                observed = np.load(LDS_DIR / f"observed_{metric}.npy").astype(
                    np.float64
                )
                das = correlations(das_scores, membership, observed)
                old = correlations(old_scores, membership, observed)
                metric_result = {}
                lines.extend(["", f"  METRIC: {metric}"])
                for sign_name in ("negative", "positive"):
                    delta = old[sign_name] - das[sign_name]
                    metric_result[sign_name] = {
                        "das": describe(das[sign_name]),
                        "old_mc1": describe(old[sign_name]),
                        "delta": describe(delta),
                        "old_better_queries": int(np.sum(delta > 0)),
                        "das_better_queries": int(np.sum(delta < 0)),
                    }
                    lines.append(
                        f"    sign={sign_name:<8s} "
                        f"DAS={np.nanmean(das[sign_name]):+.6f}±{np.nanstd(das[sign_name]):.6f}  "
                        f"old_MC1={np.nanmean(old[sign_name]):+.6f}±{np.nanstd(old[sign_name]):.6f}  "
                        f"delta={np.nanmean(delta):+.6f}±{np.nanstd(delta):.6f}"
                    )
                target_result[metric] = metric_result

                negative_delta = old["negative"] - das["negative"]
                lines.append(
                    "    qid      DAS100(-1)    old_MC1(-1)    old-DAS      winner"
                )
                for position, query_id in enumerate(QUERY_IDS):
                    winner = (
                        "old_MC1"
                        if negative_delta[position] > 0
                        else "DAS100"
                        if negative_delta[position] < 0
                        else "tie"
                    )
                    lines.append(
                        f"    q{query_id:02d}   {das['negative'][position]:+11.6f}  "
                        f"{old['negative'][position]:+12.6f}  "
                        f"{negative_delta[position]:+10.6f}  {winner}"
                    )
            contraction_result["targets"][target_name] = target_result
        output["results"][contraction] = contraction_result

    json_path = LDS_DIR / "das_aligned_lambda100_vs_aligned_mc1_q00_q74.json"
    text_path = LDS_DIR / "das_aligned_lambda100_vs_aligned_mc1_q00_q74.txt"
    with open(json_path, "w") as handle:
        json.dump(output, handle, indent=2)
    text_path.write_text("\n".join(lines) + "\n")
    print(f"[saved] {json_path}")
    print(f"[saved] {text_path}")


if __name__ == "__main__":
    main()
