"""Write a readable per-query timestamp-sweep versus DAS LDS comparison."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from tracin_das_config import *


TARGET_BY_METRIC = {
    "simple_loss_ema": "simple_loss",
    "simple_loss_raw": "simple_loss",
    "traj_ref_ema": "traj_ref",
    "traj_ref_raw": "traj_ref",
    "endpoint_deviation_ema": "endpoint_deviation",
    "endpoint_deviation_raw": "endpoint_deviation",
    "trajectory_state_mse_ema": "trajectory_state_mse",
    "trajectory_state_mse_raw": "trajectory_state_mse",
}


def lambda_tag(value):
    return str(float(value)).replace(".", "p")


def select_global_das(membership, observed, method):
    candidates = []
    for lam_raw in DAS_LAMBDAS:
        lam = float(lam_raw)
        scores = np.stack(
            [
                np.load(
                    ATTR_DIR
                    / method
                    / f"q{query_id:02d}"
                    / f"lambda_{lambda_tag(lam)}"
                    / "scores.npy"
                ).astype(np.float64)
                for query_id in TRACIN_DAS_FIRST99_QUERY_IDS
            ],
            axis=0,
        )
        predictions = membership @ scores.T
        positive = np.asarray(
            [
                spearmanr(predictions[:, query_id], observed[query_id]).statistic
                for query_id in TRACIN_DAS_FIRST99_QUERY_IDS
            ],
            dtype=np.float64,
        )
        for sign_name, sign, values in (
            ("positive", 1.0, positive),
            ("negative", -1.0, -positive),
        ):
            candidates.append(
                {
                    "lambda": lam,
                    "sign_name": sign_name,
                    "sign": sign,
                    "mean": float(np.nanmean(values)),
                    "std": float(np.nanstd(values)),
                    "per_query": values,
                }
            )
    return max(candidates, key=lambda item: item["mean"])


def select_ours(result, count, contraction, target, metric):
    entry = result["results"][str(count)]
    selected = None
    for value in entry["methods"].values():
        if value["contraction"] == contraction:
            selected = value["targets"][target][metric]
            break
    if selected is None:
        raise KeyError((count, contraction, target, metric))
    sign_name = max(
        ("negative", "positive"),
        key=lambda name: float(selected[name]["mean"]),
    )
    return {
        "sign_name": sign_name,
        "mean": float(selected[sign_name]["mean"]),
        "per_query": np.asarray(selected[sign_name]["per_query"], dtype=np.float64),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric",
        choices=tuple(TARGET_BY_METRIC),
        default="endpoint_deviation_ema",
    )
    parser.add_argument(
        "--contraction",
        choices=("linear", "termwise_squared", "timestamp_sum_squared"),
        default="timestamp_sum_squared",
    )
    parser.add_argument(
        "--das-method",
        choices=("das_ema", "das_ema_aligned_noise"),
        default="das_ema",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    sweep_path = LDS_DIR / "tracin_das_checkpoint_lr_99q_timestamp_sweep.json"
    with open(sweep_path) as handle:
        result = json.load(handle)
    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    observed = np.load(LDS_DIR / f"observed_{args.metric}.npy").astype(np.float64)
    target = TARGET_BY_METRIC[args.metric]
    das = select_global_das(membership, observed, args.das_method)
    ours = {
        count: select_ours(
            result, count, args.contraction, target, args.metric
        )
        for count in TRACIN_DAS_TIMESTAMP_COUNTS
    }
    for item in ours.values():
        item["std"] = float(np.nanstd(item["per_query"]))

    output = args.output or (
        LDS_DIR
        / (
            f"timestamp_sweep_{args.contraction}_{args.metric}_vs_"
            f"{args.das_method}_per_query.txt"
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "TracIn-DAS timestamp sweep versus DAS: per-query LDS",
        f"metric       : {args.metric}",
        f"contraction  : {args.contraction}",
        f"DAS method   : {args.das_method}",
        f"DAS lambda   : {das['lambda']:g}",
        f"DAS sign     : {das['sign_name']}",
        f"DAS mean LDS : {das['mean']:+.6f}",
        f"DAS std LDS  : {das['std']:.6f}",
        "Selection    : global across q00-q98; never per-query",
        "Improvement  : ours - DAS",
        "",
        "SUMMARY",
        "timestamps  sign       ours_mean   ours_std    DAS_mean    DAS_std     imp_mean     imp_std",
    ]
    for count in TRACIN_DAS_TIMESTAMP_COUNTS:
        item = ours[count]
        improvements = item["per_query"] - das["per_query"]
        lines.append(
            f"{count:10d}  {item['sign_name']:<8s}  "
            f"{item['mean']:+.6f}  {item['std']:.6f}  "
            f"{das['mean']:+.6f}  {das['std']:.6f}  "
            f"{float(np.nanmean(improvements)):+.6f}  "
            f"{float(np.nanstd(improvements)):.6f}"
        )

    for count in TRACIN_DAS_TIMESTAMP_COUNTS:
        item = ours[count]
        lines.extend(
            [
                "",
                "=" * 76,
                f"TIMESTAMPS={count} | ours sign={item['sign_name']} | "
                f"DAS lambda={das['lambda']:g} sign={das['sign_name']}",
                "=" * 76,
                "query       DAS_LDS      OURS_LDS    improvement    status",
            ]
        )
        improvements = item["per_query"] - das["per_query"]
        for query_id in TRACIN_DAS_FIRST99_QUERY_IDS:
            delta = improvements[query_id]
            status = "better" if delta > 0 else "worse" if delta < 0 else "same"
            lines.append(
                f"q{query_id:02d}     {das['per_query'][query_id]:+10.6f}  "
                f"{item['per_query'][query_id]:+10.6f}  "
                f"{delta:+11.6f}    {status}"
            )
        lines.append(
            f"MEAN    {das['mean']:+10.6f}  {item['mean']:+10.6f}  "
            f"{float(np.nanmean(improvements)):+11.6f}"
        )
        lines.append(
            f"STD     {das['std']:10.6f}  {item['std']:10.6f}  "
            f"{float(np.nanstd(improvements)):11.6f}"
        )
        lines.append(
            f"better queries: {int(np.sum(improvements > 0))}/99 | "
            f"worse queries: {int(np.sum(improvements < 0))}/99"
        )

    output.write_text("\n".join(lines) + "\n")
    print(f"[saved] {output}")


if __name__ == "__main__":
    main()
