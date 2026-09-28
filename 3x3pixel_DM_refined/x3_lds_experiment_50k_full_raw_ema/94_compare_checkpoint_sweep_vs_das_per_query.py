"""Write a readable per-query checkpoint-count sweep versus DAS comparison."""

import argparse
import json
from pathlib import Path

import numpy as np

from tracin_das_config import *
from importlib.util import module_from_spec, spec_from_file_location


def load_report_helpers():
    path = Path(__file__).with_name("93_compare_timestamp_sweep_vs_das_per_query.py")
    spec = spec_from_file_location("timestamp_sweep_report", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    helpers = load_report_helpers()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric",
        choices=tuple(helpers.TARGET_BY_METRIC),
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

    sweep_path = LDS_DIR / "tracin_das_interval_mean_lr_99q_checkpoint_sweep.json"
    with open(sweep_path) as handle:
        result = json.load(handle)
    membership = np.load(MASK_DIR / "membership.npy").astype(np.float64)
    observed = np.load(LDS_DIR / f"observed_{args.metric}.npy").astype(np.float64)
    target = helpers.TARGET_BY_METRIC[args.metric]
    das = helpers.select_global_das(membership, observed, args.das_method)
    ours = {
        count: helpers.select_ours(
            result, count, args.contraction, target, args.metric
        )
        for count in TRACIN_DAS_AVG_LR_CHECKPOINT_COUNTS
    }
    for item in ours.values():
        item["std"] = float(np.nanstd(item["per_query"]))

    output = args.output or (
        LDS_DIR
        / (
            f"checkpoint_sweep_{args.contraction}_{args.metric}_vs_"
            f"{args.das_method}_per_query.txt"
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "TracIn-DAS checkpoint-count sweep versus DAS: per-query LDS",
        f"metric       : {args.metric}",
        f"contraction  : {args.contraction}",
        "LR source    : interval mean LR (as stored in this checkpoint sweep)",
        f"DAS method   : {args.das_method}",
        f"DAS lambda   : {das['lambda']:g}",
        f"DAS sign     : {das['sign_name']}",
        f"DAS mean LDS : {das['mean']:+.6f}",
        f"DAS std LDS  : {das['std']:.6f}",
        "Selection    : global across q00-q98; never per-query",
        "Improvement  : ours - DAS",
        "STD          : population std across q00-q98 (ddof=0)",
        "",
        "SUMMARY",
        "checkpoints  sign       ours_mean   ours_std    DAS_mean    DAS_std     imp_mean     imp_std",
    ]
    for count in TRACIN_DAS_AVG_LR_CHECKPOINT_COUNTS:
        item = ours[count]
        improvements = item["per_query"] - das["per_query"]
        lines.append(
            f"{count:11d}  {item['sign_name']:<8s}  "
            f"{item['mean']:+.6f}  {item['std']:.6f}  "
            f"{das['mean']:+.6f}  {das['std']:.6f}  "
            f"{float(np.nanmean(improvements)):+.6f}  "
            f"{float(np.nanstd(improvements)):.6f}"
        )

    for count in TRACIN_DAS_AVG_LR_CHECKPOINT_COUNTS:
        item = ours[count]
        improvements = item["per_query"] - das["per_query"]
        lines.extend(
            [
                "",
                "=" * 76,
                f"CHECKPOINTS={count} | ours sign={item['sign_name']} | "
                f"DAS lambda={das['lambda']:g} sign={das['sign_name']}",
                "=" * 76,
                "query       DAS_LDS      OURS_LDS    improvement    status",
            ]
        )
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
