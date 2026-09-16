#!/usr/bin/env python3
"""Print per-query LDS for timestamp-grouped checkpoint-square probe scores."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


SHAPES_ROOT = Path(__file__).resolve().parents[1]
TARGETS = (
    "endpoint_contarfactual",
    "noise_trajectory",
    "simple_loss",
    "traj_contarfactual",
)
VARIANTS = ("raw", "query_l2", "train_l2", "query_train_l2")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--num-probes", type=int, default=8)
    parser.add_argument("--prediction-sign", choices=("p1", "m1"), default="m1")
    parser.add_argument("--namespace-suffix", default="")
    parser.add_argument(
        "--reduction",
        choices=(
            "linear",
            "timestamp_checkpoint_square",
            "termwise_square",
            "signed_square",
            "absolute",
            "rms",
            "coordinate_square",
            "checkpoint_timestamp_square",
        ),
        default="timestamp_checkpoint_square",
    )
    args = parser.parse_args()
    if args.num_probes <= 0:
        raise ValueError("--num-probes must be positive")

    namespace_bases = {
        "linear": "traj_tracin_predicted_noise_jvp_signed",
        "timestamp_checkpoint_square": (
            "traj_tracin_predicted_noise_jvp_timestamp_checkpoint_sum_square"
        ),
        "termwise_square": "traj_tracin_predicted_noise_jvp_l2_squared",
        "signed_square": "traj_tracin_predicted_noise_jvp_signed_squared",
        "absolute": "traj_tracin_predicted_noise_jvp_absolute",
        "rms": "traj_tracin_predicted_noise_jvp_rms",
        "coordinate_square": "traj_tracin_predicted_noise_jvp_coordinatewise_squared",
        "checkpoint_timestamp_square": "traj_tracin_predicted_noise_jvp_checkpoint_timestamp_sum_square",
    }
    namespace_base = namespace_bases[args.reduction]
    namespace = f"{namespace_base}_probe{args.num_probes}"
    suffix = args.namespace_suffix.strip().strip("_/")
    if suffix:
        namespace = f"{namespace}_{suffix}"
    result_root = SHAPES_ROOT / "result" / args.experiment
    records = json.loads((SHAPES_ROOT / "queries_seed_0_9.json").read_text())["queries"]
    grouped: dict[tuple[str, str], list[float]] = {}

    titles = {
        "linear": "LINEAR TERM SUM (NO SQUARE)",
        "timestamp_checkpoint_square": "TIMESTAMP-GROUPED CHECKPOINT-SUM SQUARE",
        "termwise_square": "TERMWISE GRADIENT-PRODUCT SQUARE",
        "signed_square": "SIGNED SQUARE z*abs(z)",
        "absolute": "TERMWISE ABSOLUTE CONTRACTION |z|",
        "rms": "TERMWISE PROBE RMS sqrt(mean_r(z_r^2))",
        "coordinate_square": "COORDINATEWISE PRODUCT SQUARE",
        "checkpoint_timestamp_square": "CHECKPOINT-GROUPED TIMESTAMP-MEAN SQUARE",
    }
    title = titles[args.reduction]
    print(f"{title} ({args.num_probes} probes, sign {args.prediction_sign})")
    print(
        f"{'TARGET':24s} {'Q':>2s} {'RAW':>9s} {'QUERY-L2':>9s} "
        f"{'TRAIN-L2':>9s} {'BOTH-L2':>9s}  PROMPT"
    )
    print("-" * 132)

    for query_id, record in enumerate(records):
        prompt_tag = str(record["prompt"]).replace(",", "_")
        eval_root = (
            result_root
            / "eval"
            / "prompted_solo"
            / f"query_{prompt_tag}"
            / f"initial_seed_{int(record['initial_seed'])}"
        )
        for target in TARGETS:
            row = []
            for variant in VARIANTS:
                matches = list(
                    (
                        eval_root
                        / "lds"
                        / f"{namespace}_{variant}"
                        / target
                        / f"pred_kept_sign_{args.prediction_sign}"
                    ).glob("*/lds_summary.json")
                )
                if len(matches) != 1:
                    raise RuntimeError(
                        f"expected one LDS summary for query={query_id}, "
                        f"target={target}, variant={variant}; found {len(matches)}"
                    )
                value = float(json.loads(matches[0].read_text())["lds_percent"])
                row.append(value)
                grouped.setdefault((target, variant), []).append(value)
            print(
                f"{target:24s} {query_id:2d} "
                + " ".join(f"{value:8.3f}%" for value in row)
                + f"  {prompt_tag}"
            )

    print("\n10-query mean")
    print(
        f"{'TARGET':24s} {'RAW':>9s} {'QUERY-L2':>9s} "
        f"{'TRAIN-L2':>9s} {'BOTH-L2':>9s}"
    )
    print("-" * 68)
    for target in TARGETS:
        means = [
            statistics.mean(grouped[(target, variant)])
            for variant in VARIANTS
        ]
        print(f"{target:24s} " + " ".join(f"{value:8.3f}%" for value in means))


if __name__ == "__main__":
    main()
