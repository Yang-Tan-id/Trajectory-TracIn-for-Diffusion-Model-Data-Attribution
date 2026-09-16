#!/usr/bin/env python3
"""Evaluate cached LDS independently for every retained checkpoint score."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
REFINE_ROOT = SHAPES_ROOT.parent
for path in (SHAPES_ROOT, REFINE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataset_config import _prompt_tag
from run_predicted_noise_jvp_l2_squared import shard_root
from run_traj_tracin_lds_cached import TARGETS, cache_group, materialize_target_csv


VARIANTS = (
    ("raw", "score"),
    ("query_l2", "score_query_normalized"),
    ("train_l2", "score_train_l2_normalized"),
    ("query_train_l2", "score_query_train_l2_normalized"),
)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def subset_matrix(rows: list[dict], score_indices: np.ndarray) -> np.ndarray:
    positions = {int(index): column for column, index in enumerate(score_indices)}
    matrix = np.zeros((len(rows), len(score_indices)), dtype=np.float32)
    for row_index, row in enumerate(rows):
        kept = np.asarray(
            np.load(Path(row["subset_dir"]) / "kept_attribution_indices.npy"),
            dtype=np.int64,
        )
        columns = [positions[int(index)] for index in kept if int(index) in positions]
        matrix[row_index, columns] = 1.0
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--num-probes", type=int, default=8)
    parser.add_argument(
        "--contraction",
        choices=("checkpoint_timestamp_sum_square", "squared"),
        default="checkpoint_timestamp_sum_square",
    )
    parser.add_argument("--prediction-sign", type=float, choices=(-1.0, 1.0), default=-1.0)
    args = parser.parse_args()

    legacy_root = REFINE_ROOT / "legacy_jax"
    if str(legacy_root) not in sys.path:
        sys.path.insert(0, str(legacy_root))
    from LDS.DM_cifar_lds import spearman_corr

    source = (
        shard_root(
            args.experiment,
            args.train_seed,
            args.run_id,
            args.num_probes,
            args.contraction,
            "",
        )
        / "per_checkpoint_scores.npz"
    )
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as payload:
        scores = {
            variant: np.asarray(payload[f"checkpoint_scores_{component}"], dtype=np.float64)
            for variant, component in VARIANTS
        }
        score_indices = np.asarray(payload["score_indices"], dtype=np.int64)
        checkpoint_indices = np.asarray(payload["checkpoint_indices"], dtype=np.int32)
    expected_shape = (50, 10, 5000)
    for variant, values in scores.items():
        if values.shape != expected_shape:
            raise ValueError(f"{variant} expected {expected_shape}, got {values.shape}")

    records = json.loads((SHAPES_ROOT / "queries_seed_0_9.json").read_text())["queries"]
    result_root = SHAPES_ROOT / "result" / args.experiment
    reduction_tag = (
        "checkpoint_timestamp_mean_square"
        if args.contraction == "checkpoint_timestamp_sum_square"
        else "termwise_product_square"
    )
    output = (
        result_root
        / "eval"
        / f"probe{args.num_probes}_{reduction_tag}_per_checkpoint"
        / f"run_{args.run_id}"
    )
    rows: list[dict] = []

    for query_id, record in enumerate(records):
        prompt = str(record["prompt"])
        seed = int(record["initial_seed"])
        eval_root = (
            result_root
            / "eval"
            / "prompted_solo"
            / f"query_{_prompt_tag(prompt)}"
            / f"initial_seed_{seed}"
        )
        group = cache_group(eval_root)
        target_rows = {
            target: list(
                csv.DictReader(
                    materialize_target_csv(group / target).open(newline="")
                )
            )
            for target in TARGETS
        }
        first_target_rows = target_rows[TARGETS[0]]
        kept_matrix = subset_matrix(first_target_rows, score_indices)

        for variant, values in scores.items():
            predictions = (
                args.prediction_sign
                * (values[:, query_id, :] @ kept_matrix.T)
            )
            for target in TARGETS:
                true = np.asarray(
                    [float(row["true_f"]) for row in target_rows[target]],
                    dtype=np.float64,
                )
                for checkpoint_slot, checkpoint_index in enumerate(checkpoint_indices):
                    lds = spearman_corr(predictions[checkpoint_slot], true)
                    rows.append(
                        {
                            "checkpoint": int(checkpoint_index) + 1,
                            "epoch": 4 * (int(checkpoint_index) + 1),
                            "query": query_id,
                            "initial_seed": seed,
                            "target": target,
                            "variant": variant,
                            "lds_percent": 100.0 * lds if math.isfinite(lds) else float("nan"),
                            "prompt": prompt,
                        }
                    )
        print(f"[query {query_id}/9] evaluated 50 checkpoints", flush=True)

    write_rows(output / "per_query_checkpoint_lds.csv", rows)
    summary_rows = []
    for checkpoint in range(1, 51):
        for target in TARGETS:
            for variant, _ in VARIANTS:
                values = [
                    float(row["lds_percent"])
                    for row in rows
                    if row["checkpoint"] == checkpoint
                    and row["target"] == target
                    and row["variant"] == variant
                    and math.isfinite(float(row["lds_percent"]))
                ]
                summary_rows.append(
                    {
                        "checkpoint": checkpoint,
                        "epoch": 4 * checkpoint,
                        "target": target,
                        "variant": variant,
                        "n": len(values),
                        "mean_lds_percent": statistics.mean(values) if values else float("nan"),
                        "std_lds_percent": statistics.stdev(values) if len(values) > 1 else 0.0,
                    }
                )
    write_rows(output / "checkpoint_summary.csv", summary_rows)

    print(f"\nBOTH-L2 — {reduction_tag} — 10-query mean LDS per checkpoint")
    print(f"{'CKPT':>4s} {'EPOCH':>5s} {'ENDPOINT':>10s} {'TRAJ':>10s} {'CF JOINT':>10s} {'NOISE':>10s} {'SIMPLE':>10s}")
    print("-" * 78)
    for checkpoint in range(1, 51):
        means = {
            row["target"]: float(row["mean_lds_percent"])
            for row in summary_rows
            if row["checkpoint"] == checkpoint and row["variant"] == "query_train_l2"
        }
        joint = 0.5 * (
            means["endpoint_contarfactual"] + means["traj_contarfactual"]
        )
        print(
            f"{checkpoint:4d} {4 * checkpoint:5d} "
            f"{means['endpoint_contarfactual']:9.3f}% "
            f"{means['traj_contarfactual']:9.3f}% "
            f"{joint:9.3f}% "
            f"{means['noise_trajectory']:9.3f}% "
            f"{means['simple_loss']:9.3f}%"
        )
    print(f"[saved] {output / 'per_query_checkpoint_lds.csv'}")
    print(f"[saved] {output / 'checkpoint_summary.csv'}")


if __name__ == "__main__":
    main()
