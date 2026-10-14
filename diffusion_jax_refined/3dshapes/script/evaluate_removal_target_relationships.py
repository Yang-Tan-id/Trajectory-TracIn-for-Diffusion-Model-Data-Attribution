#!/usr/bin/env python3
"""Evaluate and relate four targets for completed top-k removal checkpoints.

This script never trains.  It scans atomic ``counterfactual_metrics.json``
files produced by the removal job, evaluates only simple loss and predicted
noise deviation for those completed checkpoints, and writes one sidecar JSON
per run.  A separate summarize mode produces CSV, pairwise correlations, and
scatter plots, making the evaluation safely restartable and shardable on CPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
REFINE_ROOT = SHAPES_ROOT.parent
LEGACY_ROOT = REFINE_ROOT / "legacy_jax"
for path in (SHAPES_ROOT, REFINE_ROOT, LEGACY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataset_config import _prompt_tag


METHOD = "endpoint_tracin_adamw_full_100x1_raw_neg"
SIDECAR_NAME = "four_target_relationship_metrics.json"
TARGET_COLUMNS = (
    "endpoint_difference",
    "trajectory_difference",
    "noise_trajectory",
    "simple_loss",
    "simple_loss_delta",
    "abs_simple_loss_delta",
)


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value]


def load_queries(path: Path) -> dict[int, dict]:
    payload = json.loads(path.read_text())
    records = payload["queries"] if isinstance(payload, dict) else payload
    return {int(record.get("query_id", index)): record for index, record in enumerate(records)}


def sample_root(result_root: Path, record: dict, train_seed: int, epochs: int) -> Path:
    return (
        result_root
        / "sample_ddim_eta0_1000"
        / "cifar"
        / f"prompt_{_prompt_tag(str(record['prompt']))}"
        / f"model_prompted_solo__ckpt_seed_{train_seed}_epoch_{epochs:04d}"
    )


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def completed_metric_paths(root: Path, query_ids: set[int]) -> list[Path]:
    paths = []
    method_root = root / METHOD
    for path in sorted(method_root.glob("topk_*/query_*_seed_*/counterfactual_metrics.json")):
        try:
            payload = json.loads(path.read_text())
            query = int(payload["query_id"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if query in query_ids:
            paths.append(path)
    return paths


def selected_metric_paths(args: argparse.Namespace, root: Path, query_ids: set[int]) -> list[Path]:
    if args.metrics_manifest is None:
        return completed_metric_paths(root, query_ids)
    paths = [Path(line) for line in args.metrics_manifest.read_text().splitlines() if line.strip()]
    selected = []
    for path in paths:
        payload = json.loads(path.read_text())
        if int(payload["query_id"]) in query_ids:
            selected.append(path)
    return sorted(selected)


def evaluate(args: argparse.Namespace) -> None:
    records = load_queries(args.query_file)
    selected_ids = set(parse_ints(args.query_ids))
    unknown = sorted(selected_ids - records.keys())
    if unknown:
        raise KeyError(f"query IDs absent from query file: {unknown}")
    result_root = SHAPES_ROOT / "result" / args.experiment
    removal_root = result_root / "retrac_endpoint_topk_removal_60q"
    inputs = selected_metric_paths(args, removal_root, selected_ids)
    assigned = [path for index, path in enumerate(inputs) if index % args.shard_count == args.shard_index]
    print(
        f"[scan] completed={len(inputs)} assigned={len(assigned)} "
        f"shard={args.shard_index}/{args.shard_count}",
        flush=True,
    )
    if not inputs:
        raise RuntimeError(f"no completed removal metrics found under {removal_root / METHOD}")

    base_checkpoint = (
        result_root
        / "model"
        / "prompted_jax"
        / f"seed_{args.train_seed}_epoch_{args.epochs:04d}.ckpt"
    )
    dataset_root = REFINE_ROOT / "dataset" / "3dshapes" / "20000"
    if not base_checkpoint.is_file():
        raise FileNotFoundError(base_checkpoint)

    # Imported after JAX_PLATFORMS=cpu is established by the launcher.
    from LDS.DM_cifar_lds import CifarTargetEvaluator

    started = time.time()
    for position, metrics_path in enumerate(assigned, start=1):
        sidecar = metrics_path.parent / SIDECAR_NAME
        if sidecar.is_file() and not args.overwrite:
            print(f"[skip {position}/{len(assigned)}] {sidecar}", flush=True)
            continue
        source = json.loads(metrics_path.read_text())
        query = int(source["query_id"])
        record = records[query]
        checkpoint = Path(source["removal_checkpoint"])
        if not checkpoint.is_file():
            # Permit relocation when the saved absolute path came from the same
            # repository on another login host.
            checkpoint = metrics_path.parent / "model" / f"seed_{args.train_seed}_epoch_{args.epochs:04d}.ckpt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

        evaluator = CifarTargetEvaluator(
            code_file=str(LEGACY_ROOT / "DM__training_3DSHAPES_pixel.py"),
            base_checkpoint=str(base_checkpoint),
            prompt=str(record["prompt"]),
            prefer_device="cpu",
            data_root=str(dataset_root),
            target_function="noise_trajectory",
            sample_root=str(sample_root(result_root, record, args.train_seed, args.epochs)),
            sample_seed=int(record["initial_seed"]),
            sample_index=0,
            max_trajectory_steps=None,
            trajectory_reduction="mean",
            trajectory_projection=None,
            simple_loss_timesteps=[0],
            simple_loss_noise_seeds=None,
            simple_loss_num_mc=1,
            simple_loss_mc_seed=0,
            trajectory_sampler="ddim_eta0",
        )
        removal_values = evaluator.evaluate_many(
            str(checkpoint), ("noise_trajectory", "simple_loss")
        )
        base_values = evaluator.evaluate_many(str(base_checkpoint), ("simple_loss",))
        simple_loss = float(removal_values["simple_loss"][0])
        base_simple_loss = float(base_values["simple_loss"][0])
        output = {
            "method": METHOD,
            "topk": int(source["topk"]),
            "query_id": query,
            "initial_seed": int(source["initial_seed"]),
            "prompt": str(source["prompt"]),
            "removal_checkpoint": str(checkpoint.resolve()),
            "counterfactual_metrics_source": str(metrics_path.resolve()),
            "endpoint_difference": float(source["endpoint_difference"]),
            "trajectory_difference": float(source["trajectory_difference"]),
            "noise_trajectory": float(removal_values["noise_trajectory"][0]),
            "simple_loss": simple_loss,
            "base_simple_loss": base_simple_loss,
            "simple_loss_delta": simple_loss - base_simple_loss,
            "abs_simple_loss_delta": abs(simple_loss - base_simple_loss),
            "target_details": {
                "noise_trajectory": removal_values["noise_trajectory"][1],
                "simple_loss": removal_values["simple_loss"][1],
                "base_simple_loss": base_values["simple_loss"][1],
            },
        }
        atomic_json(sidecar, output)
        elapsed = time.time() - started
        eta = elapsed / position * (len(assigned) - position)
        print(
            f"[done {position}/{len(assigned)}] topk={output['topk']} Q{query} "
            f"noise={output['noise_trajectory']:.6g} "
            f"simple_delta={output['simple_loss_delta']:+.6g} "
            f"eta={eta / 60:.1f}m",
            flush=True,
        )


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def correlation(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 2:
        return math.nan
    if kind == "spearman":
        x = average_ranks(x)
        y = average_ranks(y)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def summarize(args: argparse.Namespace) -> None:
    selected_ids = set(parse_ints(args.query_ids))
    result_root = SHAPES_ROOT / "result" / args.experiment
    root = result_root / "retrac_endpoint_topk_removal_60q" / METHOD
    rows = []
    missing = []
    for metrics_path in selected_metric_paths(args, root.parent, selected_ids):
        sidecar = metrics_path.parent / SIDECAR_NAME
        if not sidecar.is_file():
            missing.append(sidecar)
            continue
        rows.append(json.loads(sidecar.read_text()))
    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        raise RuntimeError(f"missing {len(missing)} relationship sidecars:\n{preview}")
    if len(rows) < 2:
        raise RuntimeError(f"need at least two evaluated checkpoints, found {len(rows)}")
    rows.sort(key=lambda row: (int(row["topk"]), int(row["query_id"])))

    output_root = root.parent / f"{METHOD}_four_target_relationships"
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "four_target_values.csv"
    fieldnames = (
        "method", "topk", "query_id", "initial_seed", "prompt",
        "endpoint_difference", "trajectory_difference", "noise_trajectory",
        "simple_loss", "base_simple_loss", "simple_loss_delta",
        "abs_simple_loss_delta", "removal_checkpoint",
    )
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    groups: list[tuple[str, list[dict]]] = [("all", rows)]
    for topk in sorted({int(row["topk"]) for row in rows}):
        selected = [row for row in rows if int(row["topk"]) == topk]
        if len(selected) >= 2:
            groups.append((f"topk_{topk}", selected))
    correlation_rows = []
    for group_name, group_rows in groups:
        arrays = {
            column: np.asarray([float(row[column]) for row in group_rows], dtype=np.float64)
            for column in TARGET_COLUMNS
        }
        for left_index, left in enumerate(TARGET_COLUMNS):
            for right in TARGET_COLUMNS[left_index + 1:]:
                for kind in ("pearson", "spearman"):
                    correlation_rows.append(
                        {
                            "group": group_name,
                            "n": len(group_rows),
                            "left": left,
                            "right": right,
                            "correlation": kind,
                            "value": correlation(arrays[left], arrays[right], kind),
                        }
                    )
    correlation_path = output_root / "pairwise_correlations.csv"
    with correlation_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(correlation_rows[0]))
        writer.writeheader()
        writer.writerows(correlation_rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = (
        ("noise_trajectory", "endpoint_difference"),
        ("noise_trajectory", "trajectory_difference"),
        ("noise_trajectory", "simple_loss"),
        ("noise_trajectory", "simple_loss_delta"),
        ("noise_trajectory", "abs_simple_loss_delta"),
        ("endpoint_difference", "trajectory_difference"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), squeeze=False)
    topks = sorted({int(row["topk"]) for row in rows})
    colors = plt.get_cmap("tab10")
    for axis, (left, right) in zip(axes.ravel(), pairs, strict=True):
        for color_index, topk in enumerate(topks):
            selected = [row for row in rows if int(row["topk"]) == topk]
            axis.scatter(
                [float(row[left]) for row in selected],
                [float(row[right]) for row in selected],
                s=30,
                alpha=0.8,
                label=f"topk={topk} (n={len(selected)})",
                color=colors(color_index),
            )
        all_left = np.asarray([float(row[left]) for row in rows])
        all_right = np.asarray([float(row[right]) for row in rows])
        rho = correlation(all_left, all_right, "spearman")
        r = correlation(all_left, all_right, "pearson")
        axis.set_xlabel(left)
        axis.set_ylabel(right)
        axis.set_title(f"Pearson={r:+.3f}  Spearman={rho:+.3f}")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(
        f"Endpoint-TracIn removal targets · completed checkpoints n={len(rows)}",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    plot_path = output_root / "four_target_relationship_scatter.png"
    figure.savefig(plot_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    print(f"COMPLETED CHECKPOINTS: {len(rows)}")
    print("GROUP       LEFT                         RIGHT                        PEARSON  SPEARMAN")
    print("-" * 100)
    report_pairs = {
        ("noise_trajectory", "endpoint_difference"),
        ("noise_trajectory", "trajectory_difference"),
        ("noise_trajectory", "simple_loss"),
        ("noise_trajectory", "simple_loss_delta"),
        ("noise_trajectory", "abs_simple_loss_delta"),
    }
    for group_name, group_rows in groups:
        for left, right in report_pairs:
            x = np.asarray([float(row[left]) for row in group_rows])
            y = np.asarray([float(row[right]) for row in group_rows])
            print(
                f"{group_name:<11s} {left:<28s} {right:<28s} "
                f"{correlation(x, y, 'pearson'):+8.3f} "
                f"{correlation(x, y, 'spearman'):+9.3f}"
            )
    print(f"[saved] {csv_path}")
    print(f"[saved] {correlation_path}")
    print(f"[saved] {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("evaluate", "summarize"), required=True)
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--query-ids", default=",".join(str(value) for value in range(60)))
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--metrics-manifest", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be in [0, shard-count)")
    if args.mode == "evaluate":
        evaluate(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
