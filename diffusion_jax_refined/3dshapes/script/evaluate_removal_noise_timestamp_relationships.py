#!/usr/bin/env python3
"""Relate per-timestamp predicted-noise deviation to removal outcomes.

This is a restartable, sharded CPU evaluation over already-trained removal
checkpoints.  It stores the predicted-noise MSE at every saved diffusion
timestamp, then reports Pearson and Spearman correlations against endpoint and
full-trajectory counterfactual differences across queries.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_removal_target_relationships import (
    LEGACY_ROOT,
    METHOD,
    REFINE_ROOT,
    SHAPES_ROOT,
    correlation,
    load_queries,
    parse_ints,
    sample_root,
)


SIDECAR_NAME = "noise_timestamp_relationship_metrics.npz"


def selected_manifest_paths(args: argparse.Namespace, selected_ids: set[int]) -> list[Path]:
    """Select exact manifest entries without opening every JSON in every worker."""
    paths = [
        Path(line)
        for line in args.metrics_manifest.read_text().splitlines()
        if line.strip()
    ]
    selected = []
    for path in paths:
        match = re.search(r"/query_(\d+)_seed_", path.as_posix())
        if match is None:
            raise ValueError(f"cannot recover query ID from manifest path: {path}")
        if int(match.group(1)) in selected_ids:
            selected.append(path)
    return sorted(selected)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def noise_per_timestamp(evaluator, target_adapter, chunk_size: int) -> np.ndarray:
    import jax
    import jax.numpy as jnp

    target_params = target_adapter.state.ema_params
    base_params = evaluator.base_adapter.state.ema_params

    @jax.jit
    def chunk_fn(target_params, base_params, xt_refs, t_seq):
        def one_step(x, t_scalar):
            t = jnp.full((x.shape[0],), t_scalar, dtype=jnp.int32)
            eps_target = evaluator._eps_apply(target_adapter, target_params, x, t)
            eps_base = evaluator._eps_apply(evaluator.base_adapter, base_params, x, t)
            return jnp.mean((eps_target - eps_base) ** 2)

        return jax.vmap(one_step)(xt_refs, t_seq)

    values = []
    count = int(len(evaluator.t_seq))
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        chunk = chunk_fn(
            target_params,
            base_params,
            evaluator.xt_ref[start:stop],
            jax.device_put(
                jnp.asarray(evaluator.t_seq[start:stop], dtype=jnp.int32),
                evaluator.device,
            ),
        )
        values.append(np.asarray(jax.device_get(chunk), dtype=np.float64))
    return np.concatenate(values, axis=0)


def evaluate(args: argparse.Namespace) -> None:
    records = load_queries(args.query_file)
    selected_ids = set(parse_ints(args.query_ids))
    result_root = SHAPES_ROOT / "result" / args.experiment
    removal_root = result_root / "retrac_endpoint_topk_removal_60q"
    inputs = selected_manifest_paths(args, selected_ids)
    assigned = [path for index, path in enumerate(inputs) if index % args.shard_count == args.shard_index]
    print(
        f"[scan] completed={len(inputs)} assigned={len(assigned)} "
        f"shard={args.shard_index}/{args.shard_count}",
        flush=True,
    )
    if not inputs:
        raise RuntimeError("no completed removal metrics selected")

    base_checkpoint = (
        result_root / "model" / "prompted_jax"
        / f"seed_{args.train_seed}_epoch_{args.epochs:04d}.ckpt"
    )
    dataset_root = REFINE_ROOT / "dataset" / "3dshapes" / "20000"
    if not base_checkpoint.is_file():
        raise FileNotFoundError(base_checkpoint)

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
        target_adapter = evaluator._make_adapter(str(checkpoint))
        values = noise_per_timestamp(evaluator, target_adapter, args.noise_chunk_size)
        timesteps = np.asarray(evaluator.t_seq, dtype=np.int32)
        if values.shape != timesteps.shape:
            raise ValueError(f"noise shape {values.shape} != timestep shape {timesteps.shape}")
        atomic_npz(
            sidecar,
            query_id=np.asarray(query, dtype=np.int32),
            topk=np.asarray(int(source["topk"]), dtype=np.int32),
            endpoint_difference=np.asarray(float(source["endpoint_difference"]), dtype=np.float64),
            trajectory_difference=np.asarray(float(source["trajectory_difference"]), dtype=np.float64),
            timesteps=timesteps,
            noise_per_timestamp=values,
        )
        del evaluator, target_adapter
        gc.collect()
        elapsed = time.time() - started
        eta = elapsed / position * (len(assigned) - position)
        print(
            f"[done {position}/{len(assigned)}] topk={int(source['topk'])} Q{query} "
            f"timestamps={len(timesteps)} eta={eta / 60:.1f}m",
            flush=True,
        )


def summarize(args: argparse.Namespace) -> None:
    selected_ids = set(parse_ints(args.query_ids))
    result_root = SHAPES_ROOT / "result" / args.experiment
    removal_root = result_root / "retrac_endpoint_topk_removal_60q"
    inputs = selected_manifest_paths(args, selected_ids)
    rows = []
    missing = []
    reference_timesteps = None
    for metrics_path in inputs:
        sidecar = metrics_path.parent / SIDECAR_NAME
        if not sidecar.is_file():
            missing.append(sidecar)
            continue
        with np.load(sidecar, allow_pickle=False) as payload:
            timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
            if reference_timesteps is None:
                reference_timesteps = timesteps
            elif not np.array_equal(reference_timesteps, timesteps):
                raise ValueError(f"timestamp mismatch in {sidecar}")
            rows.append(
                {
                    "query_id": int(payload["query_id"]),
                    "topk": int(payload["topk"]),
                    "endpoint": float(payload["endpoint_difference"]),
                    "trajectory": float(payload["trajectory_difference"]),
                    "noise": np.asarray(payload["noise_per_timestamp"], dtype=np.float64),
                }
            )
    if missing:
        raise RuntimeError(f"missing {len(missing)} timestamp sidecars; first={missing[0]}")
    if len(rows) < 3 or reference_timesteps is None:
        raise RuntimeError(f"need at least three rows, found {len(rows)}")
    rows.sort(key=lambda row: row["query_id"])

    noise = np.stack([row["noise"] for row in rows], axis=0)
    endpoint = np.asarray([row["endpoint"] for row in rows], dtype=np.float64)
    trajectory = np.asarray([row["trajectory"] for row in rows], dtype=np.float64)
    output_root = removal_root / f"{METHOD}_noise_timestamp_relationships"
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "noise_timestamp_correlations.csv"
    output_rows = []
    for index, timestep in enumerate(reference_timesteps):
        values = noise[:, index]
        output_rows.append(
            {
                "position": index,
                "timestep": int(timestep),
                "n": len(rows),
                "endpoint_pearson": correlation(values, endpoint, "pearson"),
                "endpoint_spearman": correlation(values, endpoint, "spearman"),
                "trajectory_pearson": correlation(values, trajectory, "pearson"),
                "trajectory_spearman": correlation(values, trajectory, "spearman"),
                "noise_mean": float(np.mean(values)),
                "noise_std": float(np.std(values)),
            }
        )
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(output_rows, key=lambda row: row["timestep"])
    x = np.asarray([row["timestep"] for row in ordered])
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(x, [row["endpoint_pearson"] for row in ordered], label="Pearson", alpha=0.8)
    axes[0].plot(x, [row["endpoint_spearman"] for row in ordered], label="Spearman", alpha=0.8)
    axes[0].set_title("Per-timestamp noise deviation vs endpoint difference")
    axes[1].plot(x, [row["trajectory_pearson"] for row in ordered], label="Pearson", alpha=0.8)
    axes[1].plot(x, [row["trajectory_spearman"] for row in ordered], label="Spearman", alpha=0.8)
    axes[1].set_title("Per-timestamp noise deviation vs trajectory difference")
    for axis in axes:
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axis.set_ylabel("correlation")
        axis.grid(alpha=0.2)
        axis.legend()
    axes[1].set_xlabel("diffusion timestep t")
    figure.suptitle(f"Removal checkpoints n={len(rows)} · topk={sorted({row['topk'] for row in rows})}")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    plot_path = output_root / "noise_timestamp_correlations.png"
    figure.savefig(plot_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    print(f"COMPLETED CHECKPOINTS: {len(rows)}")
    for target in ("endpoint", "trajectory"):
        ranked = sorted(output_rows, key=lambda row: row[f"{target}_spearman"], reverse=True)
        print(f"TOP 10 {target.upper()} SPEARMAN")
        for row in ranked[:10]:
            print(
                f"t={row['timestep']:4d} pearson={row[f'{target}_pearson']:+.4f} "
                f"spearman={row[f'{target}_spearman']:+.4f}"
            )
    print(f"[saved] {csv_path}")
    print(f"[saved] {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("evaluate", "summarize"), required=True)
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--query-ids", default=",".join(str(value) for value in range(57)))
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--metrics-manifest", type=Path, required=True)
    parser.add_argument("--noise-chunk-size", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be in [0, shard-count)")
    if args.noise_chunk_size <= 0:
        parser.error("--noise-chunk-size must be positive")
    if args.mode == "evaluate":
        evaluate(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
