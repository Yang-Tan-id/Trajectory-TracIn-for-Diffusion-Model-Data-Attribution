#!/usr/bin/env python3
"""Cross-evaluate CelebA queries on all random top-1000 removal models."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import pearsonr, spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
CELEBA_ROOT = SCRIPT_DIR.parent
REFINED_ROOT = CELEBA_ROOT.parent
LEGACY_ROOT = REFINED_ROOT / "legacy_jax"
for path in (CELEBA_ROOT, REFINED_ROOT, LEGACY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataset_config_attr40 import DATA_ROOT, _prompt_path_tag
from LDS.DM_cifar_lds import CifarTargetEvaluator
from evaluate_top1000_q0_69_generate import (
    display_image,
    endpoint_pixel_metrics,
    find_run_dir,
    load_queries,
    local_checkpoint,
    paired_mse,
    query_seed,
)

RANDOM_METHOD = "random_full10k_seeded"
FIELDS = (
    "evaluation_query_id", "evaluation_seed", "random_model_id", "step",
    "timestep", "predicted_noise_mse", "endpoint_mse", "trajectory_mse",
)


def parse_ids(text: str) -> list[int]:
    values = [int(value) for value in text.replace(",", " ").split() if value]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def completed_pairs(rows: Sequence[dict[str, Any]], steps: int) -> set[tuple[int, int]]:
    grouped: dict[tuple[int, int], set[int]] = {}
    for row in rows:
        key = (int(row["evaluation_query_id"]), int(row["random_model_id"]))
        grouped.setdefault(key, set()).add(int(row["step"]))
    expected = set(range(steps))
    return {key for key, seen in grouped.items() if seen == expected}


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    key = lambda row: (int(row["evaluation_query_id"]),
                       int(row["random_model_id"]), int(row["step"]))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=key))
    temporary.replace(path)


def make_noise_mse_fn(evaluator: CifarTargetEvaluator):
    model = evaluator.base_adapter.model
    cond = evaluator.cond
    predict_x0 = bool(evaluator.base_adapter.cfg.predict_x0)
    alphas_cumprod = evaluator.base_adapter.schedule.alphas_cumprod

    def to_eps(pred, x, t):
        if not predict_x0:
            return pred
        abar = alphas_cumprod[t].reshape((-1, 1, 1, 1))
        return (x - jnp.sqrt(abar) * pred) / jnp.sqrt(1.0 - abar)

    @jax.jit
    def per_step(candidate_params, base_params, reference_xt, timesteps):
        def one(x, scalar_t):
            t = jnp.full((x.shape[0],), scalar_t, dtype=jnp.int32)
            candidate = model.apply(
                {"params": candidate_params}, x, t, cond, train=False
            )
            base = model.apply({"params": base_params}, x, t, cond, train=False)
            delta = to_eps(candidate, x, t) - to_eps(base, x, t)
            return jnp.mean(jnp.square(delta))
        return jax.vmap(one)(reference_xt, timesteps)

    return per_step


def correlation(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    if len(left) < 2:
        return {name: float("nan") for name in
                ("pearson", "pearson_p", "spearman", "spearman_p")}
    pearson = pearsonr(left, right)
    spearman = spearmanr(left, right)
    return {
        "pearson": float(pearson.statistic), "pearson_p": float(pearson.pvalue),
        "spearman": float(spearman.statistic), "spearman_p": float(spearman.pvalue),
    }


def write_correlations(output_dir: Path, rows: Sequence[dict[str, Any]],
                       query_ids: Sequence[int], model_ids: Sequence[int],
                       timesteps: Sequence[int]) -> None:
    lookup = {(int(row["evaluation_query_id"]), int(row["random_model_id"]),
               int(row["step"])): row for row in rows}
    output = []
    for query_id in query_ids:
        endpoint = np.asarray([float(lookup[(query_id, model_id, 0)]["endpoint_mse"])
                               for model_id in model_ids])
        trajectory = np.asarray([float(lookup[(query_id, model_id, 0)]["trajectory_mse"])
                                 for model_id in model_ids])
        for step, timestep in enumerate(timesteps):
            noise = np.asarray([float(lookup[(query_id, model_id, step)]["predicted_noise_mse"])
                                for model_id in model_ids])
            end, traj = correlation(noise, endpoint), correlation(noise, trajectory)
            output.append({
                "evaluation_query_id": query_id, "step": step,
                "timestep": int(timestep), "models": len(model_ids),
                "endpoint_pearson": end["pearson"], "endpoint_pearson_p": end["pearson_p"],
                "endpoint_spearman": end["spearman"], "endpoint_spearman_p": end["spearman_p"],
                "trajectory_pearson": traj["pearson"], "trajectory_pearson_p": traj["pearson_p"],
                "trajectory_spearman": traj["spearman"], "trajectory_spearman_p": traj["spearman_p"],
            })
    path = output_dir / "correlation_per_query_by_timestep.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)

    lines = [
        "CELEBA RANDOM TOP-1000 — CROSS-MODEL PREDICTED-NOISE CORRELATION",
        "=" * 105,
        f"queries={len(query_ids)} models_per_query={len(model_ids)} "
        f"pairs={len(query_ids) * len(model_ids)}",
        "Each correlation is within one evaluation query across random-removal models.",
        "Predicted-noise MSE uses the same full-data reference x_t for both models.", "",
    ]
    for query_id in query_ids:
        selected = [row for row in output if int(row["evaluation_query_id"]) == query_id]
        best = max(selected, key=lambda row: abs(float(row["trajectory_pearson"])))
        lines.append(
            f"Q{query_id}: strongest |trajectory Pearson| at t={best['timestep']} "
            f"r={best['trajectory_pearson']:+.6f} p={best['trajectory_pearson_p']:.4g}"
        )
    (output_dir / "correlation_report.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--query-ids", type=parse_ids, default=list(range(10)))
    parser.add_argument("--model-ids", type=parse_ids, default=list(range(70)))
    parser.add_argument("--experiment", default="attr40_experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--prefer-device", default="gpu")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    queries = load_queries(args.query_file)
    result_root = CELEBA_ROOT / "result" / args.experiment
    random_root = result_root / "endpoint_random_top1000_removal_q0_69"
    output_dir = args.output_dir or (
        result_root / "analysis" / "random_cross_model_noise_correlation_q0_9"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    base_checkpoint = (result_root / "model" / "prompted_jax" /
                       f"seed_{args.train_seed}_epoch_{args.epochs:04d}.ckpt")
    if not base_checkpoint.is_file():
        raise FileNotFoundError(base_checkpoint)

    checkpoints = {}
    for model_id in args.model_ids:
        record = queries[model_id]
        run_dir = find_run_dir((random_root,), RANDOM_METHOD, model_id)
        checkpoints[model_id] = local_checkpoint(run_dir, args.train_seed, args.epochs)

    data_path = output_dir / "cross_model_per_query_timestep.csv"
    rows: list[dict[str, Any]] = [dict(row) for row in read_rows(data_path)]
    complete = completed_pairs(rows, steps=50)
    pair_total = len(args.query_ids) * len(args.model_ids)
    pair_position = 0
    common_timesteps = None

    for query_id in args.query_ids:
        record = queries[query_id]
        seed, prompt = query_seed(record), str(record["prompt"])
        reference_root = (
            result_root / "sample_ddim_eta0_50" / "cifar" /
            f"prompt_{_prompt_path_tag(prompt)}" /
            f"model_prompted_solo__ckpt_{base_checkpoint.stem}"
        )
        evaluator = CifarTargetEvaluator(
            code_file=str(LEGACY_ROOT / "DM__training_CELEBA_pixel.py"),
            base_checkpoint=str(base_checkpoint), prompt=prompt,
            prefer_device=args.prefer_device, data_root=DATA_ROOT,
            target_function="traj_counterfactual", sample_root=str(reference_root),
            sample_seed=seed, sample_index=0, max_trajectory_steps=None,
            trajectory_reduction="mean", trajectory_projection=None,
            simple_loss_timesteps=[0], simple_loss_noise_seeds=None,
            simple_loss_num_mc=1, simple_loss_mc_seed=0,
            trajectory_sampler="ddim_eta0",
        )
        reference = np.asarray(jax.device_get(evaluator.xt_ref), dtype=np.float32)
        timesteps = np.asarray(evaluator.t_seq, dtype=np.int32)
        if len(timesteps) != 50:
            raise RuntimeError(f"Q{query_id} expected 50 steps, found {len(timesteps)}")
        if common_timesteps is None:
            common_timesteps = timesteps
        elif not np.array_equal(common_timesteps, timesteps):
            raise RuntimeError(f"Q{query_id} has a different DDIM schedule")
        noise_mse_fn = make_noise_mse_fn(evaluator)
        print(f"[reference] Q{query_id} models={len(args.model_ids)}", flush=True)

        for model_id in args.model_ids:
            pair_position += 1
            if (query_id, model_id) in complete:
                print(f"[skip {pair_position}/{pair_total}] Q{query_id} R{model_id}", flush=True)
                continue
            print(f"[pair {pair_position}/{pair_total}] Q{query_id} R{model_id}", flush=True)
            adapter = evaluator._make_adapter(str(checkpoints[model_id]))
            candidate = np.asarray(
                evaluator._sample_model_space_trajectory(
                    adapter, seed=seed, timesteps_to_save=timesteps
                ), dtype=np.float32,
            )
            _, trajectory_mse, _ = paired_mse(reference, candidate)
            endpoint_mse, _ = endpoint_pixel_metrics(reference[-1], candidate[-1])
            noise_mse = np.asarray(jax.device_get(noise_mse_fn(
                adapter.state.ema_params, evaluator.base_adapter.state.ema_params,
                evaluator.xt_ref, jnp.asarray(timesteps, dtype=jnp.int32),
            )), dtype=np.float64)
            for step, (timestep, value) in enumerate(zip(timesteps, noise_mse)):
                rows.append({
                    "evaluation_query_id": query_id, "evaluation_seed": seed,
                    "random_model_id": model_id, "step": step,
                    "timestep": int(timestep), "predicted_noise_mse": float(value),
                    "endpoint_mse": endpoint_mse, "trajectory_mse": trajectory_mse,
                })
            write_rows(data_path, rows)
            complete.add((query_id, model_id))
            print(
                f"[saved] Q{query_id} R{model_id} endpoint={endpoint_mse:.8g} "
                f"trajectory={trajectory_mse:.8g} noise_mean={noise_mse.mean():.8g}",
                flush=True,
            )
            del adapter
            gc.collect()

        del evaluator
        gc.collect()

    expected = {(query, model) for query in args.query_ids for model in args.model_ids}
    missing = sorted(expected - completed_pairs(rows, steps=50))
    if missing:
        raise RuntimeError(f"incomplete pairs: {missing}")
    assert common_timesteps is not None
    write_correlations(output_dir, rows, args.query_ids, args.model_ids, common_timesteps)
    (output_dir / "manifest.json").write_text(json.dumps({
        "query_file": str(args.query_file.resolve()),
        "query_ids": args.query_ids, "random_model_ids": args.model_ids,
        "pairs": len(args.query_ids) * len(args.model_ids),
        "timesteps": [int(value) for value in common_timesteps],
        "endpoint_definition": "pixel MSE after mapping model state to [0,1]",
        "trajectory_definition": "mean model-space x_t MSE over all 50 DDIM states",
        "noise_definition": "per-timestep epsilon MSE on identical full-data reference x_t",
    }, indent=2) + "\n")
    print(f"[done] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
