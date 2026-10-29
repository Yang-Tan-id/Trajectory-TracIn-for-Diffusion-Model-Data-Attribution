#!/usr/bin/env python3
"""Relate 50x1 simple-loss changes to CelebA removal endpoint changes."""

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
    find_run_dir,
    load_queries,
    local_checkpoint,
    query_seed,
)

RANDOM_METHOD = "random_full10k_seeded"
SUMMARY_FIELDS = (
    "evaluation_query_id",
    "evaluation_seed",
    "random_model_id",
    "base_simple_loss_50x1",
    "candidate_simple_loss_50x1",
    "simple_loss_signed_delta",
    "simple_loss_absolute_delta",
    "simple_loss_squared_delta",
    "endpoint_mse",
    "trajectory_mse",
)
STEP_FIELDS = (
    "evaluation_query_id",
    "evaluation_seed",
    "random_model_id",
    "step",
    "timestep",
    "base_simple_loss",
    "candidate_simple_loss",
    "simple_loss_signed_delta",
    "simple_loss_absolute_delta",
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


def write_rows(path: Path, fields: Sequence[str], rows: Sequence[dict[str, Any]],
               key) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=key))
    temporary.replace(path)


def load_endpoint_metrics(root: Path) -> dict[tuple[int, int], tuple[float, float]]:
    output: dict[tuple[int, int], tuple[float, float]] = {}
    for path in sorted(root.glob("shard_*/cross_model_per_query_timestep.csv")):
        for row in read_rows(path):
            key = (int(row["evaluation_query_id"]), int(row["random_model_id"]))
            value = (float(row["endpoint_mse"]), float(row["trajectory_mse"]))
            previous = output.setdefault(key, value)
            if not np.allclose(previous, value, rtol=0.0, atol=1e-15):
                raise RuntimeError(f"inconsistent endpoint metrics for {key}")
    return output


def make_simple_loss_per_step_fn(evaluator: CifarTargetEvaluator):
    adapter = evaluator.base_adapter
    schedule = adapter.schedule
    cond = evaluator.cond

    @jax.jit
    def per_step(params, x0_ref, timesteps, rng_keys):
        x0 = x0_ref[0]

        def one(t_scalar, rng):
            t = jnp.full((x0.shape[0],), t_scalar, dtype=jnp.int32)
            noise = jax.random.normal(rng, x0.shape, dtype=x0.dtype)
            xt = (
                adapter.module.extract(
                    schedule.sqrt_alphas_cumprod, t, x0.shape
                ) * x0
                + adapter.module.extract(
                    schedule.sqrt_one_minus_alphas_cumprod, t, x0.shape
                ) * noise
            )
            pred = evaluator._eps_apply(adapter, params, xt, t)
            return jnp.mean(jnp.square(pred - noise))

        return jax.vmap(one)(timesteps, rng_keys)

    return per_step


def correlation(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float, float]:
    pearson = pearsonr(left, right)
    spearman = spearmanr(left, right)
    return (
        float(pearson.statistic),
        float(pearson.pvalue),
        float(spearman.statistic),
        float(spearman.pvalue),
    )


def write_report(output_dir: Path, rows: Sequence[dict[str, Any]],
                 query_ids: Sequence[int]) -> None:
    lines = [
        "CELEBA RANDOM TOP-1000 — SIMPLE LOSS 50x1 VS REMOVAL CHANGE",
        "=" * 116,
        "50x1 = 50 evenly spaced diffusion timesteps (999 to 0), one independent noise draw each.",
        "Each correlation is within one evaluation query across random-removal models.",
        "ABS uses |mean simple loss(candidate) - mean simple loss(full-data)|.",
        "SIGNED preserves the direction of the simple-loss change.",
        "",
        (
            f"{'QUERY':<7}{'N':>5}{'ABS-END-P':>12}{'ABS-END-S':>12}"
            f"{'ABS-TRJ-P':>12}{'ABS-TRJ-S':>12}{'SGN-END-P':>12}"
            f"{'SGN-END-S':>12}{'SGN-TRJ-P':>12}{'SGN-TRJ-S':>12}"
        ),
        "-" * 116,
    ]
    aggregate = []
    for query_id in query_ids:
        selected = [row for row in rows if int(row["evaluation_query_id"]) == query_id]
        selected.sort(key=lambda row: int(row["random_model_id"]))
        absolute = np.asarray([float(row["simple_loss_absolute_delta"]) for row in selected])
        signed = np.asarray([float(row["simple_loss_signed_delta"]) for row in selected])
        endpoint = np.asarray([float(row["endpoint_mse"]) for row in selected])
        trajectory = np.asarray([float(row["trajectory_mse"]) for row in selected])
        values = (
            correlation(absolute, endpoint)[0], correlation(absolute, endpoint)[2],
            correlation(absolute, trajectory)[0], correlation(absolute, trajectory)[2],
            correlation(signed, endpoint)[0], correlation(signed, endpoint)[2],
            correlation(signed, trajectory)[0], correlation(signed, trajectory)[2],
        )
        aggregate.append(values)
        lines.append(
            f"Q{query_id:<6}{len(selected):>5}" + "".join(
                f"{value:>+12.4f}" for value in values
            )
        )
    values = np.asarray(aggregate, dtype=np.float64)
    lines.extend([
        "-" * 116,
        f"{'MEAN':<12}" + "".join(f"{value:>+12.4f}" for value in values.mean(axis=0)),
        f"{'STD':<12}" + "".join(f"{value:>12.4f}" for value in values.std(axis=0)),
    ])
    report = "\n".join(lines) + "\n"
    (output_dir / "simple_loss_50x1_correlation_report.txt").write_text(
        report, encoding="utf-8"
    )
    print(report, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--query-ids", type=parse_ids, default=list(range(10)))
    parser.add_argument("--model-ids", type=parse_ids, default=list(range(70)))
    parser.add_argument("--metric-root", type=Path, required=True)
    parser.add_argument("--experiment", default="attr40_experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--mc-seed", type=int, default=2026)
    parser.add_argument("--prefer-device", default="gpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    queries = load_queries(args.query_file)
    result_root = CELEBA_ROOT / "result" / args.experiment
    random_root = result_root / "endpoint_random_top1000_removal_q0_69"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    base_checkpoint = (
        result_root / "model" / "prompted_jax" /
        f"seed_{args.train_seed}_epoch_{args.epochs:04d}.ckpt"
    )
    if not base_checkpoint.is_file():
        raise FileNotFoundError(base_checkpoint)

    endpoint_metrics = load_endpoint_metrics(args.metric_root)
    expected = {(query, model) for query in args.query_ids for model in args.model_ids}
    missing_metrics = sorted(expected - set(endpoint_metrics))
    if missing_metrics:
        raise RuntimeError(f"missing endpoint/trajectory metrics: {missing_metrics}")

    checkpoints = {}
    for model_id in args.model_ids:
        run_dir = find_run_dir((random_root,), RANDOM_METHOD, model_id)
        checkpoints[model_id] = local_checkpoint(run_dir, args.train_seed, args.epochs)

    summary_path = output_dir / "simple_loss_50x1_cross_model.csv"
    step_path = output_dir / "simple_loss_50x1_per_timestep.csv"
    summary_rows: list[dict[str, Any]] = [dict(row) for row in read_rows(summary_path)]
    step_rows: list[dict[str, Any]] = [dict(row) for row in read_rows(step_path)]
    step_counts: dict[tuple[int, int], set[int]] = {}
    for row in step_rows:
        key = (int(row["evaluation_query_id"]), int(row["random_model_id"]))
        step_counts.setdefault(key, set()).add(int(row["step"]))
    summary_pairs = {
        (int(row["evaluation_query_id"]), int(row["random_model_id"]))
        for row in summary_rows
    }
    complete = {
        key for key in summary_pairs if step_counts.get(key) == set(range(50))
    }

    timesteps_np = np.linspace(999, 0, 50, dtype=np.int32)
    timesteps = None
    pair_total = len(expected)
    pair_position = 0

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
            target_function="simple_loss", sample_root=str(reference_root),
            sample_seed=seed, sample_index=0, max_trajectory_steps=None,
            trajectory_reduction="mean", trajectory_projection=None,
            simple_loss_timesteps=timesteps_np.tolist(),
            simple_loss_noise_seeds=None, simple_loss_num_mc=50,
            simple_loss_mc_seed=args.mc_seed, trajectory_sampler="ddim_eta0",
        )
        loss_fn = make_simple_loss_per_step_fn(evaluator)
        if timesteps is None:
            timesteps = jax.device_put(jnp.asarray(timesteps_np), evaluator.device)
        root_key = jax.random.fold_in(jax.random.PRNGKey(args.mc_seed), query_id)
        rng_keys = jax.device_put(jax.random.split(root_key, 50), evaluator.device)
        x0_ref = evaluator.xt_ref[-1:]
        base_losses = np.asarray(jax.device_get(loss_fn(
            evaluator.base_adapter.state.ema_params, x0_ref, timesteps, rng_keys
        )), dtype=np.float64)
        print(f"[reference] Q{query_id} base={base_losses.mean():.8g}", flush=True)

        for model_id in args.model_ids:
            pair_position += 1
            key = (query_id, model_id)
            if key in complete:
                print(f"[skip {pair_position}/{pair_total}] Q{query_id} R{model_id}", flush=True)
                continue
            adapter = evaluator._make_adapter(str(checkpoints[model_id]))
            candidate_losses = np.asarray(jax.device_get(loss_fn(
                adapter.state.ema_params, x0_ref, timesteps, rng_keys
            )), dtype=np.float64)
            signed_steps = candidate_losses - base_losses
            base_mean = float(base_losses.mean())
            candidate_mean = float(candidate_losses.mean())
            signed = candidate_mean - base_mean
            endpoint, trajectory = endpoint_metrics[key]

            summary_rows = [row for row in summary_rows if (
                int(row["evaluation_query_id"]), int(row["random_model_id"])
            ) != key]
            step_rows = [row for row in step_rows if (
                int(row["evaluation_query_id"]), int(row["random_model_id"])
            ) != key]
            summary_rows.append({
                "evaluation_query_id": query_id, "evaluation_seed": seed,
                "random_model_id": model_id,
                "base_simple_loss_50x1": base_mean,
                "candidate_simple_loss_50x1": candidate_mean,
                "simple_loss_signed_delta": signed,
                "simple_loss_absolute_delta": abs(signed),
                "simple_loss_squared_delta": signed * signed,
                "endpoint_mse": endpoint, "trajectory_mse": trajectory,
            })
            for step, timestep in enumerate(timesteps_np):
                step_rows.append({
                    "evaluation_query_id": query_id, "evaluation_seed": seed,
                    "random_model_id": model_id, "step": step,
                    "timestep": int(timestep),
                    "base_simple_loss": float(base_losses[step]),
                    "candidate_simple_loss": float(candidate_losses[step]),
                    "simple_loss_signed_delta": float(signed_steps[step]),
                    "simple_loss_absolute_delta": float(abs(signed_steps[step])),
                })
            write_rows(
                summary_path, SUMMARY_FIELDS, summary_rows,
                key=lambda row: (int(row["evaluation_query_id"]),
                                 int(row["random_model_id"])),
            )
            write_rows(
                step_path, STEP_FIELDS, step_rows,
                key=lambda row: (int(row["evaluation_query_id"]),
                                 int(row["random_model_id"]), int(row["step"])),
            )
            complete.add(key)
            print(
                f"[saved {pair_position}/{pair_total}] Q{query_id} R{model_id} "
                f"signed={signed:+.8g} abs={abs(signed):.8g}", flush=True
            )
            del adapter
            gc.collect()

        del evaluator
        gc.collect()

    selected_rows = [row for row in summary_rows if (
        int(row["evaluation_query_id"]), int(row["random_model_id"])
    ) in expected]
    if len(selected_rows) != len(expected):
        raise RuntimeError(f"incomplete summary: {len(selected_rows)}/{len(expected)}")
    write_report(output_dir, selected_rows, args.query_ids)
    (output_dir / "manifest.json").write_text(json.dumps({
        "query_ids": args.query_ids, "random_model_ids": args.model_ids,
        "pairs": len(expected), "mc_definition": "50 timesteps x 1 noise each",
        "timesteps": timesteps_np.tolist(), "mc_seed": args.mc_seed,
        "reference_endpoint": "full-data DDIM eta=0 generated x0",
        "metric_root": str(args.metric_root.resolve()),
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
