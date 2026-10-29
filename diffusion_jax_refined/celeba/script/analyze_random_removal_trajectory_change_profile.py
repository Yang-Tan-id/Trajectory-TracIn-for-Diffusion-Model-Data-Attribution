#!/usr/bin/env python3
"""Measure where random-removal trajectory state changes concentrate."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import jax
import numpy as np

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
    paired_mse,
    query_seed,
)

RANDOM_METHOD = "random_full10k_seeded"
FIELDS = (
    "evaluation_query_id",
    "evaluation_seed",
    "random_model_id",
    "step",
    "timestep",
    "trajectory_state_mse",
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


def completed_pairs(rows: Sequence[dict[str, Any]], steps: int = 50) -> set[tuple[int, int]]:
    grouped: dict[tuple[int, int], set[int]] = {}
    for row in rows:
        key = (int(row["evaluation_query_id"]), int(row["random_model_id"]))
        grouped.setdefault(key, set()).add(int(row["step"]))
    expected = set(range(steps))
    return {key for key, seen in grouped.items() if seen == expected}


def write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(
            rows,
            key=lambda row: (
                int(row["evaluation_query_id"]),
                int(row["random_model_id"]),
                int(row["step"]),
            ),
        ))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--query-ids", type=parse_ids, default=list(range(10)))
    parser.add_argument("--model-ids", type=parse_ids, default=list(range(70)))
    parser.add_argument("--experiment", default="attr40_experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
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

    checkpoints = {}
    for model_id in args.model_ids:
        run_dir = find_run_dir((random_root,), RANDOM_METHOD, model_id)
        checkpoints[model_id] = local_checkpoint(run_dir, args.train_seed, args.epochs)

    data_path = output_dir / "trajectory_state_mse_per_timestep.csv"
    rows: list[dict[str, Any]] = [dict(row) for row in read_rows(data_path)]
    complete = completed_pairs(rows)
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
        print(f"[reference] Q{query_id} models={len(args.model_ids)}", flush=True)

        for model_id in args.model_ids:
            pair_position += 1
            key = (query_id, model_id)
            if key in complete:
                print(f"[skip {pair_position}/{pair_total}] Q{query_id} R{model_id}", flush=True)
                continue
            adapter = evaluator._make_adapter(str(checkpoints[model_id]))
            candidate = np.asarray(
                evaluator._sample_model_space_trajectory(
                    adapter, seed=seed, timesteps_to_save=timesteps
                ),
                dtype=np.float32,
            )
            per_step, trajectory_mse, _ = paired_mse(reference, candidate)
            rows = [row for row in rows if (
                int(row["evaluation_query_id"]), int(row["random_model_id"])
            ) != key]
            for step, (timestep, value) in enumerate(zip(timesteps, per_step)):
                rows.append({
                    "evaluation_query_id": query_id,
                    "evaluation_seed": seed,
                    "random_model_id": model_id,
                    "step": step,
                    "timestep": int(timestep),
                    "trajectory_state_mse": float(value),
                })
            write_rows(data_path, rows)
            complete.add(key)
            print(
                f"[saved {pair_position}/{pair_total}] Q{query_id} R{model_id} "
                f"trajectory={trajectory_mse:.8g} peak_step={int(np.argmax(per_step))} "
                f"peak_t={int(timesteps[int(np.argmax(per_step))])}",
                flush=True,
            )
            del adapter, candidate
            gc.collect()

        del evaluator
        gc.collect()

    expected = {(query, model) for query in args.query_ids for model in args.model_ids}
    missing = sorted(expected - completed_pairs(rows))
    if missing:
        raise RuntimeError(f"incomplete pairs: {missing}")
    assert common_timesteps is not None
    (output_dir / "manifest.json").write_text(json.dumps({
        "query_ids": args.query_ids,
        "random_model_ids": args.model_ids,
        "pairs": len(expected),
        "timesteps": common_timesteps.tolist(),
        "metric": "model-space MSE between candidate and full-data x_t",
        "trajectory_reduction": "mean over all 50 saved states",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
