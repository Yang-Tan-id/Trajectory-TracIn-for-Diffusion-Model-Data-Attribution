#!/usr/bin/env python3
"""Recompute x3 Top-K simple-loss effects with 100 timestamps x MC4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import x3pixel_DM_training as base
from dataset_loader import ColorGridDataset
from exp_config import (
    BASE_CSV,
    EPOCHS,
    MODEL_DIR,
    QUERY_DIR,
    T,
)
from topk_removal_eval_worker import atomic_json, build_model, condition_for
from importlib.util import module_from_spec, spec_from_file_location


TIMESTAMPS = tuple(
    int(value) for value in np.linspace(0, T - 1, 100, dtype=np.int64)
)
MC = 4
SAMPLE_COUNT = len(TIMESTAMPS) * MC


def load_export_config():
    path = Path(__file__).with_name("301_export_topk_removal_results_60q.py")
    spec = spec_from_file_location("topk_export_config", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def simple_loss_100t_mc4(
    model,
    x0,
    condition,
    schedule,
    query_id,
    batch_size,
):
    noisy_inputs = []
    noises = []
    timesteps = []
    for timestep_value in TIMESTAMPS:
        for mc_index in range(MC):
            generator = torch.Generator(device=str(x0.device))
            generator.manual_seed(
                90000000
                + int(query_id) * 10000
                + int(timestep_value) * 10
                + mc_index
            )
            noise = torch.randn(
                x0.shape,
                generator=generator,
                device=x0.device,
                dtype=x0.dtype,
            )
            timestep = torch.tensor(
                [int(timestep_value)], device=x0.device, dtype=torch.long
            )
            noisy_inputs.append(base.q_sample(x0, timestep, noise, schedule))
            noises.append(noise)
            timesteps.append(timestep)
    noisy_inputs = torch.cat(noisy_inputs, dim=0)
    noises = torch.cat(noises, dim=0)
    timesteps = torch.cat(timesteps, dim=0)

    squared_error = 0.0
    element_count = 0
    for start in range(0, SAMPLE_COUNT, batch_size):
        stop = min(start + batch_size, SAMPLE_COUNT)
        inputs = noisy_inputs[start:stop]
        targets = noises[start:stop]
        times = timesteps[start:stop]
        conditions = condition.expand(stop - start, -1)
        predictions = model(inputs, times, conditions)
        squared_error += float(
            F.mse_loss(predictions, targets, reduction="sum").item()
        )
        element_count += targets.numel()
    return squared_error / element_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--query-shard-index", type=int, required=True)
    parser.add_argument("--query-shard-count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=200)
    args = parser.parse_args()
    if not 0 <= args.query_shard_index < args.query_shard_count:
        raise ValueError("invalid query shard")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")

    config = load_export_config()
    query_ids = config.QUERY_IDS[
        args.query_shard_index :: args.query_shard_count
    ]
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    schedule = base.make_linear_schedule(T, device=device)
    queries = json.loads((QUERY_DIR / "manifest.json").read_text())
    reference_models = {}

    print(
        f"[simple-loss gpu={args.gpu}] queries={list(query_ids)} "
        f"timestamps={len(TIMESTAMPS)} mc={MC} samples={SAMPLE_COUNT}",
        flush=True,
    )
    for query_position, query_id in enumerate(query_ids, start=1):
        query = queries[int(query_id)]
        family = query["family"]
        if family not in reference_models:
            reference_path = (
                MODEL_DIR / "base" / family / f"epoch_{EPOCHS:04d}.pt"
            )
            reference_models[family] = build_model(
                reference_path, "ema", device, dataset
            )
        condition = condition_for(query, dataset, device)
        endpoint = torch.from_numpy(
            np.load(Path(query["dir"]) / "final_state.npy")
        ).to(device=device, dtype=torch.float32)
        base_loss = simple_loss_100t_mc4(
            reference_models[family],
            endpoint,
            condition,
            schedule,
            query_id,
            args.batch_size,
        )

        for arm_position, (method, transform, root, tag) in enumerate(
            config.ARMS, start=1
        ):
            job_dir = root / tag / config.FRACTION / f"q{query_id:02d}"
            job_path = job_dir / "job.json"
            evaluation_path = job_dir / "evaluation.json"
            if not job_path.is_file() or not evaluation_path.is_file():
                raise FileNotFoundError(
                    job_path if not job_path.is_file() else evaluation_path
                )
            job = json.loads(job_path.read_text())
            if job["eval_param_source"] != "ema":
                raise ValueError(f"expected EMA evaluation: {job_path}")
            removal_path = (
                Path(job["model_dir"]) / f"epoch_{EPOCHS:04d}.pt"
            )
            removal_model = build_model(removal_path, "ema", device, dataset)
            removal_loss = simple_loss_100t_mc4(
                removal_model,
                endpoint,
                condition,
                schedule,
                query_id,
                args.batch_size,
            )
            increase = removal_loss - base_loss
            payload = json.loads(evaluation_path.read_text())
            payload.update(
                {
                    "simple_loss_base": float(base_loss),
                    "simple_loss_removal": float(removal_loss),
                    "simple_loss_increase": float(increase),
                    "simple_loss_abs_change": float(abs(increase)),
                    "simple_loss_eval_version": 2,
                    "simple_loss_eval_timestamps": list(TIMESTAMPS),
                    "simple_loss_eval_mc": MC,
                    "simple_loss_eval_sample_count": SAMPLE_COUNT,
                    "simple_loss_definition": (
                        "query endpoint simple loss averaged over 100 evenly "
                        "spaced timesteps x MC4 fixed noises; removal and base "
                        "EMA share every sample"
                    ),
                }
            )
            atomic_json(payload, evaluation_path)
            del removal_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(
                f"[simple-loss gpu={args.gpu}] "
                f"q={query_position}/{len(query_ids)} qid={query_id:02d} "
                f"arm={arm_position}/{len(config.ARMS)} {transform} {method} "
                f"abs-change={abs(increase):.8g}",
                flush=True,
            )


if __name__ == "__main__":
    main()
