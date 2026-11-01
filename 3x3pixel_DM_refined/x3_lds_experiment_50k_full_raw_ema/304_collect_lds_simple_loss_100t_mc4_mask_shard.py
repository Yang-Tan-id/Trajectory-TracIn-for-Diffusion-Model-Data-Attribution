#!/usr/bin/env python3
"""Collect x3 LDS simple loss with 100 timestamps x MC4 by mask shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import x3pixel_DM_training as base
from dataset_loader import ColorGridDataset
from exp_config import *


TIMESTAMPS = tuple(
    int(value) for value in np.linspace(0, T - 1, 100, dtype=np.int64)
)
MC = 4


def model_path(mask_record, family):
    return (
        MODEL_DIR
        / "subsets"
        / f"seed_{int(mask_record['lds_seed']):02d}"
        / f"subset_{int(mask_record['subset_id']):02d}"
        / family
        / f"epoch_{EPOCHS:04d}.pt"
    )


def build(path, source, device, dataset):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = base.CondEpsModel(3, len(dataset.vocab), BASE_CH, TIME_DIM).to(device)
    key = "model_state" if source == "raw" else "ema_model_state"
    model.load_state_dict(checkpoint[key], strict=True)
    return model.eval()


def condition_for(query, dataset, device):
    condition = torch.zeros((1, len(dataset.vocab)), device=device)
    if query["family"] == "prompted":
        for label in query.get("labels", []):
            condition[0, dataset.vocab[label]] = 1.0
    return condition


@torch.no_grad()
def build_query_bank(query, dataset, schedule, device):
    query_id = int(query["query_id"])
    endpoint = torch.from_numpy(
        np.load(Path(query["dir"]) / "final_state.npy")
    ).to(device=device, dtype=torch.float32)
    condition = condition_for(query, dataset, device)
    inputs = []
    noises = []
    times = []
    for timestep_value in TIMESTAMPS:
        timestep = torch.tensor(
            [timestep_value], device=device, dtype=torch.long
        )
        for mc_index in range(MC):
            generator = torch.Generator(device=str(device))
            generator.manual_seed(
                90000000
                + query_id * 10000
                + timestep_value * 10
                + mc_index
            )
            noise = torch.randn(
                endpoint.shape,
                generator=generator,
                device=device,
                dtype=endpoint.dtype,
            )
            inputs.append(base.q_sample(endpoint, timestep, noise, schedule))
            noises.append(noise)
            times.append(timestep)
    return {
        "inputs": torch.cat(inputs, dim=0),
        "noises": torch.cat(noises, dim=0),
        "times": torch.cat(times, dim=0),
        "condition": condition,
    }


@torch.no_grad()
def evaluate_bank(model, bank, batch_size):
    sample_count = bank["inputs"].shape[0]
    squared_error = 0.0
    element_count = 0
    for start in range(0, sample_count, batch_size):
        stop = min(start + batch_size, sample_count)
        targets = bank["noises"][start:stop]
        predictions = model(
            bank["inputs"][start:stop],
            bank["times"][start:stop],
            bank["condition"].expand(stop - start, -1),
        )
        squared_error += float(
            F.mse_loss(predictions, targets, reduction="sum").item()
        )
        element_count += targets.numel()
    return squared_error / element_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--mask-shard-index", type=int, required=True)
    parser.add_argument("--mask-shard-count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=200)
    args = parser.parse_args()
    if not 0 <= args.mask_shard_index < args.mask_shard_count:
        raise ValueError("invalid mask shard")

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    schedule = base.make_linear_schedule(T, device=device)
    masks = json.loads((MASK_DIR / "manifest.json").read_text())
    queries = json.loads((QUERY_DIR / "manifest.json").read_text())
    query_ids = [int(query["query_id"]) for query in queries]
    if query_ids != list(range(len(queries))):
        raise ValueError("query manifest must be ordered by query_id")

    print(
        f"[lds-simple gpu={args.gpu}] building {len(queries)} shared banks; "
        f"timestamps={len(TIMESTAMPS)} mc={MC}",
        flush=True,
    )
    banks = {
        int(query["query_id"]): build_query_bank(
            query, dataset, schedule, device
        )
        for query in queries
    }
    family_queries = {
        family: [
            int(query["query_id"])
            for query in queries
            if query["family"] == family
        ]
        for family in FAMILIES
    }
    shard_dir = LDS_DIR / "observed_simple_loss_100t_mc4_mask_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    mask_indices = list(
        range(args.mask_shard_index, len(masks), args.mask_shard_count)
    )
    for position, mask_index in enumerate(mask_indices, start=1):
        destination = shard_dir / f"mask_{mask_index:03d}.npz"
        if destination.is_file():
            print(f"[skip] {destination}", flush=True)
            continue
        values = {
            "simple_loss_ema": np.empty(len(queries), dtype=np.float64),
            "simple_loss_raw": np.empty(len(queries), dtype=np.float64),
        }
        mask_record = masks[mask_index]
        for family in FAMILIES:
            path = model_path(mask_record, family)
            if not path.is_file():
                raise FileNotFoundError(path)
            for source in ("ema", "raw"):
                model = build(path, source, device, dataset)
                for query_id in family_queries[family]:
                    values[f"simple_loss_{source}"][query_id] = evaluate_bank(
                        model, banks[query_id], args.batch_size
                    )
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        temporary = destination.with_suffix(".tmp.npz")
        np.savez(
            temporary,
            mask_index=np.asarray(mask_index, dtype=np.int64),
            timestamps=np.asarray(TIMESTAMPS, dtype=np.int64),
            mc=np.asarray(MC, dtype=np.int64),
            **values,
        )
        temporary.replace(destination)
        print(
            f"[lds-simple gpu={args.gpu}] mask={position}/{len(mask_indices)} "
            f"global={mask_index:03d} saved={destination}",
            flush=True,
        )


if __name__ == "__main__":
    main()
