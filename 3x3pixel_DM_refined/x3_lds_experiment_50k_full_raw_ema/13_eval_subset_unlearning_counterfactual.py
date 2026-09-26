"""Evaluate all 192 subset-level parameter-unlearning counterfactuals."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call, vmap

import x3pixel_DM_training as base
from attribution_one_query import build_model, cond_for
from checkpoint_counterfactual_config import *
from dataset_loader import ColorGridDataset
from exp_config import *


def atomic_npz(path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **values)
    os.replace(temporary, path)


def parameter_bank(model, payload, scale, device):
    names = tuple(name for name, _ in model.named_parameters())
    if tuple(payload["parameter_names"]) != names:
        raise ValueError("update bank parameter names do not match final model")
    updates = payload["updates"].to(device=device, dtype=torch.float32)
    bank = {}
    offset = 0
    for name, parameter in model.named_parameters():
        width = parameter.numel()
        delta = updates[:, offset : offset + width].reshape(
            updates.shape[0], *parameter.shape
        )
        # The bank contains optimizer directions (-eta * gradient). Removing
        # that direction therefore undoes the selected examples' update.
        bank[name] = parameter.detach().unsqueeze(0) - float(scale) * delta
        offset += width
    if offset != updates.shape[1]:
        raise ValueError(f"unused update columns: consumed={offset}, total={updates.shape[1]}")
    return bank


@torch.no_grad()
def batched_counterfactual_trajectory(model, params, schedule, condition, x_t):
    count = next(iter(params.values())).shape[0]
    x = x_t.expand(count, *x_t.shape[1:]).clone()
    conditions = condition.expand(count, condition.shape[-1])
    ts = torch.linspace(T - 1, 0, DDIM_STEPS, device=x.device).long()
    save_steps = np.linspace(
        0, DDIM_STEPS - 1, TRAJ_SNAPSHOTS, dtype=np.int64
    ).tolist()
    save_set = set(save_steps)
    saved = {0: x.detach().cpu()} if 0 in save_set else {}

    def predict(one_params, one_x, one_condition, one_t):
        return functional_call(
            model,
            one_params,
            (one_x.unsqueeze(0), one_t.unsqueeze(0), one_condition.unsqueeze(0)),
        ).squeeze(0)

    batched_predict = vmap(predict, in_dims=(0, 0, 0, 0))
    for index in range(len(ts) - 1):
        t = ts[index].expand(count)
        t_previous = int(ts[index + 1].item())
        epsilon = batched_predict(params, x, conditions, t)
        alpha = schedule.alpha_bars[t].view(-1, 1, 1, 1)
        alpha_previous = schedule.alpha_bars[t_previous].view(1, 1, 1, 1)
        x0 = (x - torch.sqrt(1.0 - alpha) * epsilon) / torch.sqrt(alpha)
        x = torch.sqrt(alpha_previous) * x0 + torch.sqrt(1.0 - alpha_previous) * epsilon
        step = index + 1
        if step in save_set:
            saved[step] = x.detach().cpu()
    return torch.stack([saved[step] for step in save_steps], dim=0).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--query-shard-index", type=int, required=True)
    parser.add_argument("--query-shard-count", type=int, required=True)
    parser.add_argument("--update-scale", type=float, default=CF_UPDATE_SCALE)
    parser.add_argument(
        "--final-source", choices=("raw", "ema"), default=CF_FINAL_PARAM_SOURCE
    )
    args = parser.parse_args()
    if not 0 <= args.query_shard_index < args.query_shard_count:
        raise ValueError("invalid query shard")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    schedule = base.make_linear_schedule(T, device=device)
    with open(QUERY_DIR / "manifest.json") as handle:
        queries = json.load(handle)
    query_ids = list(range(args.query_shard_index, len(queries), args.query_shard_count))
    model_cache = {}
    params_cache = {}
    started = time.perf_counter()

    for position, query_id in enumerate(query_ids, start=1):
        query = queries[query_id]
        family = query["family"]
        output = (
            CF_RESPONSE_ROOT / f"source_{args.final_source}_scale_{args.update_scale:g}"
            / f"q{query_id:02d}.npz"
        )
        if output.is_file():
            print(f"[skip] {output}", flush=True)
            continue
        if family not in model_cache:
            final_path = MODEL_DIR / "base" / family / f"epoch_{EPOCHS:04d}.pt"
            model, _, _ = build_model(final_path, args.final_source, device)
            payload = torch.load(
                CF_UPDATE_ROOT / f"{family}.pt", map_location="cpu", weights_only=False
            )
            model_cache[family] = model
            params_cache[family] = parameter_bank(
                model, payload, args.update_scale, device
            )
        model = model_cache[family]
        params = params_cache[family]
        condition = cond_for(query, dataset, device)
        query_dir = Path(query["dir"])
        reference = np.load(query_dir / "trajectory_xt.npy").astype(np.float32)
        initial = torch.from_numpy(reference[0]).to(device=device, dtype=torch.float32)
        counterfactual = batched_counterfactual_trajectory(
            model, params, schedule, condition, initial
        )
        reference_without_model_axis = reference[:, 0]
        delta = counterfactual - reference_without_model_axis[:, None]
        reduce_axes = tuple(range(2, delta.ndim))
        per_snapshot_mse = np.mean(delta.astype(np.float64) ** 2, axis=reduce_axes)
        trajectory_mse = per_snapshot_mse.mean(axis=0)
        endpoint_mse = per_snapshot_mse[-1]
        atomic_npz(
            output,
            query_id=np.asarray(query_id, dtype=np.int64),
            endpoint_deviation=endpoint_mse,
            trajectory_state_mse=trajectory_mse,
            per_snapshot_mse=per_snapshot_mse,
            final_param_source=np.asarray(args.final_source),
            update_scale=np.asarray(args.update_scale, dtype=np.float64),
        )
        elapsed = time.perf_counter() - started
        eta = elapsed / position * (len(query_ids) - position)
        print(
            f"[counterfactual gpu={args.gpu}] q{query_id:02d} "
            f"{position}/{len(query_ids)} endpoint_mean={endpoint_mse.mean():.7g} "
            f"trajectory_mean={trajectory_mse.mean():.7g} "
            f"elapsed={elapsed/3600:.2f}h eta≈{eta/3600:.2f}h",
            flush=True,
        )


if __name__ == "__main__":
    main()
