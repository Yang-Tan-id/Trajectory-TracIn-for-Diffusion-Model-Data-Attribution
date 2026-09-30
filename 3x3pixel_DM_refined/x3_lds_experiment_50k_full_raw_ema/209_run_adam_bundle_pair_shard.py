"""Propagate LDS-subset weight tangents through frozen-start AdamW pairs."""

import argparse
import importlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call, grad, vmap

import x3pixel_DM_training as base
from adam_bundle_pair_config import *
from attribution_one_query import _project_batched_grads, build_model, cond_for, model_paths, preload_dataset
from dataset_loader import ColorGridDataset
from forward_loss_alignment_config import replay_noise_path, replay_t_path
from train_worker import lr_at
from x3_endpoint_das_jax_logic_pytorch import build_countsketch_specs


transition = importlib.import_module("204_run_checkpoint_transition_diagnostic_shard")
endpoint = importlib.import_module("206_run_checkpoint_endpoint_adam_diagnostic")


def output_jacobian(model, parameters, names, specs, xt, timestep, condition):
    prediction = model(xt, timestep, condition).reshape(-1)
    basis = torch.eye(prediction.numel(), device=prediction.device, dtype=prediction.dtype)
    rows = torch.autograd.grad(prediction, parameters, grad_outputs=basis, is_grads_batched=True)
    return _project_batched_grads(
        dict(zip(names, rows)), names, specs, BUNDLE_TRACIN_PROJ_DIM, False, 1e-8
    )


def tuple_dot(left, right):
    return sum((a.double() * b.double()).sum() for a, b in zip(left, right))


def run_pair(pair_index, args, dataset, x_all, condition_all, membership, replay_t, replay_noise, records):
    pair_root = ADAM_BUNDLE_ROOT / args.family / f"pair_{pair_index:02d}"
    if (pair_root / "done.json").is_file():
        print(f"[skip] pair={pair_index:02d}", flush=True)
        return
    pair_root.mkdir(parents=True, exist_ok=True)
    device = x_all.device
    paths = model_paths(args.family)
    start_model, _, start_checkpoint = build_model(paths[pair_index], "raw", device)
    target_model, _, _ = build_model(paths[pair_index + 1], "raw", device)
    start_model.train()
    target_model.eval()
    shadow, optimizer = endpoint.make_shadow(paths[pair_index], start_checkpoint, device)
    shadow.train()
    names = tuple(name for name, _ in start_model.named_parameters())
    fixed_params = dict(start_model.named_parameters())
    fixed_tuple = tuple(fixed_params.values())
    shadow_params = tuple(shadow.parameters())
    initial_shadow = tuple(p.detach().clone() for p in shadow_params)
    target_state = dict(target_model.named_parameters())
    exact_delta = tuple(
        target_state[name].detach() - fixed_params[name].detach() for name in names
    )
    specs = build_countsketch_specs(
        list(fixed_tuple), BUNDLE_TRACIN_PROJ_DIM, device=device,
        seed_parts=(TRAIN_SEED, "adam_bundle_pair_projection", args.family, pair_index),
    )
    schedule = base.make_linear_schedule(T, device=device)
    orders, batches_per_epoch = transition.checkpoint_orders(dataset, device, [pair_index + 1])
    total_steps = EPOCHS * batches_per_epoch
    global_step = int(start_checkpoint["global_step"])
    mask_count = membership.shape[0]
    dtheta = [torch.zeros((mask_count,) + tuple(p.shape), device=device, dtype=p.dtype) for p in shadow_params]
    dm = [torch.zeros_like(value) for value in dtheta]
    dv = [torch.zeros_like(value) for value in dtheta]

    def one_loss(params, x0, condition, timestep, noise):
        xt = base.q_sample(x0.unsqueeze(0), timestep.unsqueeze(0), noise.unsqueeze(0), schedule)
        prediction = functional_call(start_model, params, (xt, timestep[None], condition[None]))
        return (prediction - noise[None]).square().mean()

    per_example_grad = vmap(grad(one_loss), in_dims=(None, 0, 0, 0, 0))
    for event_index, epoch_batches in enumerate(orders[pair_index + 1]):
        for batch_position, indices_np in enumerate(epoch_batches):
            indices = torch.from_numpy(indices_np).to(device=device, dtype=torch.long)
            x = x_all.index_select(0, indices)
            condition = condition_all.index_select(0, indices)
            timestep = torch.from_numpy(np.array(replay_t[pair_index + 1, indices_np, event_index], copy=True)).to(device=device, dtype=torch.long)
            noise = torch.from_numpy(np.array(replay_noise[pair_index + 1, indices_np, event_index], copy=True)).to(device=device, dtype=x.dtype)
            grads = per_example_grad(fixed_params, x, condition, timestep, noise)
            batch_size = len(indices_np)
            batch_grad = [grads[name].mean(dim=0) for name in names]
            norm_sq = sum(value.double().square().sum() for value in batch_grad)
            norm = float(norm_sq.sqrt())
            clip_scale = min(1.0, GRAD_CLIP / (norm + 1e-6))
            weights = membership[:, indices].to(dtype=x.dtype) / float(batch_size)
            perturbations = []
            for name, base_gradient in zip(names, batch_grad):
                values = grads[name]
                flat = values.reshape(batch_size, -1)
                perturbations.append((weights @ flat).reshape((mask_count,) + tuple(base_gradient.shape)))
            if clip_scale < 1.0:
                dots = sum(
                    (delta.double() * value.double().unsqueeze(0)).flatten(1).sum(1)
                    for delta, value in zip(perturbations, batch_grad)
                )
                perturbations = [
                    clip_scale * (delta - value.unsqueeze(0) * (dots / norm_sq).to(value.dtype).reshape((-1,) + (1,) * value.ndim))
                    for delta, value in zip(perturbations, batch_grad)
                ]
            else:
                perturbations = [clip_scale * value for value in perturbations]

            learning_rate = lr_at(global_step, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            for parameter, value in zip(shadow_params, batch_grad):
                parameter.grad = (clip_scale * value).detach().clone()
            optimizer.step()
            group = optimizer.param_groups[0]
            beta1, beta2 = group["betas"]
            decay = 1.0 - learning_rate * float(group["weight_decay"])
            for position, (parameter, delta_gradient) in enumerate(zip(shadow_params, perturbations)):
                state = optimizer.state[parameter]
                step = float(state["step"].item())
                bc1 = 1.0 - beta1 ** step
                bc2 = 1.0 - beta2 ** step
                new_dm = beta1 * dm[position] + (1.0 - beta1) * delta_gradient
                clipped = clip_scale * batch_grad[position]
                new_dv = beta2 * dv[position] + 2.0 * (1.0 - beta2) * clipped.unsqueeze(0) * delta_gradient
                root_v = state["exp_avg_sq"].detach().sqrt()
                denominator = root_v / math.sqrt(bc2) + float(group["eps"])
                safe_root = root_v.clamp_min(1e-30)
                delta_denominator = new_dv / (2.0 * safe_root.unsqueeze(0) * math.sqrt(bc2))
                step_size = learning_rate / bc1
                new_dtheta = decay * dtheta[position] - step_size * (
                    new_dm / denominator.unsqueeze(0)
                    - state["exp_avg"].detach().unsqueeze(0) * delta_denominator / denominator.square().unsqueeze(0)
                )
                dm[position], dv[position], dtheta[position] = new_dm, new_dv, new_dtheta
            global_step += 1
        print(f"[adam-bundle gpu={args.gpu}] pair={pair_index:02d} event={event_index + 1}/4", flush=True)

    frozen_delta = tuple(after.detach() - before for after, before in zip(shadow_params, initial_shadow))
    alpha = float(tuple_dot(frozen_delta, exact_delta) / tuple_dot(frozen_delta, frozen_delta).clamp_min(1e-30))
    projected_subset = _project_batched_grads(
        dict(zip(names, dtheta)), names, specs, BUNDLE_TRACIN_PROJ_DIM, False, 1e-8
    ) * (-alpha)
    trajectories = {int(r["query_id"]): np.load(Path(r["dir"]) / "trajectory_xt.npy") for r in records}
    timestamp_arrays = {int(r["query_id"]): np.load(Path(r["dir"]) / "trajectory_t.npy") for r in records}
    conditions = {int(r["query_id"]): cond_for(r, dataset, device) for r in records}
    for record in records:
        qid = int(record["query_id"])
        result = np.empty((mask_count, TRAJ_SNAPSHOTS, BUNDLE_TRACIN_OUTPUT_DIM), dtype=np.float32)
        for timestamp_position, timestep_value in enumerate(timestamp_arrays[qid]):
            xt = torch.from_numpy(trajectories[qid][timestamp_position]).to(device=device, dtype=torch.float32)
            timestep = torch.tensor([int(timestep_value)], device=device)
            jacobian = output_jacobian(start_model, fixed_tuple, names, specs, xt, timestep, conditions[qid])
            result[:, timestamp_position] = (projected_subset @ jacobian.T).detach().cpu().numpy()
        np.save(pair_root / f"q{qid:02d}_vectors.npy", result)
        print(f"[adam-bundle gpu={args.gpu}] pair={pair_index:02d} q{qid:02d} done", flush=True)
    with open(pair_root / "done.json", "w") as handle:
        json.dump({"pair_index": pair_index, "alpha": alpha, "query_ids": [int(r["query_id"]) for r in records], "definition": "negative subset-weight tangent through frozen-start AdamW, scalar-calibrated to observed checkpoint delta"}, handle, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--pair-shard-index", type=int, required=True)
    parser.add_argument("--pair-shard-count", type=int, required=True)
    parser.add_argument("--query-ids", default="0-9")
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    x_all, condition_all = preload_dataset(dataset, args.family, device)
    membership = torch.from_numpy(np.load(MASK_DIR / "membership.npy").astype(np.float32)).to(device)
    replay_t = np.load(replay_t_path(), mmap_mode="r")
    replay_noise = np.load(replay_noise_path(), mmap_mode="r")
    query_ids = parse_query_ids(args.query_ids)
    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    records = [r for r in manifest if int(r["query_id"]) in query_ids and r["family"] == args.family]
    for pair_index in range(args.pair_shard_index, 49, args.pair_shard_count):
        run_pair(pair_index, args, dataset, x_all, condition_all, membership, replay_t, replay_noise, records)


if __name__ == "__main__":
    main()
