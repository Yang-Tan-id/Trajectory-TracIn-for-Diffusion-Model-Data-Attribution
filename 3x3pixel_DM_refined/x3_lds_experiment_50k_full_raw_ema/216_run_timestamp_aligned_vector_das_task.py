"""Compute one timestamp task for vector-valued trajectory DAS Bundle."""

import argparse
import importlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call, grad, vmap

import x3pixel_DM_training as base
from attribution_one_query import _project_batched_grads, build_model, cond_for, model_paths, preload_dataset
from timestamp_aligned_vector_das_config import *
from x3_endpoint_das_jax_logic_pytorch import build_countsketch_specs, make_torch_generator


bundle = importlib.import_module("199_run_bundle_tracin_checkpoint_shard")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--family", choices=FAMILIES, default="prompted")
    parser.add_argument("--query-ids", default="0-9")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if not 0 <= args.task_index < len(TAVD_POSITIONS):
        raise ValueError("task index outside the ten selected timestamps")
    device = torch.device(f"cuda:{args.gpu}")
    query_ids = bundle.parse_query_ids(args.query_ids)
    with open(QUERY_DIR / "manifest.json") as handle:
        records = [r for r in json.load(handle) if int(r["query_id"]) in query_ids and r["family"] == args.family]
    task_root = TAVD_ROOT / args.family / f"task_{args.task_index:02d}"
    if (task_root / "done.json").is_file():
        print(f"[skip] task={args.task_index}", flush=True)
        return
    task_root.mkdir(parents=True, exist_ok=True)
    model, dataset, _ = build_model(model_paths(args.family)[-1], "ema", device)
    model.eval()
    x_all, condition_all = preload_dataset(dataset, args.family, device)
    schedule = base.make_linear_schedule(T, device=device)
    named = dict(model.named_parameters())
    names = tuple(named)
    parameters = tuple(named.values())
    projection_dim = 4096
    specs = build_countsketch_specs(list(parameters), projection_dim, device=device, seed_parts=(TRAIN_SEED, TAVD_METHOD, args.family, args.task_index))
    membership = torch.from_numpy(np.load(MASK_DIR / "membership.npy").astype(np.float32)).to(device)
    gram = torch.zeros((projection_dim, projection_dim), device=device)
    subset_features = torch.zeros((membership.shape[0], projection_dim), device=device)
    first_t = np.load(Path(records[0]["dir"]) / "trajectory_t.npy")
    timestamp_position = TAVD_POSITIONS[args.task_index]
    timestep_value = int(first_t[timestamp_position])
    train_mc = 10
    train_t = torch.full((train_mc,), timestep_value, device=device, dtype=torch.long)

    def point_loss(params, x0, condition, noises):
        x = x0.unsqueeze(0).expand(train_mc, *x0.shape)
        c = condition.unsqueeze(0).expand(train_mc, condition.shape[-1])
        prediction = functional_call(model, params, (base.q_sample(x, train_t, noises, schedule), train_t, c))
        return (prediction - noises).square().mean()

    batched_gradient = vmap(grad(point_loss), in_dims=(None, 0, 0, 0))
    batches = math.ceil(N_TRAIN / args.batch_size)
    for batch_position, start in enumerate(range(0, N_TRAIN, args.batch_size), start=1):
        end = min(start + args.batch_size, N_TRAIN)
        x, condition = x_all[start:end], condition_all[start:end]
        generator = make_torch_generator(device, TRAIN_SEED, TAVD_METHOD, args.family, args.task_index, start)
        noises = torch.randn((end - start, train_mc, *x.shape[1:]), generator=generator, device=device, dtype=x.dtype)
        gradients = batched_gradient(named, x, condition, noises)
        features = _project_batched_grads(gradients, names, specs, projection_dim, bool(DAS_NORMALIZE_PROJECTED_GRADS), 1e-8)
        gram.addmm_(features.T, features)
        subset_features.add_(membership[:, start:end] @ features)
        if batch_position == 1 or batch_position == batches or batch_position % max(1, batches // 10) == 0:
            print(f"[aligned-vector-das gpu={args.gpu}] task={args.task_index + 1}/10 t={timestep_value} batch={batch_position}/{batches}", flush=True)

    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    largest = float(eigenvalues[-1].abs())
    tolerance = max(largest * projection_dim * torch.finfo(eigenvalues.dtype).eps, 1e-12)
    coefficients = eigenvectors.T @ subset_features.T
    directions = {}
    for damping in TAVD_LAMBDAS:
        if damping == 0.0:
            inverse = torch.where(eigenvalues > tolerance, eigenvalues.reciprocal(), torch.zeros_like(eigenvalues))
        else:
            inverse = (eigenvalues + float(damping)).reciprocal()
        directions[damping] = (eigenvectors @ (inverse[:, None] * coefficients)).T

    for record in records:
        qid = int(record["query_id"])
        trajectory = np.load(Path(record["dir"]) / "trajectory_xt.npy")
        trajectory_t = np.load(Path(record["dir"]) / "trajectory_t.npy")
        if int(trajectory_t[timestamp_position]) != timestep_value:
            raise ValueError("query trajectories do not share timestamp grid")
        xt = torch.from_numpy(trajectory[timestamp_position]).to(device=device, dtype=torch.float32)
        timestep = torch.tensor([timestep_value], device=device, dtype=torch.long)
        jacobian = bundle.projected_output_jacobian(model, parameters, names, specs, xt, timestep, cond_for(record, dataset, device))
        for damping, subset_direction in directions.items():
            value = (subset_direction @ jacobian.T).detach().cpu().numpy().astype(np.float32)
            np.save(task_root / f"q{qid:02d}_lambda_{lambda_tag(damping)}.npy", value)
    with open(task_root / "done.json", "w") as handle:
        json.dump({"task_index": args.task_index, "timestamp_position": timestamp_position, "timestep": timestep_value, "query_ids": [int(r["query_id"]) for r in records], "lambdas": list(TAVD_LAMBDAS), "lambda_zero_solver": "eigendecomposition pseudoinverse", "eigen_tolerance": tolerance}, handle, indent=2)
    print(f"[done] task={args.task_index} t={timestep_value}", flush=True)


if __name__ == "__main__":
    main()
