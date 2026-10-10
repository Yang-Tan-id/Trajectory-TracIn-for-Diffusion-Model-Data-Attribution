"""Timestamp-aligned vector DAS terms for ten trajectory positions."""

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
from exp_config import *
from x3_endpoint_das_jax_logic_pytorch import build_countsketch_specs, make_torch_generator


bundle = importlib.import_module("199_run_bundle_tracin_checkpoint_shard")
METHOD = "timestamp_aligned_vector_trajectory_das_ema_mc10_projected4096_bundle_10t"
POSITIONS = (0, 11, 22, 33, 44, 55, 66, 77, 88, 99)
LAMBDAS = (10.0, 100.0, 1000.0)


def tag(value):
    return str(float(value)).replace(".", "p")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--timestamp-shard-index", type=int, required=True)
    parser.add_argument("--timestamp-shard-count", type=int, required=True)
    parser.add_argument("--family", choices=FAMILIES, default="prompted")
    parser.add_argument("--query-ids", default="0-9")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    query_ids = bundle.parse_query_ids(args.query_ids)
    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    records = [r for r in manifest if int(r["query_id"]) in query_ids and r["family"] == args.family]
    model, dataset, _ = build_model(model_paths(args.family)[-1], "ema", device)
    model.eval()
    x_all, condition_all = preload_dataset(dataset, args.family, device)
    schedule = base.make_linear_schedule(T, device=device)
    named = dict(model.named_parameters())
    names = tuple(named)
    parameters = tuple(named.values())
    membership = torch.from_numpy(np.load(MASK_DIR / "membership.npy").astype(np.float32)).to(device)
    trajectories = {int(r["query_id"]): np.load(Path(r["dir"]) / "trajectory_xt.npy") for r in records}
    trajectory_times = {int(r["query_id"]): np.load(Path(r["dir"]) / "trajectory_t.npy") for r in records}
    conditions = {int(r["query_id"]): cond_for(r, dataset, device) for r in records}
    projection_dim, train_mc = 4096, 10
    root = ROOT / f"_{METHOD}_terms" / args.family

    for position in POSITIONS[args.timestamp_shard_index::args.timestamp_shard_count]:
        term_root = root / f"position_{position:03d}"
        if (term_root / "done.json").is_file():
            print(f"[skip] position={position}", flush=True)
            continue
        reference_times = {int(trajectory_times[int(r['query_id'])][position]) for r in records}
        if len(reference_times) != 1:
            raise ValueError("selected queries do not share trajectory timestamps")
        timestep_value = reference_times.pop()
        specs = build_countsketch_specs(
            list(parameters), projection_dim, device=device,
            seed_parts=(TRAIN_SEED, METHOD, args.family, position, timestep_value),
        )
        gram = torch.zeros((projection_dim, projection_dim), device=device)
        subset_features = torch.zeros((membership.shape[0], projection_dim), device=device)
        fixed_times = torch.full((train_mc,), timestep_value, device=device, dtype=torch.long)

        def point_loss(params, x0, condition, noises):
            x = x0.unsqueeze(0).expand(train_mc, *x0.shape)
            c = condition.unsqueeze(0).expand(train_mc, condition.shape[-1])
            prediction = functional_call(model, params, (base.q_sample(x, fixed_times, noises, schedule), fixed_times, c))
            return (prediction - noises).square().mean()

        batched_gradient = vmap(grad(point_loss), in_dims=(None, 0, 0, 0))
        batches = math.ceil(N_TRAIN / args.batch_size)
        for batch_position, start in enumerate(range(0, N_TRAIN, args.batch_size), start=1):
            end = min(start + args.batch_size, N_TRAIN)
            x, condition = x_all[start:end], condition_all[start:end]
            generator = make_torch_generator(device, TRAIN_SEED, METHOD, args.family, position, start)
            noises = torch.randn((end - start, train_mc, *x.shape[1:]), generator=generator, device=device, dtype=x.dtype)
            gradients = batched_gradient(named, x, condition, noises)
            features = _project_batched_grads(gradients, names, specs, projection_dim, bool(DAS_NORMALIZE_PROJECTED_GRADS), 1e-8)
            gram.addmm_(features.T, features)
            subset_features.add_(membership[:, start:end] @ features)
            if batch_position == 1 or batch_position == batches or batch_position % max(1, batches // 10) == 0:
                print(f"[aligned-vector-das gpu={args.gpu}] pos={position} t={timestep_value} batch={batch_position}/{batches}", flush=True)
        eye = torch.eye(projection_dim, device=device)
        directions = {
            lam: torch.linalg.solve(gram + lam * eye, subset_features.T).T
            for lam in LAMBDAS
        }
        term_root.mkdir(parents=True, exist_ok=True)
        for record in records:
            qid = int(record["query_id"])
            xt = torch.from_numpy(trajectories[qid][position]).to(device=device, dtype=torch.float32)
            timestep = torch.tensor([timestep_value], device=device, dtype=torch.long)
            jacobian = bundle.projected_output_jacobian(model, parameters, names, specs, xt, timestep, conditions[qid])
            for lam, direction in directions.items():
                np.save(term_root / f"q{qid:02d}_lambda_{tag(lam)}.npy", (direction @ jacobian.T).detach().cpu().numpy().astype(np.float32))
        with open(term_root / "done.json", "w") as handle:
            json.dump({"position": position, "timestep": timestep_value, "lambdas": LAMBDAS, "query_ids": [int(r["query_id"]) for r in records]}, handle, indent=2)
        print(f"[done] position={position} t={timestep_value}", flush=True)


if __name__ == "__main__":
    main()
