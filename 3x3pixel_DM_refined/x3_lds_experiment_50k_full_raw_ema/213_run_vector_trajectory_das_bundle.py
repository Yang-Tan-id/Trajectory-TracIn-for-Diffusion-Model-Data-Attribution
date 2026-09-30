"""Vector-valued trajectory DAS Bundle score for q00-q09."""

import argparse
import importlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.func import functional_call, grad, vmap

import x3pixel_DM_training as base
from attribution_one_query import _project_batched_grads, build_model, cond_for, model_paths, preload_dataset
from exp_config import *
from x3_endpoint_das_jax_logic_pytorch import build_countsketch_specs, make_torch_generator


bundle = importlib.import_module("199_run_bundle_tracin_checkpoint_shard")
METHOD = "vector_trajectory_das_ema_mc10_projected4096_lambda100_bundle"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--family", choices=FAMILIES, default="prompted")
    parser.add_argument("--query-ids", default="0-9")
    parser.add_argument("--batch-size", type=int, default=DAS_FEATURE_BATCH_SIZE)
    parser.add_argument("--lambda", dest="damping", type=float, default=100.0)
    args = parser.parse_args()
    query_ids = bundle.parse_query_ids(args.query_ids)
    device = torch.device(f"cuda:{args.gpu}")
    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    records = [r for r in manifest if int(r["query_id"]) in query_ids and r["family"] == args.family]
    if not records:
        raise ValueError(f"no queries selected for family={args.family}")

    model, dataset, _ = build_model(model_paths(args.family)[-1], "ema", device)
    model.eval()
    x_all, condition_all = preload_dataset(dataset, args.family, device)
    schedule = base.make_linear_schedule(T, device=device)
    named = dict(model.named_parameters())
    names = tuple(named)
    parameters = tuple(named.values())
    projection_dim = 4096
    specs = build_countsketch_specs(
        list(parameters), projection_dim, device=device,
        seed_parts=(TRAIN_SEED, "vector_trajectory_das_bundle", args.family),
    )
    membership = torch.from_numpy(np.load(MASK_DIR / "membership.npy").astype(np.float32)).to(device)
    mask_count = membership.shape[0]
    gram = torch.zeros((projection_dim, projection_dim), device=device, dtype=torch.float32)
    subset_features = torch.zeros((mask_count, projection_dim), device=device, dtype=torch.float32)
    train_mc = 10

    def point_loss(params, x0, condition, timesteps, noises):
        x = x0.unsqueeze(0).expand(train_mc, *x0.shape)
        c = condition.unsqueeze(0).expand(train_mc, condition.shape[-1])
        xt = base.q_sample(x, timesteps, noises, schedule)
        prediction = functional_call(model, params, (xt, timesteps, c))
        return (prediction - noises).square().mean()

    batched_gradient = vmap(grad(point_loss), in_dims=(None, 0, 0, 0, 0))
    batches = math.ceil(N_TRAIN / args.batch_size)
    for position, start in enumerate(range(0, N_TRAIN, args.batch_size), start=1):
        end = min(start + args.batch_size, N_TRAIN)
        count = end - start
        x, condition = x_all[start:end], condition_all[start:end]
        generator = make_torch_generator(device, TRAIN_SEED, METHOD, args.family, start)
        timesteps = torch.randint(0, T, (count, train_mc), generator=generator, device=device).long()
        noises = torch.randn((count, train_mc, *x.shape[1:]), generator=generator, device=device, dtype=x.dtype)
        gradients = batched_gradient(named, x, condition, timesteps, noises)
        features = _project_batched_grads(
            gradients, names, specs, projection_dim,
            bool(DAS_NORMALIZE_PROJECTED_GRADS), 1e-8,
        )
        gram.addmm_(features.T, features)
        subset_features.add_(membership[:, start:end] @ features)
        if position == 1 or position == batches or position % max(1, batches // 20) == 0:
            print(f"[vector-das] train batch={position}/{batches} points={end}/{N_TRAIN}", flush=True)

    eye = torch.eye(projection_dim, device=device)
    subset_directions = torch.linalg.solve(
        gram + float(args.damping) * eye, subset_features.T
    ).T
    output_root = ATTR_DIR / METHOD
    scores = {}
    for record in records:
        qid = int(record["query_id"])
        trajectory = np.load(Path(record["dir"]) / "trajectory_xt.npy")
        trajectory_t = np.load(Path(record["dir"]) / "trajectory_t.npy")
        condition = cond_for(record, dataset, device)
        vectors = np.empty((mask_count, len(trajectory_t), 27), dtype=np.float32)
        for timestamp_position, timestep_value in enumerate(trajectory_t):
            xt = torch.from_numpy(trajectory[timestamp_position]).to(device=device, dtype=torch.float32)
            timestep = torch.tensor([int(timestep_value)], device=device, dtype=torch.long)
            jacobian = bundle.projected_output_jacobian(
                model, parameters, names, specs, xt, timestep, condition
            )
            vectors[:, timestamp_position] = (subset_directions @ jacobian.T).detach().cpu().numpy()
        bundle_scores = np.mean(np.sum(np.square(vectors.astype(np.float64)), axis=-1), axis=-1)
        root = output_root / f"q{qid:02d}"
        root.mkdir(parents=True, exist_ok=True)
        np.save(root / "subset_vectors.npy", vectors)
        np.save(root / "bundle_scores.npy", bundle_scores)
        with open(root / "info.json", "w") as handle:
            json.dump({"method": METHOD, "query": record, "lambda": args.damping, "train_mc": train_mc, "projection_dim": projection_dim, "vector_shape": list(vectors.shape), "score": "mean_t ||J_t (C+lambda I)^-1 sum_{i in S} g_i||^2"}, handle, indent=2)
        scores[qid] = bundle_scores
        print(f"[saved] q{qid:02d} vectors={vectors.shape}", flush=True)

    result = {"method": METHOD, "query_ids": [int(r["query_id"]) for r in records], "lambda": args.damping, "metrics": {}}
    print(f"\nMETHOD: {METHOD}")
    for metric in LDS_METRICS:
        observed = np.load(LDS_DIR / f"observed_{metric}.npy").astype(np.float64)
        values = np.array([spearmanr(scores[int(r["query_id"])], observed[int(r["query_id"])]).statistic for r in records])
        result["metrics"][metric] = {"positive": {"mean": float(np.nanmean(values)), "std": float(np.nanstd(values)), "per_query": values.tolist()}, "negative": {"mean": float(np.nanmean(-values)), "std": float(np.nanstd(-values)), "per_query": (-values).tolist()}}
        print(f"{metric:30s} sign=+1 {np.nanmean(values):+.6f}±{np.nanstd(values):.6f} | sign=-1 {np.nanmean(-values):+.6f}±{np.nanstd(values):.6f}")
    output = LDS_DIR / f"{METHOD}_q{result['query_ids'][0]:02d}_q{result['query_ids'][-1]:02d}.json"
    with open(output, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"[saved] {output}")


if __name__ == "__main__":
    main()
