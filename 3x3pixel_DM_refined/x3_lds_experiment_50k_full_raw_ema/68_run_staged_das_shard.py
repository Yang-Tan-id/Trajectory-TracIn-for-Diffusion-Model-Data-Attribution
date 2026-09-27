"""One term shard of original DAS: final staged EMA and all 50k points."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call, grad, vmap

import x3pixel_DM_training as base
from attribution_one_query import _project_batched_grads, build_model, cond_for, preload_dataset
from staged_lds_config import *
from x3_endpoint_das_jax_logic_pytorch import build_countsketch_specs, compute_projected_eps_feature, make_torch_generator, sample_noise, sample_output_probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--term-shard-index", type=int, required=True)
    parser.add_argument("--term-shard-count", type=int, default=4)
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    root = STAGED_ATTR_DIR / "_das_shards" / f"shard_{args.term_shard_index:02d}_of_{args.term_shard_count:02d}"
    if (root / "done.json").is_file():
        print(f"[skip] {root}", flush=True)
        return
    with open(STAGED_QUERY_DIR / "manifest.json") as handle:
        records = json.load(handle)
    model, dataset, _ = build_model(staged_base_checkpoint(STAGED_EPOCHS), "ema", device)
    x_all, cond_all = preload_dataset(dataset, STAGED_FAMILY, device)
    sched = base.make_linear_schedule(T, device=device)
    endpoints = [torch.from_numpy(np.load(Path(r["dir"]) / "final_state.npy")).to(device=device, dtype=torch.float32) for r in records]
    conditions = [cond_for(r, dataset, device) for r in records]
    named = dict(model.named_parameters())
    names = tuple(named)
    active = tuple(named.values())
    d = int(DAS_PROJ_DIM)
    train_mc = int(DAS_TRAIN_GRAD_MC)
    batch_size = int(DAS_FEATURE_BATCH_SIZE)
    normalize = bool(DAS_NORMALIZE_PROJECTED_GRADS)
    epsilon = 1e-8
    all_terms = [(int(t), mc) for t in DAS_TIMESTEPS for mc in range(int(DAS_NUM_MC))]
    terms = all_terms[args.term_shard_index::args.term_shard_count]
    scores = {float(lam): torch.zeros((len(records), N_TRAIN), device=device, dtype=torch.float64) for lam in DAS_LAMBDAS}
    for term_position, (tval, mc_index) in enumerate(terms, start=1):
        specs = build_countsketch_specs(
            list(active), d, device=device,
            seed_parts=(808, "pdas_gradient_projection", 0, tval, mc_index),
        )
        probe_rng = make_torch_generator(device, 808, "pdas_output_probe", 0, tval, mc_index)
        output_probe = sample_output_probe(tuple(endpoints[0].shape), device=device, rng=probe_rng)
        query_rng = make_torch_generator(device, 808, "pdas_q", 0, tval, mc_index)
        query_noise = sample_noise(endpoints[0], rng=query_rng)
        t_query = torch.tensor([tval], device=device)
        query_features = []
        for endpoint, condition in zip(endpoints, conditions):
            _, phi_q = compute_projected_eps_feature(
                model=model, active=list(active), sched=sched, x0=endpoint,
                cond=condition, t=t_query, noise=query_noise,
                output_probe=output_probe, projection_specs=specs, proj_dim=d,
                device=device, normalize_projected_grads=normalize,
                normalize_eps=epsilon,
            )
            query_features.append(phi_q.float().detach())
        query_features = torch.stack(query_features)
        probe = output_probe[0]
        probe_scale = math.sqrt(float(probe.numel()))
        t_train = torch.full((train_mc,), tval, device=device, dtype=torch.long)

        def scalar_feature(parameter_dict, x0, condition, noises):
            xb = x0.unsqueeze(0).expand(train_mc, *x0.shape)
            cb = condition.unsqueeze(0).expand(train_mc, condition.shape[-1])
            xt = base.q_sample(xb, t_train, noises, sched)
            prediction = functional_call(model, parameter_dict, (xt, t_train, cb))
            return ((prediction * probe.unsqueeze(0)).reshape(train_mc, -1).sum(dim=1) / probe_scale).mean()

        batched_gradient = vmap(grad(scalar_feature), in_dims=(None, 0, 0, 0))
        phi_cache = torch.empty((N_TRAIN, d), device=device)
        residual_cache = torch.empty(N_TRAIN, device=device)
        gram = torch.zeros((d, d), device=device)
        for start in range(0, N_TRAIN, batch_size):
            end = min(start + batch_size, N_TRAIN)
            xb, cb = x_all[start:end], cond_all[start:end]
            generator = make_torch_generator(device, 808, "pdas_tr_batch", 0, tval, mc_index, start)
            noises = torch.randn((end-start, train_mc, *xb.shape[1:]), generator=generator, device=device, dtype=xb.dtype)
            gradients = batched_gradient(named, xb, cb, noises)
            phi = _project_batched_grads(gradients, names, specs, d, normalize, epsilon)
            phi_cache[start:end] = phi
            gram.addmm_(phi.T, phi)
            xb_mc = xb[:, None].expand(end-start, train_mc, *xb.shape[1:]).reshape((end-start)*train_mc, *xb.shape[1:])
            cb_mc = cb[:, None].expand(end-start, train_mc, cb.shape[-1]).reshape((end-start)*train_mc, cb.shape[-1])
            noise_flat = noises.reshape((end-start)*train_mc, *xb.shape[1:])
            t_flat = torch.full(((end-start)*train_mc,), tval, device=device, dtype=torch.long)
            with torch.no_grad():
                prediction = model(base.q_sample(xb_mc, t_flat, noise_flat, sched), t_flat, cb_mc)
                residual_cache[start:end] = (((prediction-noise_flat)*probe.unsqueeze(0)).reshape(end-start, train_mc, -1).sum(dim=2).mean(dim=1)/probe_scale)
        eye = torch.eye(d, device=device)
        for lam in DAS_LAMBDAS:
            lam = float(lam)
            solved_queries = torch.linalg.solve(gram + lam * eye, query_features.T)
            raw = (phi_cache @ solved_queries).to(torch.float64)
            raw *= residual_cache.to(torch.float64).unsqueeze(1)
            if DAS_USE_SM_DENOMINATOR:
                solved_train = torch.linalg.solve(gram + lam * eye, phi_cache.T).T
                denominator = 1.0 - (phi_cache * solved_train).sum(dim=1).to(torch.float64)
                denominator = torch.where(denominator.abs() < 1e-6, denominator.sign() * 1e-6, denominator)
                raw /= denominator.unsqueeze(1)
            scores[lam] += raw.T.square()
        print(f"[das gpu={args.gpu}] term={term_position}/{len(terms)} t={tval} mc={mc_index}", flush=True)
        del phi_cache, residual_cache, gram, eye, query_features, gradients, phi
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root / "scores.npz", **{f"lambda_{str(lam).replace('.', 'p')}": value.cpu().numpy() for lam, value in scores.items()})
    with open(root / "done.json", "w") as handle:
        json.dump({"term_count": len(terms), "terms": terms, "parameter_source": "final_ema", "train_pool": "all_50000"}, handle, indent=2)


if __name__ == "__main__":
    main()
