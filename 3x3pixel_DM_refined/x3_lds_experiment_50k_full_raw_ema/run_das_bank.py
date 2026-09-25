"""Run EMA DAS for a query family, sharing every 100x10 train/Gram term."""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call, grad, vmap

import x3pixel_DM_training as base
from attribution_one_query import (
    _project_batched_grads, build_model, cond_for, model_paths, preload_dataset,
)
from exp_config import *
from x3_endpoint_das_jax_logic_pytorch import (
    build_countsketch_specs, compute_projected_eps_feature, make_torch_generator,
    sample_noise, sample_output_probe,
)


def tag(value):
    return str(float(value)).replace(".", "p")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    with open(QUERY_DIR / "manifest.json") as handle:
        records = [r for r in json.load(handle) if r["family"] == args.family]
    complete = all(
        (ATTR_DIR / "das_ema" / f"q{int(r['query_id']):02d}" / f"lambda_{tag(lam)}" / "scores.npy").is_file()
        for r in records for lam in DAS_LAMBDAS
    )
    if complete:
        print(f"[skip] DAS bank complete for {args.family}", flush=True)
        return

    final_ckpt = model_paths(args.family)[-1]
    model, ds, _ = build_model(final_ckpt, "ema", device)
    sched = base.make_linear_schedule(T, device=device)
    x_all, cond_all = preload_dataset(ds, args.family, device)
    endpoints = [
        torch.from_numpy(np.load(Path(r["dir"]) / "final_state.npy")).to(device=device, dtype=torch.float32)
        for r in records
    ]
    conditions = [cond_for(r, ds, device) for r in records]
    named = dict(model.named_parameters())
    names = tuple(named)
    params_dict = dict(named)
    active = tuple(named.values())
    d = int(DAS_PROJ_DIM)
    batch_size = int(DAS_FEATURE_BATCH_SIZE)
    train_mc = int(DAS_TRAIN_GRAD_MC)
    normalize = bool(DAS_NORMALIZE_PROJECTED_GRADS)
    eps = 1e-8
    q_count = len(records)
    scores = {
        float(lam): torch.zeros((q_count, N_TRAIN), device=device, dtype=torch.float64)
        for lam in DAS_LAMBDAS
    }
    total_terms = len(DAS_TIMESTEPS) * int(DAS_NUM_MC)
    completed_terms = 0
    started = time.perf_counter()

    for tval_raw in DAS_TIMESTEPS:
        tval = int(tval_raw)
        t_q = torch.tensor([tval], device=device, dtype=torch.long)
        for mc_i in range(int(DAS_NUM_MC)):
            specs = build_countsketch_specs(
                list(active), d, device=device,
                seed_parts=(808, "pdas_gradient_projection", 0, tval, mc_i),
            )
            probe_rng = make_torch_generator(device, 808, "pdas_output_probe", 0, tval, mc_i)
            output_probe = sample_output_probe(tuple(endpoints[0].shape), device=device, rng=probe_rng)
            query_rng = make_torch_generator(device, 808, "pdas_q", 0, tval, mc_i)
            query_noise = sample_noise(endpoints[0], rng=query_rng)
            phi_queries = []
            for endpoint, condition in zip(endpoints, conditions):
                _, phi_q = compute_projected_eps_feature(
                    model=model, active=list(active), sched=sched, x0=endpoint,
                    cond=condition, t=t_q, noise=query_noise,
                    output_probe=output_probe, projection_specs=specs, proj_dim=d,
                    device=device, normalize_projected_grads=normalize,
                    normalize_eps=eps,
                )
                phi_queries.append(phi_q.to(torch.float32))
            phi_queries = torch.stack(phi_queries)

            probe_single = output_probe[0]
            scalar_denom = math.sqrt(float(probe_single.numel()))
            t_train = torch.full((train_mc,), tval, device=device, dtype=torch.long)

            def single_scalar(pdict, x0, cond, noises):
                xb = x0.unsqueeze(0).expand(train_mc, *x0.shape)
                cb = cond.unsqueeze(0).expand(train_mc, cond.shape[-1])
                xt = base.q_sample(xb, t_train, noises, sched)
                pred = functional_call(model, pdict, (xt, t_train, cb))
                values = (pred * probe_single.unsqueeze(0)).reshape(train_mc, -1).sum(dim=1)
                return (values / scalar_denom).mean()

            batched_grad = vmap(grad(single_scalar), in_dims=(None, 0, 0, 0))
            phi_cache = torch.empty((N_TRAIN, d), device=device, dtype=torch.float32)
            residual_cache = torch.empty(N_TRAIN, device=device, dtype=torch.float32)
            gram = torch.zeros((d, d), device=device, dtype=torch.float32)
            for start in range(0, N_TRAIN, batch_size):
                end = min(start + batch_size, N_TRAIN)
                xb, cb = x_all[start:end], cond_all[start:end]
                bsize = end - start
                generator = make_torch_generator(
                    device, 808, "pdas_tr_batch", 0, tval, mc_i, start
                )
                noises = torch.randn(
                    (bsize, train_mc, *xb.shape[1:]), generator=generator,
                    device=device, dtype=xb.dtype,
                )
                grads_b = batched_grad(params_dict, xb, cb, noises)
                phi = _project_batched_grads(grads_b, names, specs, d, normalize, eps)
                phi_cache[start:end] = phi
                gram.addmm_(phi.T, phi)

                xb_mc = xb[:, None].expand(bsize, train_mc, *xb.shape[1:]).reshape(bsize * train_mc, *xb.shape[1:])
                cb_mc = cb[:, None].expand(bsize, train_mc, cb.shape[-1]).reshape(bsize * train_mc, cb.shape[-1])
                noise_flat = noises.reshape(bsize * train_mc, *xb.shape[1:])
                t_flat = torch.full((bsize * train_mc,), tval, device=device, dtype=torch.long)
                with torch.no_grad():
                    pred = model(base.q_sample(xb_mc, t_flat, noise_flat, sched), t_flat, cb_mc)
                    residual_cache[start:end] = (
                        ((pred - noise_flat) * probe_single.unsqueeze(0))
                        .reshape(bsize, train_mc, -1).sum(dim=2).mean(dim=1) / scalar_denom
                    )

            eye = torch.eye(d, device=device, dtype=torch.float32)
            for lam in DAS_LAMBDAS:
                lam = float(lam)
                solved_queries = torch.linalg.solve(gram + lam * eye, phi_queries.T)
                raw = (phi_cache @ solved_queries).to(torch.float64)
                raw *= residual_cache.to(torch.float64).unsqueeze(1)
                if DAS_USE_SM_DENOMINATOR:
                    solved_train = torch.linalg.solve(gram + lam * eye, phi_cache.T).T
                    denom = 1.0 - (phi_cache * solved_train).sum(dim=1).to(torch.float64)
                    denom = torch.where(denom.abs() < 1e-6, denom.sign() * 1e-6, denom)
                    raw /= denom.unsqueeze(1)
                scores[lam] += raw.T.square()

            completed_terms += 1
            elapsed = time.perf_counter() - started
            eta = elapsed / completed_terms * (total_terms - completed_terms)
            print(
                f"[das-bank {args.family}] term {completed_terms}/{total_terms} "
                f"train_mc={train_mc} elapsed={elapsed/3600:.2f}h eta={eta/3600:.2f}h",
                flush=True,
            )

    for qi, record in enumerate(records):
        for lam, values in scores.items():
            out = ATTR_DIR / "das_ema" / f"q{int(record['query_id']):02d}" / f"lambda_{tag(lam)}"
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / "scores.npy", (values[qi] / total_terms).cpu().numpy())
            with open(out / "info.json", "w") as handle:
                json.dump({
                    "query": record, "proj_dim": d, "lambda": lam,
                    "param_source": "ema", "timesteps": list(DAS_TIMESTEPS),
                    "num_mc": DAS_NUM_MC, "train_gradient_mc": train_mc,
                    "normalize_projected_grads": normalize, "bank_scoring": True,
                }, handle, indent=2)


if __name__ == "__main__":
    main()
