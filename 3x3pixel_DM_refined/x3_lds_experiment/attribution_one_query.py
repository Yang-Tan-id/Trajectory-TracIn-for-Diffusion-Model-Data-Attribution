import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp, grad, vmap

import x3pixel_DM_training as base
from dataset_loader import ColorGridDataset
from exp_config import *

from x3_endpoint_das_jax_logic_pytorch import (
    build_countsketch_specs,
    compute_projected_eps_feature,
    make_torch_generator,
    sample_noise,
    sample_output_probe,
)


def nkey(p):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", p.name)]


def model_paths(family):
    return sorted((MODEL_DIR / "base" / family).glob("epoch_*.pt"), key=nkey)


def build_model(path, source, device):
    # weights_only=False is explicit because our own checkpoint contains metadata.
    ck = torch.load(path, map_location=device, weights_only=False)
    ds = ColorGridDataset(str(BASE_CSV), grid_size=3)
    m = base.CondEpsModel(3, len(ds.vocab), BASE_CH, TIME_DIM).to(device)
    m.load_state_dict(
        ck["model_state" if source == "raw" else "ema_model_state"],
        strict=True,
    )
    m.eval()
    return m, ds, ck


def cond_for(rec, ds, device):
    c = torch.zeros((1, len(ds.vocab)), device=device)
    for lab in rec["labels"]:
        c[0, ds.vocab[lab]] = 1.0
    return c


def tracin_lr_weight(ck):
    if not TRACIN_USE_LR_WEIGHTS:
        return 1.0
    if "eta" in ck:
        return float(ck["eta"])
    if "learning_rate_at_checkpoint" in ck:
        return float(ck["learning_rate_at_checkpoint"])

    step = max(0, int(ck.get("global_step", 1)) - 1)
    steps_per_epoch = math.ceil(N_TRAIN / BATCH_SIZE)
    total = EPOCHS * steps_per_epoch
    warm = int(math.ceil(total * WARMUP_RATIO))
    if step < warm:
        return PEAK_LR * float(step + 1) / max(1, warm)
    p = (step - warm) / max(1, total - warm)
    p = min(max(p, 0.0), 1.0)
    return 0.5 * PEAK_LR * (1.0 + math.cos(math.pi * p))


def preload_dataset(ds, family, device):
    xs, cs = [], []
    for i in range(len(ds)):
        x, c = ds[i]
        xs.append(x)
        cs.append(c)
    x_all = torch.stack(xs, dim=0).to(device=device, dtype=torch.float32)
    c_all = torch.stack(cs, dim=0).to(device=device, dtype=torch.float32)
    if family == "unprompted":
        c_all.zero_()
    return x_all, c_all


def run_traj(rec, target_mode, device):
    """
    Exact same Traj score, faster execution.

    Key identity:
        <g_q, grad_theta L_i>
      = JVP_theta[L_i](direction=g_q)

    Therefore one JVP of a VECTOR of per-example losses returns the exact
    query-gradient dot product for every example in a GPU batch. No
    per-example backward loop is necessary.

    MC samples are also evaluated in one (B * MC) forward.
    """
    family = rec["family"]
    paths = model_paths(family)
    ds = ColorGridDataset(str(BASE_CSV), grid_size=3)
    traj = np.load(Path(rec["dir"]) / "trajectory_xt.npy")
    t_seq = np.load(Path(rec["dir"]) / "trajectory_t.npy")
    sched = base.make_linear_schedule(T, device=device)
    cond_q = cond_for(rec, ds, device)

    x_all, cond_all = preload_dataset(ds, family, device)
    scores = torch.zeros((N_TRAIN,), device=device, dtype=torch.float64)

    batch_size = int(TRACIN_SCORE_BATCH_SIZE)
    mc_count = int(TRACIN_TRAIN_MC)
    snap_weight = 1.0 / float(len(t_seq))
    traj_t0 = time.perf_counter()

    ref_path = paths[-1]
    effective_total = len(paths) - (1 if target_mode == "next" else 0)
    completed = 0

    for ci, cur_path in enumerate(paths):
        if target_mode == "next" and ci + 1 >= len(paths):
            continue

        model, _, ck = build_model(cur_path, TRACIN_PARAM_SOURCE, device)
        target_path = ref_path if target_mode == "reference" else paths[ci + 1]
        target, _, _ = build_model(target_path, TRACIN_PARAM_SOURCE, device)

        named = dict(model.named_parameters())
        names = tuple(named.keys())
        params_tuple = tuple(named[n] for n in names)
        w_ck = tracin_lr_weight(ck)

        # Query gradient direction for every trajectory snapshot.
        gq = []
        for si, tval in enumerate(t_seq.tolist()):
            xt_q = torch.from_numpy(traj[si]).to(device=device, dtype=torch.float32)
            t_q = torch.tensor([int(tval)], device=device, dtype=torch.long)
            with torch.no_grad():
                eps_tgt = target(xt_q, t_q, cond_q).detach()
            f_q = (model(xt_q, t_q, cond_q) - eps_tgt).pow(2).sum()
            grads_q = torch.autograd.grad(f_q, params_tuple, allow_unused=False)
            gq.append(tuple(g.detach() for g in grads_q))

        # Ref at the exact reference checkpoint is mathematically zero.
        all_query_zero = all(
            all(bool(torch.count_nonzero(g).item() == 0) for g in gs)
            for gs in gq
        )
        if all_query_zero:
            completed += 1
            print(
                f"[traj {target_mode}] ckpt {completed}/{effective_total} "
                f"query gradient is exactly zero -> skipped",
                flush=True,
            )
            del model, target, gq
            continue

        ck_t0 = time.perf_counter()
        num_batches = math.ceil(N_TRAIN / batch_size)

        for si, tval in enumerate(t_seq.tolist()):
            tangent = gq[si]
            tval = int(tval)

            for bi, start in enumerate(range(0, N_TRAIN, batch_size)):
                end = min(start + batch_size, N_TRAIN)
                xb = x_all[start:end]
                cb = cond_all[start:end]
                B = xb.shape[0]

                # One deterministic generator per batch/term. Ref and Next use
                # the same train-noise stream for the same checkpoint/snapshot.
                gen = make_torch_generator(
                    device,
                    70000000,
                    "traj_train",
                    ci,
                    si,
                    start,
                    mc_count,
                )
                noise = torch.randn(
                    (B, mc_count, *xb.shape[1:]),
                    generator=gen,
                    device=device,
                    dtype=xb.dtype,
                )

                xb_mc = (
                    xb[:, None]
                    .expand(B, mc_count, *xb.shape[1:])
                    .reshape(B * mc_count, *xb.shape[1:])
                )
                cb_mc = (
                    cb[:, None]
                    .expand(B, mc_count, cb.shape[-1])
                    .reshape(B * mc_count, cb.shape[-1])
                )
                noise_flat = noise.reshape(B * mc_count, *xb.shape[1:])
                t_flat = torch.full(
                    (B * mc_count,),
                    tval,
                    device=device,
                    dtype=torch.long,
                )
                xt_flat = base.q_sample(xb_mc, t_flat, noise_flat, sched)

                def per_example_loss(*param_values):
                    pd = {n: p for n, p in zip(names, param_values)}
                    pred = functional_call(
                        model,
                        pd,
                        (xt_flat, t_flat, cb_mc),
                    )
                    per_mc = (
                        (pred - noise_flat)
                        .pow(2)
                        .reshape(B, mc_count, -1)
                        .mean(dim=-1)
                    )
                    return per_mc.mean(dim=1)  # [B]

                _, dots = jvp(
                    per_example_loss,
                    params_tuple,
                    tangent,
                )

                scores[start:end] += (
                    dots.detach().to(torch.float64)
                    * float(snap_weight)
                    * float(w_ck)
                )

            done_snap = si + 1
            if done_snap == 1 or done_snap % 5 == 0 or done_snap == len(t_seq):
                elapsed = time.perf_counter() - ck_t0
                eta = elapsed / done_snap * (len(t_seq) - done_snap)
                print(
                    f"[traj-fast {target_mode}] ckpt {completed+1}/{effective_total} | "
                    f"snap {done_snap}/{len(t_seq)} | "
                    f"batch={batch_size} mc={mc_count} | "
                    f"elapsed={elapsed/60:.1f}m eta≈{eta/60:.1f}m",
                    flush=True,
                )

        completed += 1
        total_elapsed = time.perf_counter() - traj_t0
        avg = total_elapsed / completed
        eta_total = avg * max(0, effective_total - completed)
        print(
            f"[traj-fast {target_mode}] ckpt {completed}/{effective_total} done | "
            f"total_elapsed={total_elapsed/3600:.2f}h | "
            f"query_eta≈{eta_total/3600:.2f}h",
            flush=True,
        )

        del model, target, gq
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = (
        ATTR_DIR
        / ("traj_ref" if target_mode == "reference" else "traj_next")
        / f"q{rec['query_id']:02d}"
    )
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "scores.npy", scores.detach().cpu().numpy())

    with open(out / "info.json", "w") as f:
        json.dump(
            {
                "query": rec,
                "target": target_mode,
                "param_source": TRACIN_PARAM_SOURCE,
                "train_mc": TRACIN_TRAIN_MC,
                "lr_weighted": TRACIN_USE_LR_WEIGHTS,
                "num_checkpoints": len(paths),
                "num_snapshots": len(t_seq),
                "backend": "torch.func.jvp_batched_exact_dot",
                "score_batch_size": int(batch_size),
            },
            f,
            indent=2,
        )


def _project_batched_grads(grads_dict, names, projection_specs, d, normalize, eps):
    first = grads_dict[names[0]]
    B = first.shape[0]
    device = first.device
    out = torch.zeros((B, d), device=device, dtype=torch.float32)

    for name, (idx, sign) in zip(names, projection_specs):
        g = grads_dict[name].reshape(B, -1).to(torch.float32)
        src = g * sign.unsqueeze(0)
        out.scatter_add_(
            1,
            idx.unsqueeze(0).expand(B, -1),
            src,
        )

    out /= math.sqrt(float(d))

    if normalize:
        norms = torch.linalg.vector_norm(out, dim=1, keepdim=True)
        out = out / (norms + float(eps))

    return out


def run_das(rec, device):
    """
    Same later-JAX DAS logic, but feature construction is vmap'ed over a batch
    of train examples and the 4096-D Gram / lambda solves stay on GPU.
    """
    family = rec["family"]
    final_ckpt = model_paths(family)[-1]
    model, ds, _ = build_model(final_ckpt, DAS_PARAM_SOURCE, device)
    sched = base.make_linear_schedule(T, device=device)

    x_all, cond_all = preload_dataset(ds, family, device)
    x0_ref = torch.from_numpy(
        np.load(Path(rec["dir"]) / "final_state.npy")
    ).to(device=device, dtype=torch.float32)
    cond_q = cond_for(rec, ds, device)

    named = dict(model.named_parameters())
    names = tuple(named.keys())
    params_dict = {n: named[n] for n in names}
    active = tuple(named[n] for n in names)

    d = int(DAS_PROJ_DIM)
    batch_size = int(DAS_FEATURE_BATCH_SIZE)
    normalize = bool(DAS_NORMALIZE_PROJECTED_GRADS)
    normalize_eps = 1e-8

    scores_by_lambda = {
        float(l): torch.zeros(N_TRAIN, device=device, dtype=torch.float64)
        for l in DAS_LAMBDAS
    }
    total_terms = 0

    for tval in DAS_TIMESTEPS:
        tval = int(tval)
        t_q = torch.tensor([tval], device=device, dtype=torch.long)

        for mc_i in range(int(DAS_NUM_MC)):
            term_t0 = time.perf_counter()

            projection_specs = build_countsketch_specs(
                list(active),
                d,
                device=device,
                seed_parts=(
                    808,
                    "pdas_gradient_projection",
                    0,
                    tval,
                    mc_i,
                ),
            )

            rng_probe = make_torch_generator(
                device, 808, "pdas_output_probe", 0, tval, mc_i
            )
            output_probe = sample_output_probe(
                tuple(x0_ref.shape),
                device=device,
                rng=rng_probe,
            )

            rng_q = make_torch_generator(
                device, 808, "pdas_q", 0, tval, mc_i
            )
            noise_q = sample_noise(x0_ref, rng=rng_q)

            query_residual, phi_q = compute_projected_eps_feature(
                model=model,
                active=list(active),
                sched=sched,
                x0=x0_ref,
                cond=cond_q,
                t=t_q,
                noise=noise_q,
                output_probe=output_probe,
                projection_specs=projection_specs,
                proj_dim=d,
                device=device,
                normalize_projected_grads=normalize,
                normalize_eps=normalize_eps,
            )

            probe_single = output_probe[0]
            scalar_denom = math.sqrt(float(probe_single.numel()))

            def single_scalar_eps(pdict, x0, cond, noise):
                xb = x0.unsqueeze(0)
                cb = cond.unsqueeze(0)
                nb = noise.unsqueeze(0)
                xt = base.q_sample(xb, t_q, nb, sched)
                pred = functional_call(model, pdict, (xt, t_q, cb))
                return (pred[0] * probe_single).sum() / scalar_denom

            single_grad = grad(single_scalar_eps)
            batched_grad = vmap(
                single_grad,
                in_dims=(None, 0, 0, 0),
            )

            phi_cache = torch.empty(
                (N_TRAIN, d),
                device=device,
                dtype=torch.float32,
            )
            residual_cache = torch.empty(
                (N_TRAIN,),
                device=device,
                dtype=torch.float32,
            )
            H_base = torch.zeros(
                (d, d),
                device=device,
                dtype=torch.float32,
            )

            num_batches = math.ceil(N_TRAIN / batch_size)
            feat_t0 = time.perf_counter()

            for bi, start in enumerate(range(0, N_TRAIN, batch_size)):
                end = min(start + batch_size, N_TRAIN)
                xb = x_all[start:end]
                cb = cond_all[start:end]
                B = xb.shape[0]

                gen = make_torch_generator(
                    device,
                    808,
                    "pdas_tr_batch",
                    0,
                    tval,
                    mc_i,
                    start,
                )
                noise = torch.randn(
                    xb.shape,
                    generator=gen,
                    device=device,
                    dtype=xb.dtype,
                )

                grads_b = batched_grad(
                    params_dict,
                    xb,
                    cb,
                    noise,
                )
                phi_b = _project_batched_grads(
                    grads_b,
                    names,
                    projection_specs,
                    d,
                    normalize,
                    normalize_eps,
                )

                t_batch = torch.full(
                    (B,),
                    tval,
                    device=device,
                    dtype=torch.long,
                )
                xt = base.q_sample(xb, t_batch, noise, sched)
                with torch.no_grad():
                    pred = model(xt, t_batch, cb)
                    residual_b = (
                        ((pred - noise) * probe_single.unsqueeze(0))
                        .reshape(B, -1)
                        .sum(dim=1)
                        / scalar_denom
                    )

                phi_cache[start:end] = phi_b
                residual_cache[start:end] = residual_b
                H_base.addmm_(phi_b.T, phi_b)

                done_b = bi + 1
                if done_b == 1 or done_b % 10 == 0 or done_b == num_batches:
                    elapsed = time.perf_counter() - feat_t0
                    eta = elapsed / done_b * (num_batches - done_b)
                    print(
                        f"[das-fast] features {end}/{N_TRAIN} "
                        f"({100.0*end/N_TRAIN:.1f}%) | "
                        f"batch={batch_size} | elapsed={elapsed/60:.1f}m "
                        f"eta≈{eta/60:.1f}m",
                        flush=True,
                    )

            eye = torch.eye(d, device=device, dtype=torch.float32)
            phi_q = phi_q.to(device=device, dtype=torch.float32)

            for lam in DAS_LAMBDAS:
                lam = float(lam)
                solve_t0 = time.perf_counter()
                H = H_base + lam * eye
                u = torch.linalg.solve(H, phi_q)

                raw = (
                    (phi_cache @ u).to(torch.float64)
                    * residual_cache.to(torch.float64)
                )

                if DAS_USE_SM_DENOMINATOR:
                    # Expensive optional correction; default is False.
                    solved = torch.linalg.solve(H, phi_cache.T).T
                    leverage = (phi_cache * solved).sum(dim=1).to(torch.float64)
                    denom = 1.0 - leverage
                    tiny = denom.abs() < 1e-6
                    denom = torch.where(
                        tiny,
                        torch.where(
                            denom >= 0,
                            torch.full_like(denom, 1e-6),
                            torch.full_like(denom, -1e-6),
                        ),
                        denom,
                    )
                    raw = raw / denom

                scores_by_lambda[lam] += raw.square()
                print(
                    f"[das-fast] lambda={lam:g} solve+score "
                    f"{time.perf_counter()-solve_t0:.2f}s",
                    flush=True,
                )

            total_terms += 1
            print(
                f"[das-fast] term done | query_residual={query_residual:.6f} | "
                f"elapsed={(time.perf_counter()-term_t0)/60:.1f}m",
                flush=True,
            )

    out0 = ATTR_DIR / "das" / f"q{rec['query_id']:02d}"
    for lam, s in scores_by_lambda.items():
        s = s / max(1, total_terms)
        out = out0 / f"lambda_{str(lam).replace('.','p')}"
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "scores.npy", s.detach().cpu().numpy())
        with open(out / "info.json", "w") as f:
            json.dump(
                {
                    "query": rec,
                    "proj_dim": DAS_PROJ_DIM,
                    "lambda": lam,
                    "param_source": DAS_PARAM_SOURCE,
                    "timesteps": list(DAS_TIMESTEPS),
                    "num_mc": DAS_NUM_MC,
                    "normalize_projected_grads": DAS_NORMALIZE_PROJECTED_GRADS,
                    "backend": "torch.func.vmap_batched_gpu_gram",
                    "feature_batch_size": int(batch_size),
                },
                f,
                indent=2,
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-json", required=True)
    ap.add_argument(
        "--method",
        choices=["traj_ref", "traj_next", "das"],
        required=True,
    )
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()

    with open(a.query_json) as f:
        rec = json.load(f)

    device = torch.device(
        f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu"
    )
    print(
        f"[device] {device} | "
        f"name={torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'}",
        flush=True,
    )

    if a.method == "traj_ref":
        run_traj(rec, "reference", device)
    elif a.method == "traj_next":
        run_traj(rec, "next", device)
    else:
        run_das(rec, device)


if __name__ == "__main__":
    main()
