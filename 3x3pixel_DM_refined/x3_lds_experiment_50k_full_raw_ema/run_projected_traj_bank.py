"""Score a whole query family while computing each projected train gradient once."""

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
    _project_batched_grads,
    _project_gradient_tuple,
    build_model,
    cond_for,
    model_paths,
    preload_dataset,
    tracin_lr_weight,
)
from dataset_loader import ColorGridDataset
from exp_config import *
from x3_endpoint_das_jax_logic_pytorch import build_countsketch_specs, make_torch_generator


def output_methods():
    return tuple(
        f"traj_projected_{order}_raw_{contraction}"
        for order in ("first", "second")
        for contraction in TRACIN_CONTRACTIONS
    )


def family_complete(records):
    return all(
        (ATTR_DIR / method / f"q{int(record['query_id']):02d}" / "scores.npy").is_file()
        for record in records
        for method in output_methods()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    with open(QUERY_DIR / "manifest.json") as handle:
        records = [r for r in json.load(handle) if r["family"] == args.family]
    if not records or family_complete(records):
        print(f"[skip] projected Traj bank complete for {args.family}", flush=True)
        return

    ds = ColorGridDataset(str(BASE_CSV), grid_size=3)
    paths = model_paths(args.family)
    sched = base.make_linear_schedule(T, device=device)
    x_all, cond_all = preload_dataset(ds, args.family, device)
    trajectories = [np.load(Path(r["dir"]) / "trajectory_xt.npy") for r in records]
    timestep_arrays = [np.load(Path(r["dir"]) / "trajectory_t.npy") for r in records]
    t_seq = timestep_arrays[0]
    if len(t_seq) != 100 or any(not np.array_equal(t_seq, values) for values in timestep_arrays):
        raise ValueError("all bank queries must share the same 100 trajectory timestamps")
    conditions = [cond_for(r, ds, device) for r in records]

    q_count = len(records)
    d = int(TRACIN_PROJ_DIM)
    batch_size = int(TRACIN_PROJECTED_BATCH_SIZE)
    mc_count = int(TRACIN_TRAIN_MC)
    coefficient = float(TRACIN_SECOND_ORDER_COEFFICIENT)
    snap_weight = 1.0 / len(t_seq)
    linear = {o: torch.zeros((q_count, N_TRAIN), device=device, dtype=torch.float64) for o in ("first", "second")}
    term_square = {o: torch.zeros_like(linear[o]) for o in linear}
    timestamp_square = {o: torch.zeros_like(linear[o]) for o in linear}
    started = time.perf_counter()

    # Timestamp-major traversal keeps only one QxN timestamp accumulator in memory.
    for si, tval_raw in enumerate(t_seq.tolist()):
        tval = int(tval_raw)
        timestamp_acc = {o: torch.zeros_like(linear[o]) for o in linear}

        for ci, cur_path in enumerate(paths[:-1]):
            model, _, ck = build_model(cur_path, "raw", device)
            target, _, _ = build_model(paths[ci + 1], "raw", device)
            named = dict(model.named_parameters())
            names = tuple(named)
            params = tuple(named.values())
            params_dict = dict(named)
            next_named = dict(target.named_parameters())
            delta = tuple(next_named[n].detach() - named[n].detach() for n in names)
            specs = build_countsketch_specs(
                list(params), d, device=device,
                seed_parts=(TRAIN_SEED, "traj_tracin_projection", ci),
            )

            first_queries, second_queries = [], []
            t_q = torch.tensor([tval], device=device, dtype=torch.long)
            for trajectory, condition in zip(trajectories, conditions):
                xt_q = torch.from_numpy(trajectory[si]).to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    eps_target = target(xt_q, t_q, condition).detach()
                query_loss = (model(xt_q, t_q, condition) - eps_target).pow(2).sum()
                query_grad = torch.autograd.grad(query_loss, params, create_graph=True)
                directional = sum((g * direction).sum() for g, direction in zip(query_grad, delta))
                query_hvp = torch.autograd.grad(directional, params)
                first_queries.append(_project_gradient_tuple(query_grad, specs, d))
                second_queries.append(_project_gradient_tuple(
                    tuple(g + coefficient * h for g, h in zip(query_grad, query_hvp)), specs, d
                ))
            query_matrix = {
                "first": torch.stack(first_queries),
                "second": torch.stack(second_queries),
            }

            t_mc = torch.full((mc_count,), tval, device=device, dtype=torch.long)

            def single_mean_loss(pdict, x0, cond, noises):
                x_mc = x0.unsqueeze(0).expand(mc_count, *x0.shape)
                c_mc = cond.unsqueeze(0).expand(mc_count, cond.shape[-1])
                xt = base.q_sample(x_mc, t_mc, noises, sched)
                pred = functional_call(model, pdict, (xt, t_mc, c_mc))
                return (pred - noises).pow(2).reshape(mc_count, -1).mean(dim=1).mean()

            batched_grad = vmap(grad(single_mean_loss), in_dims=(None, 0, 0, 0))
            term_weight = float(tracin_lr_weight(ck)) * snap_weight
            for start in range(0, N_TRAIN, batch_size):
                end = min(start + batch_size, N_TRAIN)
                xb, cb = x_all[start:end], cond_all[start:end]
                generator = make_torch_generator(
                    device, TRAIN_SEED, "projected_traj_train", ci, si, start, mc_count
                )
                noises = torch.randn(
                    (end - start, mc_count, *xb.shape[1:]),
                    generator=generator, device=device, dtype=xb.dtype,
                )
                grads_b = batched_grad(params_dict, xb, cb, noises)
                phi = _project_batched_grads(grads_b, names, specs, d, False, 1e-8)
                for order in ("first", "second"):
                    dots = (phi @ query_matrix[order].T).T.to(torch.float64)
                    linear[order][:, start:end] += term_weight * dots
                    term_square[order][:, start:end] += term_weight * dots.square()
                    timestamp_acc[order][:, start:end] += term_weight * dots

            del model, target, query_matrix, first_queries, second_queries
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for order in ("first", "second"):
            timestamp_square[order] += timestamp_acc[order].square()
        print(
            f"[projected-bank {args.family}] timestamp {si+1}/100 queries={q_count} "
            f"elapsed={(time.perf_counter()-started)/3600:.2f}h",
            flush=True,
        )

    result_map = {
        ("first", "linear"): linear["first"],
        ("first", "timestamp_sum_squared"): timestamp_square["first"],
        ("first", "termwise_squared"): term_square["first"],
        ("second", "linear"): linear["second"],
        ("second", "timestamp_sum_squared"): timestamp_square["second"],
        ("second", "termwise_squared"): term_square["second"],
    }
    for qi, record in enumerate(records):
        for (order, contraction), values in result_map.items():
            method = f"traj_projected_{order}_raw_{contraction}"
            out = ATTR_DIR / method / f"q{int(record['query_id']):02d}"
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / "scores.npy", values[qi].cpu().numpy())
            with open(out / "info.json", "w") as handle:
                json.dump({
                    "query": record, "order": order, "target": "next",
                    "param_source": "raw", "projection": "countsketch",
                    "proj_dim": d, "num_snapshots": 100, "train_mc": mc_count,
                    "contraction": contraction, "lr_weighted": TRACIN_USE_LR_WEIGHTS,
                    "second_order_coefficient": coefficient if order == "second" else 0.0,
                    "second_order_direction": "next_checkpoint_parameter_delta" if order == "second" else "disabled",
                    "bank_scoring": True,
                }, handle, indent=2)


if __name__ == "__main__":
    main()
