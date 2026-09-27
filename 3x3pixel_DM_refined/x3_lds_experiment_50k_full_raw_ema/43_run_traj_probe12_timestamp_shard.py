"""Score one timestamp shard for twelve-probe first-order raw Traj."""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call, grad, vmap

import x3pixel_DM_training as base
from attribution_one_query import (
    _project_batched_grads,
    build_model,
    cond_for,
    model_paths,
    preload_dataset,
    tracin_lr_weight,
)
from dataset_loader import ColorGridDataset
from traj_probe12_config import *
from x3_endpoint_das_jax_logic_pytorch import (
    build_countsketch_specs,
    make_torch_generator,
)


def atomic_save_npy(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        np.save(handle, value)
    os.replace(temporary, path)


def atomic_save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary, path)


def output_probe_bank(timestamp_index, device):
    generator = make_torch_generator(
        device,
        TRAJ_PROBE12_PROBE_SEED,
        "x3_traj_probe12_timestamp_shared",
        int(timestamp_index),
    )
    probes = torch.randn(
        (TRAJ_PROBE12_NUM_PROBES, 1, 3, 3, 3),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    return probes / math.sqrt(float(probes[0].numel()))


def projected_query_probe_bank(model, params, names, specs, d, xt, t_q, condition, probes):
    prediction = model(xt, t_q, condition)
    gradients = torch.autograd.grad(
        prediction,
        params,
        grad_outputs=probes.to(dtype=prediction.dtype),
        is_grads_batched=True,
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )
    norm_squared = sum(
        value.detach().to(torch.float32).reshape(len(probes), -1).square().sum(dim=1)
        for value in gradients
    )
    exact_parameter_norm = torch.sqrt(norm_squared).clamp_min(
        TRAJ_PROBE12_QUERY_NORM_EPS
    )
    projected = _project_batched_grads(
        dict(zip(names, gradients)),
        names,
        specs,
        d,
        False,
        TRAJ_PROBE12_QUERY_NORM_EPS,
    )
    return projected, exact_parameter_norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--timestamp-shard-index", type=int, required=True)
    parser.add_argument("--timestamp-shard-count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=TRAJ_PROBE12_BATCH_SIZE)
    args = parser.parse_args()
    if args.timestamp_shard_count <= 0:
        raise ValueError("--timestamp-shard-count must be positive")
    if not 0 <= args.timestamp_shard_index < args.timestamp_shard_count:
        raise ValueError("timestamp shard index is outside shard count")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    by_id = {int(record["query_id"]): record for record in manifest}
    records = [by_id[qid] for qid in TRAJ_PROBE12_QUERY_IDS]
    if any(record["family"] != TRAJ_PROBE12_FAMILY for record in records):
        raise ValueError("q00-q19 must all be prompted")

    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    paths = model_paths(TRAJ_PROBE12_FAMILY)[:TRAJ_PROBE12_CHECKPOINT_COUNT]
    if len(paths) != TRAJ_PROBE12_CHECKPOINT_COUNT:
        raise ValueError(
            f"expected {TRAJ_PROBE12_CHECKPOINT_COUNT} checkpoints, found {len(paths)}"
        )
    schedule = base.make_linear_schedule(T, device=device)
    x_all, cond_all = preload_dataset(dataset, TRAJ_PROBE12_FAMILY, device)
    trajectories = [
        np.load(Path(record["dir"]) / "trajectory_xt.npy") for record in records
    ]
    timestep_arrays = [
        np.load(Path(record["dir"]) / "trajectory_t.npy") for record in records
    ]
    t_seq = timestep_arrays[0]
    if len(t_seq) != TRAJ_SNAPSHOTS or any(
        not np.array_equal(t_seq, values) for values in timestep_arrays
    ):
        raise ValueError("all q00-q19 trajectories must share the same 100 timestamps")
    conditions = [cond_for(record, dataset, device) for record in records]
    selected_timestamps = list(
        range(
            args.timestamp_shard_index,
            len(t_seq),
            args.timestamp_shard_count,
        )
    )
    shard_root = traj_probe12_shard_root(
        args.timestamp_shard_index, args.timestamp_shard_count
    )
    shard_root.mkdir(parents=True, exist_ok=True)

    q_count = len(records)
    probe_count = TRAJ_PROBE12_NUM_PROBES
    d = int(TRACIN_PROJ_DIM)
    mc_count = int(TRACIN_TRAIN_MC)
    snap_weight = 1.0 / float(len(t_seq))
    started = time.perf_counter()
    print(
        f"[probe12 gpu={args.gpu}] q00-q19 timestamps="
        f"{len(selected_timestamps)}/{len(t_seq)} "
        f"checkpoints={len(paths)} probes={probe_count} train_mc={mc_count} "
        f"batch={args.batch_size} dim={d}",
        flush=True,
    )

    for shard_position, timestamp_index in enumerate(selected_timestamps, start=1):
        output_path = shard_root / f"timestamp_{timestamp_index:03d}.npy"
        if output_path.is_file():
            print(
                f"[probe12 gpu={args.gpu}] skip timestamp "
                f"{timestamp_index + 1}/{len(t_seq)}",
                flush=True,
            )
            continue
        tval = int(t_seq[timestamp_index])
        probes = output_probe_bank(timestamp_index, device)
        timestamp_acc_raw = torch.zeros(
            (q_count, probe_count, N_TRAIN), device=device, dtype=torch.float64
        )
        timestamp_acc_l2 = torch.zeros_like(timestamp_acc_raw)
        timestamp_term_raw = torch.zeros(
            (q_count, N_TRAIN), device=device, dtype=torch.float64
        )
        timestamp_term_l2 = torch.zeros_like(timestamp_term_raw)
        timestamp_started = time.perf_counter()

        for checkpoint_index, path in enumerate(paths):
            model, _, checkpoint = build_model(path, "raw", device)
            named = dict(model.named_parameters())
            names = tuple(named)
            params = tuple(named.values())
            params_dict = dict(named)
            specs = build_countsketch_specs(
                list(params),
                d,
                device=device,
                seed_parts=(
                    TRAIN_SEED,
                    "x3_traj_probe12_parameter_projection",
                    checkpoint_index,
                ),
            )

            t_q = torch.tensor([tval], device=device, dtype=torch.long)
            query_features = []
            query_norms = []
            for trajectory, condition in zip(trajectories, conditions):
                xt = torch.from_numpy(trajectory[timestamp_index]).to(
                    device=device, dtype=torch.float32
                )
                feature, exact_norm = projected_query_probe_bank(
                    model,
                    params,
                    names,
                    specs,
                    d,
                    xt,
                    t_q,
                    condition,
                    probes,
                )
                query_features.append(feature)
                query_norms.append(exact_norm)
            query_raw = torch.stack(query_features, dim=0)
            query_norm = torch.stack(query_norms, dim=0).unsqueeze(-1)
            query_l2 = query_raw / query_norm
            query_raw_matrix = query_raw.reshape(q_count * probe_count, d)
            query_l2_matrix = query_l2.reshape(q_count * probe_count, d)

            t_mc = torch.full(
                (mc_count,), tval, device=device, dtype=torch.long
            )

            def single_mean_loss(pdict, x0, condition, noises):
                x_mc = x0.unsqueeze(0).expand(mc_count, *x0.shape)
                c_mc = condition.unsqueeze(0).expand(mc_count, condition.shape[-1])
                xt = base.q_sample(x_mc, t_mc, noises, schedule)
                prediction = functional_call(model, pdict, (xt, t_mc, c_mc))
                return (prediction - noises).square().reshape(mc_count, -1).mean()

            batched_grad = vmap(
                grad(single_mean_loss), in_dims=(None, 0, 0, 0)
            )
            term_weight = float(tracin_lr_weight(checkpoint)) * snap_weight
            num_batches = math.ceil(N_TRAIN / args.batch_size)
            progress_every = max(1, num_batches // 5)
            for batch_index, start in enumerate(
                range(0, N_TRAIN, args.batch_size), start=1
            ):
                end = min(start + args.batch_size, N_TRAIN)
                xb, cb = x_all[start:end], cond_all[start:end]
                generator = make_torch_generator(
                    device,
                    TRAIN_SEED,
                    "projected_traj_train",
                    checkpoint_index,
                    timestamp_index,
                    start,
                    mc_count,
                )
                noises = torch.randn(
                    (end - start, mc_count, *xb.shape[1:]),
                    generator=generator,
                    device=device,
                    dtype=xb.dtype,
                )
                gradients = batched_grad(params_dict, xb, cb, noises)
                train_features = _project_batched_grads(
                    gradients, names, specs, d, False, 1e-8
                )
                raw = (train_features @ query_raw_matrix.T).T.reshape(
                    q_count, probe_count, end - start
                ).to(torch.float64)
                l2 = (train_features @ query_l2_matrix.T).T.reshape(
                    q_count, probe_count, end - start
                ).to(torch.float64)
                timestamp_term_raw[:, start:end] += (
                    term_weight * raw.square().mean(dim=1)
                )
                timestamp_term_l2[:, start:end] += (
                    term_weight * l2.square().mean(dim=1)
                )
                timestamp_acc_raw[:, :, start:end] += term_weight * raw
                timestamp_acc_l2[:, :, start:end] += term_weight * l2
                if (
                    batch_index == 1
                    or batch_index % progress_every == 0
                    or batch_index == num_batches
                ):
                    print(
                        f"[probe12 gpu={args.gpu}] timestamp="
                        f"{timestamp_index + 1}/100 checkpoint="
                        f"{checkpoint_index + 1}/{len(paths)} batch="
                        f"{batch_index}/{num_batches}",
                        flush=True,
                    )

            del (
                model,
                query_features,
                query_norms,
                query_raw,
                query_l2,
                query_raw_matrix,
                query_l2_matrix,
                train_features,
                gradients,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        values = torch.stack(
            (
                timestamp_term_raw,
                timestamp_term_l2,
                timestamp_acc_raw.square().mean(dim=1),
                timestamp_acc_l2.square().mean(dim=1),
            ),
            dim=0,
        ).to(torch.float32).cpu().numpy()
        atomic_save_npy(output_path, values)
        print(
            f"[probe12 gpu={args.gpu}] saved timestamp={timestamp_index + 1}/100 "
            f"shape={values.shape} elapsed="
            f"{(time.perf_counter()-timestamp_started)/60:.1f}m shard_elapsed="
            f"{(time.perf_counter()-started)/3600:.2f}h "
            f"position={shard_position}/{len(selected_timestamps)}",
            flush=True,
        )
        del (
            values,
            timestamp_acc_raw,
            timestamp_acc_l2,
            timestamp_term_raw,
            timestamp_term_l2,
        )

    atomic_save_json(
        shard_root / "done.json",
        {
            "timestamp_shard_index": args.timestamp_shard_index,
            "timestamp_shard_count": args.timestamp_shard_count,
            "timestamp_indices": selected_timestamps,
            "query_ids": list(TRAJ_PROBE12_QUERY_IDS),
            "variant_order": [list(item) for item in TRAJ_PROBE12_VARIANTS],
            "num_probes": TRAJ_PROBE12_NUM_PROBES,
            "num_checkpoints": len(paths),
            "train_mc": mc_count,
            "batch_size": args.batch_size,
            "proj_dim": d,
        },
    )
    print(f"[done] probe12 timestamp shard: {shard_root}", flush=True)


if __name__ == "__main__":
    main()
