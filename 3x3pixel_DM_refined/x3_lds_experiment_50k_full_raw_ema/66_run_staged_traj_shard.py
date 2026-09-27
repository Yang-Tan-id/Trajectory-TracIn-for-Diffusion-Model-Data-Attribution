"""One timestamp shard of stage-aligned projected next Traj-TracIn."""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call, grad, vmap

import x3pixel_DM_training as base
from attribution_one_query import _project_batched_grads, _project_gradient_tuple, build_model, cond_for, preload_dataset, tracin_lr_weight
from dataset_loader import ColorGridDataset
from staged_lds_config import *
from x3_endpoint_das_jax_logic_pytorch import build_countsketch_specs, make_torch_generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--timestamp-shard-index", type=int, required=True)
    parser.add_argument("--timestamp-shard-count", type=int, default=4)
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    with open(STAGED_QUERY_DIR / "manifest.json") as handle:
        records = json.load(handle)
    trajectories = [np.load(Path(r["dir"]) / "trajectory_xt.npy") for r in records]
    timestep_arrays = [np.load(Path(r["dir"]) / "trajectory_t.npy") for r in records]
    t_seq = timestep_arrays[0]
    if any(not np.array_equal(t_seq, ts) for ts in timestep_arrays):
        raise ValueError("query timestamps differ")
    timestamp_indices = list(range(args.timestamp_shard_index, len(t_seq), args.timestamp_shard_count))
    shard_dir = STAGED_ATTR_DIR / "_traj_shards" / f"shard_{args.timestamp_shard_index:02d}_of_{args.timestamp_shard_count:02d}"
    if (shard_dir / "done.json").is_file():
        print(f"[skip] {shard_dir}", flush=True)
        return
    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    x_all, cond_all = preload_dataset(dataset, STAGED_FAMILY, device)
    stages = np.load(STAGED_PARTITION_DIR / "stages.npy")
    conditions = [cond_for(r, dataset, device) for r in records]
    sched = base.make_linear_schedule(T, device=device)
    score = torch.zeros((len(records), N_TRAIN), device=device, dtype=torch.float64)
    checkpoints = list(range(STAGED_SAVE_EVERY, STAGED_EPOCHS + 1, STAGED_SAVE_EVERY))
    pairs = list(zip(checkpoints[:-1], checkpoints[1:]))
    d = int(TRACIN_PROJ_DIM)
    batch_size = int(TRACIN_PROJECTED_BATCH_SIZE)
    mc_count = int(TRACIN_TRAIN_MC)
    started = time.perf_counter()
    for local_timestamp, si in enumerate(timestamp_indices, start=1):
        tval = int(t_seq[si])
        timestamp_acc = torch.zeros_like(score)
        for pair_index, (source_epoch, target_epoch) in enumerate(pairs):
            stage_id = stage_for_interval(source_epoch, target_epoch)
            active_indices_np = np.asarray(stages[stage_id], dtype=np.int64)
            active_indices = torch.from_numpy(active_indices_np).to(device=device)
            model, _, checkpoint = build_model(staged_base_checkpoint(source_epoch), "raw", device)
            target, _, _ = build_model(staged_base_checkpoint(target_epoch), "raw", device)
            named = dict(model.named_parameters())
            names = tuple(named)
            params = tuple(named.values())
            specs = build_countsketch_specs(
                list(params), d, device=device,
                seed_parts=(TRAIN_SEED, "staged_traj_projection", pair_index),
            )
            t_query = torch.tensor([tval], device=device)
            query_features = []
            for trajectory, condition in zip(trajectories, conditions):
                xt = torch.from_numpy(trajectory[si]).to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    target_eps = target(xt, t_query, condition).detach()
                query_loss = (model(xt, t_query, condition) - target_eps).pow(2).sum()
                query_gradient = torch.autograd.grad(query_loss, params)
                query_features.append(_project_gradient_tuple(query_gradient, specs, d))
            query_matrix = torch.stack(query_features)
            t_mc = torch.full((mc_count,), tval, device=device, dtype=torch.long)

            def mean_loss(parameter_dict, x0, condition, noises):
                x_mc = x0.unsqueeze(0).expand(mc_count, *x0.shape)
                c_mc = condition.unsqueeze(0).expand(mc_count, condition.shape[-1])
                xt = base.q_sample(x_mc, t_mc, noises, sched)
                prediction = functional_call(model, parameter_dict, (xt, t_mc, c_mc))
                return (prediction - noises).pow(2).reshape(mc_count, -1).mean(dim=1).mean()

            batched_gradient = vmap(grad(mean_loss), in_dims=(None, 0, 0, 0))
            weight = float(tracin_lr_weight(checkpoint)) / len(t_seq)
            num_batches = math.ceil(STAGE_SIZE / batch_size)
            for batch_number, start in enumerate(range(0, STAGE_SIZE, batch_size), start=1):
                end = min(start + batch_size, STAGE_SIZE)
                global_indices = active_indices[start:end]
                xb, cb = x_all[global_indices], cond_all[global_indices]
                generator = make_torch_generator(
                    device, TRAIN_SEED, "staged_projected_traj_train",
                    pair_index, si, start, mc_count,
                )
                noises = torch.randn(
                    (end - start, mc_count, *xb.shape[1:]), generator=generator,
                    device=device, dtype=xb.dtype,
                )
                gradients = batched_gradient(named, xb, cb, noises)
                phi = _project_batched_grads(gradients, names, specs, d, False, 1e-8)
                dots = (phi @ query_matrix.T).T.to(torch.float64)
                timestamp_acc[:, global_indices] += weight * dots
                if batch_number == 1 or batch_number == num_batches:
                    print(
                        f"[traj gpu={args.gpu}] timestamp={local_timestamp}/{len(timestamp_indices)} "
                        f"pair={pair_index+1}/{len(pairs)} stage={stage_id+1} batch={batch_number}/{num_batches}",
                        flush=True,
                    )
            del model, target, query_features, query_matrix, gradients, phi
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        score += timestamp_acc.square()
        print(
            f"[traj gpu={args.gpu}] timestamp={local_timestamp}/{len(timestamp_indices)} "
            f"elapsed={(time.perf_counter()-started)/3600:.2f}h",
            flush=True,
        )
    shard_dir.mkdir(parents=True, exist_ok=True)
    np.save(shard_dir / "scores.npy", score.cpu().numpy())
    with open(shard_dir / "done.json", "w") as handle:
        json.dump({"timestamp_indices": timestamp_indices, "query_ids": list(STAGED_QUERY_IDS)}, handle, indent=2)


if __name__ == "__main__":
    main()
