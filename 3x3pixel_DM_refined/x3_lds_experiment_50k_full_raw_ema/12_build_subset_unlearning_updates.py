"""Aggregate checkpoint/timestamp loss-gradient updates for all 192 LDS masks."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call, grad, vmap

import x3pixel_DM_training as base
from attribution_one_query import build_model, model_paths, preload_dataset, tracin_lr_weight
from checkpoint_counterfactual_config import *
from dataset_loader import ColorGridDataset
from exp_config import *
from run_exact_traj_next_bank import flatten_batched_gradients
from x3_endpoint_das_jax_logic_pytorch import make_torch_generator


def atomic_torch_save(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--timestamp-shard-index", type=int, required=True)
    parser.add_argument("--timestamp-shard-count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=CF_UPDATE_GRAD_BATCH_SIZE)
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    membership = np.load(MASK_DIR / "membership.npy").astype(bool)
    if membership.shape != (192, N_TRAIN):
        raise ValueError(f"unexpected membership shape {membership.shape}")
    selected = ~membership if CF_UNLEARN_SET == "removed" else membership
    selector_all = torch.from_numpy(selected).to(device=device, dtype=torch.float32)
    counts = selector_all.sum(dim=1)

    with open(QUERY_DIR / "manifest.json") as handle:
        family_queries = [q for q in json.load(handle) if q["family"] == args.family]
    t_seq = np.load(Path(family_queries[0]["dir"]) / "trajectory_t.npy")
    selected_timestamps = list(
        range(args.timestamp_shard_index, len(t_seq), args.timestamp_shard_count)
    )
    out = (
        CF_UPDATE_ROOT / "shards" / args.family
        / f"shard_{args.timestamp_shard_index:02d}_of_{args.timestamp_shard_count:02d}.pt"
    )
    if out.is_file():
        print(f"[skip] {out}", flush=True)
        return
    partial_path = out.with_name(out.stem + "_partial.pt")

    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    x_all, cond_all = preload_dataset(dataset, args.family, device)
    schedule = base.make_linear_schedule(T, device=device)
    paths = model_paths(args.family)
    updates = None
    names = None
    shapes = None
    completed_timestamps = []
    if partial_path.is_file():
        partial = torch.load(partial_path, map_location="cpu", weights_only=False)
        expected_contract = {
            "family": args.family,
            "timestamp_indices": selected_timestamps,
            "unlearn_set": CF_UNLEARN_SET,
            "train_mc": CF_TRAIN_MC,
            "gradient_param_source": "raw",
        }
        for key, expected in expected_contract.items():
            if partial.get(key) != expected:
                raise ValueError(
                    f"partial update contract changed for {key}: "
                    f"saved={partial.get(key)!r}, current={expected!r}"
                )
        completed_timestamps = [int(value) for value in partial["completed_timestamps"]]
        updates = partial["update_sums"].to(device=device, dtype=torch.float32)
        names = tuple(partial["parameter_names"])
        shapes = partial["parameter_shapes"]
        print(
            f"[resume] {args.family} shard has {len(completed_timestamps)}/"
            f"{len(selected_timestamps)} timestamps complete",
            flush=True,
        )
    num_terms = len(selected_timestamps) * len(paths)
    completed = 0
    remaining_timestamps = [
        value for value in selected_timestamps if value not in set(completed_timestamps)
    ]
    started = time.perf_counter()
    for snapshot_index in remaining_timestamps:
        timestep = int(t_seq[snapshot_index])
        train_timestep = torch.full(
            (CF_TRAIN_MC,), timestep, device=device, dtype=torch.long
        )
        for checkpoint_index, path in enumerate(paths):
            model, _, checkpoint = build_model(path, "raw", device)
            named = dict(model.named_parameters())
            current_names = tuple(named)
            if names is None:
                names = current_names
                shapes = [tuple(named[name].shape) for name in names]
                parameter_count = sum(named[name].numel() for name in names)
                updates = torch.zeros(
                    (membership.shape[0], parameter_count),
                    device=device,
                    dtype=torch.float32,
                )
            elif current_names != names:
                raise ValueError("parameter order changed between checkpoints")
            params = dict(named)

            def one_loss(pdict, x0, condition, noises):
                x_mc = x0.unsqueeze(0).expand(CF_TRAIN_MC, *x0.shape)
                c_mc = condition.unsqueeze(0).expand(CF_TRAIN_MC, condition.shape[-1])
                xt = base.q_sample(x_mc, train_timestep, noises, schedule)
                prediction = functional_call(model, pdict, (xt, train_timestep, c_mc))
                return (prediction - noises).pow(2).reshape(CF_TRAIN_MC, -1).mean()

            batch_grad = vmap(grad(one_loss), in_dims=(None, 0, 0, 0))
            weight = -float(tracin_lr_weight(checkpoint)) / float(len(t_seq))
            for start in range(0, N_TRAIN, args.batch_size):
                end = min(start + args.batch_size, N_TRAIN)
                xb, cb = x_all[start:end], cond_all[start:end]
                generator = make_torch_generator(
                    device, TRAIN_SEED, "cf_subset_update", args.family,
                    checkpoint_index, snapshot_index, start, CF_TRAIN_MC,
                )
                noises = torch.randn(
                    (end - start, CF_TRAIN_MC, *xb.shape[1:]),
                    generator=generator, device=device, dtype=xb.dtype,
                )
                gradients = batch_grad(params, xb, cb, noises)
                matrix = flatten_batched_gradients(gradients, names).detach()
                selector = selector_all[:, start:end]
                updates.addmm_(selector, matrix, beta=1.0, alpha=weight)
            completed += 1
            print(
                f"[subset-update {args.family}] term={completed}/{num_terms} "
                f"checkpoint={checkpoint_index+1}/{len(paths)} "
                f"timestamp={snapshot_index+1}/{len(t_seq)}",
                flush=True,
            )
            del model, checkpoint, named, params, batch_grad, gradients, matrix
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        completed_timestamps.append(snapshot_index)
        completed_timestamps.sort()
        atomic_torch_save(
            {
                "update_sums": updates.cpu(),
                "parameter_names": names,
                "parameter_shapes": shapes,
                "completed_timestamps": completed_timestamps,
                "family": args.family,
                "timestamp_indices": selected_timestamps,
                "unlearn_set": CF_UNLEARN_SET,
                "train_mc": CF_TRAIN_MC,
                "gradient_param_source": "raw",
            },
            partial_path,
        )
        elapsed = time.perf_counter() - started
        print(
            f"[checkpoint] {args.family} shard timestamps="
            f"{len(completed_timestamps)}/{len(selected_timestamps)} "
            f"elapsed={elapsed/3600:.2f}h",
            flush=True,
        )

    if sorted(completed_timestamps) != sorted(selected_timestamps):
        raise RuntimeError("not all assigned timestamps completed")
    updates /= counts.unsqueeze(1)
    atomic_torch_save(
        {
            "updates": updates.cpu(),
            "parameter_names": names,
            "parameter_shapes": shapes,
            "family": args.family,
            "timestamp_indices": selected_timestamps,
            "timestamp_shard_count": args.timestamp_shard_count,
            "definition": "mean_unlearn_set[-lr*mean_t(train_loss_gradient)] summed_checkpoints",
            "unlearn_set": CF_UNLEARN_SET,
            "gradient_param_source": "raw",
            "checkpoints": len(paths),
            "train_mc": CF_TRAIN_MC,
        },
        out,
    )
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
