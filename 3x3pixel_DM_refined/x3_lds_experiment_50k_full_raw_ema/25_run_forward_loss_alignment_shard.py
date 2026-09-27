"""Score a query shard with one-step forward-loss alignment."""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call

import x3pixel_DM_training as base
from dataset_loader import ColorGridDataset
from forward_loss_alignment_config import *


def load_checkpoint(path):
    return torch.load(path, map_location="cpu")


def make_model(ds, state, device):
    x, cond = ds[0]
    model = base.CondEpsModel(
        in_ch=int(x.shape[0]), cond_dim=int(cond.numel()),
        base_ch=BASE_CH, time_dim=TIME_DIM,
    )
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def dataset_arrays(ds):
    images = np.empty((len(ds), 3, 3, 3), dtype=np.float32)
    conds = np.empty((len(ds), len(ds.vocab)), dtype=np.float32)
    for index in range(len(ds)):
        image, cond = ds[index]
        images[index] = image.numpy()
        conds[index] = cond.numpy()
    return images, conds


def query_gradient(model, qid, device, ds):
    query_dir = FLA_QUERY_DIR / f"q{qid:02d}"
    xt = np.load(query_dir / "trajectory_xt_1000.npy", mmap_mode="r")
    target = np.load(query_dir / "target_eps_ema_1000.npy", mmap_mode="r")
    timestamps = np.load(query_dir / "trajectory_t_1000.npy", mmap_mode="r")
    with open(QUERY_DIR / f"q{qid:02d}" / "query.json") as handle:
        record = json.load(handle)
    cond = torch.zeros((1, len(ds.vocab)), dtype=torch.float32, device=device)
    for label in record["labels"]:
        cond[0, ds.vocab[label]] = 1.0

    model.zero_grad(set_to_none=True)
    total_elements = float(DDIM_STEPS * 3 * 3 * 3)
    loss_value = 0.0
    for start in range(0, DDIM_STEPS, FLA_QUERY_BATCH_SIZE):
        end = min(start + FLA_QUERY_BATCH_SIZE, DDIM_STEPS)
        xb = torch.from_numpy(np.asarray(xt[start:end])).to(device)
        tb = torch.from_numpy(np.asarray(timestamps[start:end], dtype=np.int64)).to(device)
        yb = torch.from_numpy(np.asarray(target[start:end])).to(device)
        cb = cond.expand(end - start, -1)
        loss = F.mse_loss(model(xb, tb, cb), yb, reduction="sum") / total_elements
        loss.backward()
        loss_value += float(loss.detach())
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    squared_norm = sum(torch.sum(gradient.double() ** 2) for gradient in gradients.values())
    norm = float(torch.sqrt(squared_norm).item())
    return gradients, norm, loss_value


def updated_parameters(model, gradients, learning_rate, normalize):
    scale = float(learning_rate)
    if normalize:
        norm_sq = sum(torch.sum(gradient.double() ** 2) for gradient in gradients.values())
        scale /= max(float(torch.sqrt(norm_sq).item()), FLA_NORMALIZE_EPS)
    return {
        name: parameter.detach() - scale * gradients[name]
        for name, parameter in model.named_parameters()
    }


@torch.no_grad()
def checkpoint_loss_reductions(
    model,
    parameter_variants,
    checkpoint_index,
    images,
    conds,
    t_cache,
    noise_cache,
    baseline,
    schedule,
    device,
):
    reductions = {
        name: np.empty(N_TRAIN, dtype=np.float64)
        for name in parameter_variants
    }
    for start in range(0, N_TRAIN, FLA_DATAPOINT_BATCH_SIZE):
        end = min(start + FLA_DATAPOINT_BATCH_SIZE, N_TRAIN)
        count = end - start
        x0 = torch.from_numpy(images[start:end]).to(device)
        cond = torch.from_numpy(conds[start:end]).to(device)
        t = torch.from_numpy(np.asarray(t_cache[checkpoint_index, start:end], dtype=np.int64)).to(device)
        noise = torch.from_numpy(np.asarray(noise_cache[checkpoint_index, start:end])).to(device)
        x0 = x0[:, None].expand(-1, FLA_EVENTS_PER_CHECKPOINT, -1, -1, -1).reshape(-1, 3, 3, 3)
        cond = cond[:, None].expand(-1, FLA_EVENTS_PER_CHECKPOINT, -1).reshape(-1, cond.shape[-1])
        t = t.reshape(-1)
        noise = noise.reshape(-1, 3, 3, 3)
        xt = base.q_sample(x0, t, noise, schedule)
        for name, params in parameter_variants.items():
            prediction = functional_call(model, params, (xt, t, cond))
            event_loss = (prediction.double() - noise.double()).square().flatten(1).mean(1)
            updated_loss = event_loss.reshape(count, FLA_EVENTS_PER_CHECKPOINT).mean(1).cpu().numpy()
            reductions[name][start:end] = baseline[checkpoint_index, start:end] - updated_loss
    return reductions


def score_query(qid, gpu, images, conds, t_cache, noise_cache, baseline, schedule, ds):
    device = torch.device(f"cuda:{gpu}")
    partial_dir = FLA_PARTIAL_DIR / f"q{qid:02d}"
    partial_dir.mkdir(parents=True, exist_ok=True)
    partial_path = partial_dir / "state.npz"
    raw_scores = np.zeros(N_TRAIN, dtype=np.float64)
    normalized_scores = np.zeros(N_TRAIN, dtype=np.float64)
    next_checkpoint = 0
    if partial_path.is_file():
        partial = np.load(partial_path)
        raw_scores[:] = partial["raw_scores"]
        normalized_scores[:] = partial["normalized_scores"]
        next_checkpoint = int(partial["next_checkpoint"])
        print(f"[gpu {gpu} q{qid:02d}] resume checkpoint {next_checkpoint + 1}", flush=True)

    started = time.perf_counter()
    for checkpoint_index in range(next_checkpoint, len(FLA_CHECKPOINT_EPOCHS)):
        epoch = FLA_CHECKPOINT_EPOCHS[checkpoint_index]
        checkpoint = load_checkpoint(checkpoint_path(epoch))
        model = make_model(ds, checkpoint["model_state"], device)
        gradients, gradient_norm, query_loss = query_gradient(model, qid, device, ds)
        learning_rate = float(checkpoint.get("learning_rate_at_checkpoint", checkpoint["eta"]))
        variants = {
            "raw": updated_parameters(model, gradients, learning_rate, normalize=False),
            "normalized": updated_parameters(model, gradients, learning_rate, normalize=True),
        }
        reductions = checkpoint_loss_reductions(
            model, variants, checkpoint_index, images, conds,
            t_cache, noise_cache, baseline, schedule, device,
        )
        raw_scores += reductions["raw"]
        normalized_scores += reductions["normalized"]
        temporary_partial = partial_dir / "state.tmp.npz"
        np.savez(
            temporary_partial,
            next_checkpoint=np.asarray(checkpoint_index + 1, dtype=np.int64),
            raw_scores=raw_scores,
            normalized_scores=normalized_scores,
        )
        os.replace(temporary_partial, partial_path)
        elapsed = time.perf_counter() - started
        completed = checkpoint_index - next_checkpoint + 1
        remaining = len(FLA_CHECKPOINT_EPOCHS) - checkpoint_index - 1
        eta = elapsed / completed * remaining
        print(
            f"[gpu {gpu} q{qid:02d}] checkpoint={checkpoint_index + 1:02d}/{len(FLA_CHECKPOINT_EPOCHS)} "
            f"epoch={epoch:03d} lr={learning_rate:.3e} query_loss={query_loss:.6e} "
            f"|g|={gradient_norm:.6e} elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m",
            flush=True,
        )
        del model, gradients, variants, reductions
        torch.cuda.empty_cache()

    metadata = {
        "query_id": qid,
        "family": FLA_FAMILY,
        "checkpoint_epochs": list(FLA_CHECKPOINT_EPOCHS),
        "events_per_checkpoint": FLA_EVENTS_PER_CHECKPOINT,
        "reference_timestamps": DDIM_STEPS,
        "score_definition": "sum_c mean_event(loss_before - loss_after_one_query_SGD_step)",
        "raw_update": "theta_plus = theta - checkpoint_lr * grad(reference_imitation_loss)",
        "normalized_update": "theta_plus = theta - checkpoint_lr * grad / global_grad_norm",
        "reference_target": "final EMA predicted noise on cached EMA DDIM trajectory",
        "updated_parameter_source": "raw checkpoint",
    }
    for method, scores in (
        (FLA_METHOD_RAW_STEP, raw_scores),
        (FLA_METHOD_NORMALIZED_STEP, normalized_scores),
    ):
        output_dir = ATTR_DIR / method / f"q{qid:02d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "scores.npy", scores)
        with open(output_dir / "metadata.json", "w") as handle:
            json.dump({**metadata, "method": method}, handle, indent=2)
    print(f"[gpu {gpu} q{qid:02d}] DONE", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    qids = list(FLA_QUERY_IDS)[args.shard_index :: args.shard_count]
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    ds = ColorGridDataset(str(BASE_CSV), grid_size=3)
    images, conds = dataset_arrays(ds)
    t_cache = np.load(replay_t_path(), mmap_mode="r")
    noise_cache = np.load(replay_noise_path(), mmap_mode="r")
    baseline = np.load(baseline_path(), mmap_mode="r")
    schedule = base.make_linear_schedule(T, device=device)
    print(f"[gpu {args.gpu}] shard={args.shard_index}/{args.shard_count} qids={qids}", flush=True)
    for qid in qids:
        score_query(qid, args.gpu, images, conds, t_cache, noise_cache, baseline, schedule, ds)


if __name__ == "__main__":
    main()
