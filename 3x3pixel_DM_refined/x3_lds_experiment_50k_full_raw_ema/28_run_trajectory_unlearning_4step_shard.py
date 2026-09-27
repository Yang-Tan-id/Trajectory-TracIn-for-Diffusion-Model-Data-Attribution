"""Four-step trajectory-gradient unlearning for one q00-q49 GPU shard."""

import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call

import x3pixel_DM_training as base
from dataset_loader import ColorGridDataset
from forward_loss_alignment_config import *
from forward_loss_alignment_metrics import make_loss_condition_bins, score_contributions
from train_worker import lr_at


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


def load_query_data(qid, ds, device):
    query_dir = FLA_QUERY_DIR / f"q{qid:02d}"
    with open(QUERY_DIR / f"q{qid:02d}" / "query.json") as handle:
        record = json.load(handle)
    cond = torch.zeros((1, len(ds.vocab)), dtype=torch.float32, device=device)
    for label in record["labels"]:
        cond[0, ds.vocab[label]] = 1.0
    return {
        "xt": np.load(query_dir / "trajectory_xt_1000.npy", mmap_mode="r"),
        "target": np.load(query_dir / "target_eps_ema_1000.npy", mmap_mode="r"),
        "timestamps": np.load(query_dir / "trajectory_t_1000.npy", mmap_mode="r"),
        "cond": cond,
    }


def query_gradient(model, query_data, device):
    model.zero_grad(set_to_none=True)
    total_elements = float(DDIM_STEPS * 3 * 3 * 3)
    loss_value = 0.0
    for start in range(0, DDIM_STEPS, FLA_QUERY_BATCH_SIZE):
        end = min(start + FLA_QUERY_BATCH_SIZE, DDIM_STEPS)
        xb = torch.from_numpy(np.asarray(query_data["xt"][start:end])).to(device)
        tb = torch.from_numpy(
            np.asarray(query_data["timestamps"][start:end], dtype=np.int64)
        ).to(device)
        yb = torch.from_numpy(np.asarray(query_data["target"][start:end])).to(device)
        cb = query_data["cond"].expand(end - start, -1)
        loss = F.mse_loss(model(xb, tb, cb), yb, reduction="sum") / total_elements
        loss.backward()
        loss_value += float(loss.detach())
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    norm_sq = sum(torch.sum(gradient.double() ** 2) for gradient in gradients.values())
    return gradients, float(torch.sqrt(norm_sq).item()), loss_value


@torch.no_grad()
def ascent_step(model, gradients, learning_rate, normalize):
    scale = float(learning_rate)
    if normalize:
        norm_sq = sum(torch.sum(gradient.double() ** 2) for gradient in gradients.values())
        scale /= max(float(torch.sqrt(norm_sq).item()), FLA_NORMALIZE_EPS)
    for name, parameter in model.named_parameters():
        parameter.add_(gradients[name], alpha=scale)


def epoch_end_learning_rates(checkpoint_epoch):
    steps_per_epoch = int(math.ceil(N_TRAIN / BATCH_SIZE))
    total_steps = EPOCHS * steps_per_epoch
    # Undo the most recent epoch direction first: c, c-1, c-2, c-3.
    epochs = tuple(
        checkpoint_epoch - offset
        for offset in range(FLA_UNLEARN_STEPS)
    )
    rates = tuple(
        lr_at(epoch * steps_per_epoch - 1, total_steps)
        for epoch in epochs
    )
    return epochs, rates


def build_unlearned_variants(checkpoint, checkpoint_epoch, ds, query_data, device):
    initial = make_model(ds, checkpoint["model_state"], device)
    epochs, rates = epoch_end_learning_rates(checkpoint_epoch)

    # Step one has the same gradient for both paths, so compute it once.
    gradients, gradient_norm, query_loss = query_gradient(initial, query_data, device)
    raw_model = copy.deepcopy(initial).to(device).eval()
    normalized_model = copy.deepcopy(initial).to(device).eval()
    raw_model.zero_grad(set_to_none=True)
    normalized_model.zero_grad(set_to_none=True)
    ascent_step(raw_model, gradients, rates[0], normalize=False)
    ascent_step(normalized_model, gradients, rates[0], normalize=True)
    step_stats = [
        {
            "step": 1,
            "epoch": epochs[0],
            "lr": rates[0],
            "raw_query_loss": query_loss,
            "normalized_query_loss": query_loss,
            "raw_grad_norm": gradient_norm,
            "normalized_grad_norm": gradient_norm,
        }
    ]
    del initial, gradients

    for step_index in range(1, FLA_UNLEARN_STEPS):
        raw_gradients, raw_norm, raw_loss = query_gradient(raw_model, query_data, device)
        normalized_gradients, normalized_norm, normalized_loss = query_gradient(
            normalized_model, query_data, device
        )
        ascent_step(raw_model, raw_gradients, rates[step_index], normalize=False)
        ascent_step(
            normalized_model,
            normalized_gradients,
            rates[step_index],
            normalize=True,
        )
        step_stats.append(
            {
                "step": step_index + 1,
                "epoch": epochs[step_index],
                "lr": rates[step_index],
                "raw_query_loss": raw_loss,
                "normalized_query_loss": normalized_loss,
                "raw_grad_norm": raw_norm,
                "normalized_grad_norm": normalized_norm,
            }
        )
        del raw_gradients, normalized_gradients
    return raw_model, normalized_model, step_stats


@torch.no_grad()
def changed_event_losses(
    evaluation_model,
    parameter_variants,
    checkpoint_index,
    images,
    conds,
    t_cache,
    noise_cache,
    schedule,
    device,
):
    losses = {
        name: np.empty((N_TRAIN, FLA_EVENTS_PER_CHECKPOINT), dtype=np.float64)
        for name in parameter_variants
    }
    for start in range(0, N_TRAIN, FLA_DATAPOINT_BATCH_SIZE):
        end = min(start + FLA_DATAPOINT_BATCH_SIZE, N_TRAIN)
        count = end - start
        x0 = torch.from_numpy(images[start:end]).to(device)
        cond = torch.from_numpy(conds[start:end]).to(device)
        t = torch.from_numpy(
            np.asarray(t_cache[checkpoint_index, start:end], dtype=np.int64)
        ).to(device)
        noise = torch.from_numpy(
            np.asarray(noise_cache[checkpoint_index, start:end])
        ).to(device)
        x0 = x0[:, None].expand(
            -1, FLA_EVENTS_PER_CHECKPOINT, -1, -1, -1
        ).reshape(-1, 3, 3, 3)
        cond = cond[:, None].expand(
            -1, FLA_EVENTS_PER_CHECKPOINT, -1
        ).reshape(-1, cond.shape[-1])
        t = t.reshape(-1)
        noise = noise.reshape(-1, 3, 3, 3)
        xt = base.q_sample(x0, t, noise, schedule)
        for name, params in parameter_variants.items():
            prediction = functional_call(evaluation_model, params, (xt, t, cond))
            event_loss = (
                prediction.double() - noise.double()
            ).square().flatten(1).mean(1)
            losses[name][start:end] = event_loss.reshape(
                count, FLA_EVENTS_PER_CHECKPOINT
            ).cpu().numpy()
    return losses


def score_query(
    qid, gpu, images, conds, t_cache, noise_cache,
    baseline_events, condition_bins_by_checkpoint, schedule, ds,
):
    device = torch.device(f"cuda:{gpu}")
    query_data = load_query_data(qid, ds, device)
    partial_dir = FLA_PARTIAL_DIR / "trajectory_unlearning_4step" / f"q{qid:02d}"
    partial_dir.mkdir(parents=True, exist_ok=True)
    partial_path = partial_dir / "state_v1.npz"
    score_sums = {
        method: np.zeros(N_TRAIN, dtype=np.float64)
        for method in FLA_UNLEARN_METHODS
    }
    next_checkpoint = 0
    if partial_path.is_file():
        partial = np.load(partial_path)
        for method_index, method in enumerate(FLA_UNLEARN_METHODS):
            score_sums[method][:] = partial[f"score_{method_index}"]
        next_checkpoint = int(partial["next_checkpoint"])
        print(f"[gpu {gpu} q{qid:02d}] resume checkpoint {next_checkpoint + 1}", flush=True)

    started = time.perf_counter()
    for checkpoint_index in range(next_checkpoint, len(FLA_CHECKPOINT_EPOCHS)):
        epoch = FLA_CHECKPOINT_EPOCHS[checkpoint_index]
        checkpoint = load_checkpoint(checkpoint_path(epoch))
        raw_model, normalized_model, step_stats = build_unlearned_variants(
            checkpoint, epoch, ds, query_data, device
        )
        parameter_variants = {
            "raw": {
                name: parameter.detach()
                for name, parameter in raw_model.named_parameters()
            },
            "normalized": {
                name: parameter.detach()
                for name, parameter in normalized_model.named_parameters()
            },
        }
        after_losses = changed_event_losses(
            raw_model, parameter_variants, checkpoint_index, images, conds,
            t_cache, noise_cache, schedule, device,
        )
        baseline_checkpoint = np.asarray(
            baseline_events[checkpoint_index], dtype=np.float64
        )
        checkpoint_stats = []
        for update_name, changed_events in after_losses.items():
            contributions = score_contributions(
                baseline_checkpoint,
                changed_events,
                condition_bins_by_checkpoint[checkpoint_index],
                direction="increase",
            )
            for normalization, values in contributions.items():
                method = FLA_UNLEARN_METHOD_BY_VARIANT[(update_name, normalization)]
                score_sums[method] += values
            checkpoint_stats.append(
                f"{update_name}:growth={np.median(contributions['absolute']):.2e},"
                f"log={np.median(contributions['log_relative']):.2e}"
            )
        temporary_path = partial_dir / "state_v1.tmp.npz"
        np.savez(
            temporary_path,
            next_checkpoint=np.asarray(checkpoint_index + 1, dtype=np.int64),
            **{
                f"score_{method_index}": score_sums[method]
                for method_index, method in enumerate(FLA_UNLEARN_METHODS)
            },
        )
        os.replace(temporary_path, partial_path)
        elapsed = time.perf_counter() - started
        completed = checkpoint_index - next_checkpoint + 1
        remaining = len(FLA_CHECKPOINT_EPOCHS) - checkpoint_index - 1
        eta = elapsed / completed * remaining
        lr_text = ",".join(f"{item['lr']:.2e}" for item in step_stats)
        print(
            f"[gpu {gpu} q{qid:02d}] checkpoint={checkpoint_index + 1:02d}/"
            f"{len(FLA_CHECKPOINT_EPOCHS)} epoch={epoch:03d} reverse_lrs=[{lr_text}] "
            f"{' '.join(checkpoint_stats)} elapsed={elapsed / 60:.1f}m "
            f"eta={eta / 60:.1f}m",
            flush=True,
        )
        del raw_model, normalized_model, parameter_variants, after_losses
        torch.cuda.empty_cache()

    metadata = {
        "query_id": qid,
        "family": FLA_FAMILY,
        "checkpoint_epochs": list(FLA_CHECKPOINT_EPOCHS),
        "unlearning_steps_per_checkpoint": FLA_UNLEARN_STEPS,
        "learning_rate_definition": (
            "last real optimizer-step LR of each epoch, reverse chronological order c,c-1,c-2,c-3"
        ),
        "gradient_definition": (
            "recomputed mean 1000-timestamp reference-imitation gradient after every ascent step"
        ),
        "reference_target": "fixed final-EMA predicted noise on fixed cached EMA DDIM trajectory",
        "checkpoint_aggregation": "mean over 50 checkpoints",
        "score_direction": "loss_after_unlearning - loss_before",
        "normalizations": {
            "absolute": "mean_event(loss_after - loss_before)",
            "log_relative": "mean_event(log((loss_after + eps) / (loss_before + eps)))",
            "loss_conditioned_robust": (
                "log growth robust-z within 20 equal-count baseline-loss bins; clip [-5,5]"
            ),
        },
    }
    reverse_variants = {
        method: {"update": update, "normalization": normalization}
        for (update, normalization), method in FLA_UNLEARN_METHOD_BY_VARIANT.items()
    }
    for method in FLA_UNLEARN_METHODS:
        scores = score_sums[method] / float(len(FLA_CHECKPOINT_EPOCHS))
        output_dir = ATTR_DIR / method / f"q{qid:02d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "scores.npy", scores)
        with open(output_dir / "metadata.json", "w") as handle:
            json.dump(
                {**metadata, "method": method, **reverse_variants[method]},
                handle,
                indent=2,
            )
    print(f"[gpu {gpu} q{qid:02d}] DONE", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    qids = list(FLA_QUERY_IDS)[args.shard_index :: args.shard_count]
    ds = ColorGridDataset(str(BASE_CSV), grid_size=3)
    images, conds = dataset_arrays(ds)
    t_cache = np.load(replay_t_path(), mmap_mode="r")
    noise_cache = np.load(replay_noise_path(), mmap_mode="r")
    baseline_events = np.load(baseline_event_path(), mmap_mode="r")
    expected_shape = (
        len(FLA_CHECKPOINT_EPOCHS), N_TRAIN, FLA_EVENTS_PER_CHECKPOINT
    )
    if baseline_events.shape != expected_shape:
        raise ValueError(
            f"event baseline shape={baseline_events.shape}, expected={expected_shape}; "
            "rerun 24_prepare_forward_loss_alignment.py"
        )
    condition_bins_by_checkpoint = tuple(
        make_loss_condition_bins(np.asarray(baseline_events[index]))
        for index in range(len(FLA_CHECKPOINT_EPOCHS))
    )
    schedule = base.make_linear_schedule(T, device=device)
    print(f"[gpu {args.gpu}] qids={qids}", flush=True)
    for qid in qids:
        score_query(
            qid, args.gpu, images, conds, t_cache, noise_cache,
            baseline_events, condition_bins_by_checkpoint, schedule, ds,
        )


if __name__ == "__main__":
    main()
