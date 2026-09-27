"""Run four checkpoint-AdamW reference updates and paired MC100 scoring."""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import x3pixel_DM_training as base
from checkpoint_adamw_reference_learning_config import *
from dataset_loader import ColorGridDataset


def load_checkpoint(epoch):
    return torch.load(
        carl_checkpoint_path(epoch), map_location="cpu", weights_only=False
    )


def make_model(dataset, state, device):
    model = base.CondEpsModel(
        in_ch=3,
        cond_dim=len(dataset.vocab),
        base_ch=BASE_CH,
        time_dim=TIME_DIM,
    ).to(device)
    model.load_state_dict(state, strict=True)
    return model.eval()


def dataset_arrays(dataset):
    images = np.empty((len(dataset), 3, 3, 3), dtype=np.float32)
    conditions = np.empty((len(dataset), len(dataset.vocab)), dtype=np.float32)
    for index in range(len(dataset)):
        image, condition = dataset[index]
        images[index] = image.numpy()
        conditions[index] = condition.numpy()
    return images, conditions


def query_condition(query_id, dataset, device):
    with open(QUERY_DIR / f"q{query_id:02d}" / "query.json") as handle:
        query = json.load(handle)
    condition = torch.zeros(
        (1, len(dataset.vocab)), dtype=torch.float32, device=device
    )
    for label in query["labels"]:
        condition[0, dataset.vocab[label]] = 1.0
    return condition, query


def four_reference_adamw_steps(
    model, optimizer, query_id, condition, checkpoint_lr, clip_norm, device
):
    query_dir = FLA_QUERY_DIR / f"q{query_id:02d}"
    trajectory = np.load(query_dir / "trajectory_xt_1000.npy", mmap_mode="r")
    targets = np.load(query_dir / "target_eps_ema_1000.npy", mmap_mode="r")
    timestamps = np.load(query_dir / "trajectory_t_1000.npy", mmap_mode="r")
    step_stats = []
    for step in range(CARL_REFERENCE_STEPS):
        start = step * CARL_REFERENCE_BATCH_SIZE
        end = start + CARL_REFERENCE_BATCH_SIZE
        x = torch.from_numpy(np.asarray(trajectory[start:end])).to(device)
        target = torch.from_numpy(np.asarray(targets[start:end])).to(device)
        t = torch.from_numpy(
            np.asarray(timestamps[start:end], dtype=np.int64)
        ).to(device)
        c = condition.expand(end - start, -1)
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x, t, c), target)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), clip_norm
        )
        optimizer.step()
        step_stats.append(
            {
                "step": step,
                "trajectory_start": start,
                "trajectory_end": end,
                "loss": float(loss.detach().item()),
                "gradient_norm_before_clip": float(gradient_norm),
                "learning_rate": checkpoint_lr,
            }
        )
    return step_stats


@torch.no_grad()
def paired_training_scores(
    baseline_model,
    updated_model,
    checkpoint_index,
    images,
    conditions,
    schedule,
    device,
):
    raw = np.empty(N_TRAIN, dtype=np.float64)
    ratio = np.empty(N_TRAIN, dtype=np.float64)
    generator = torch.Generator(device=device)
    generator.manual_seed(CARL_MC_SEED_BASE + int(checkpoint_index))
    started = time.perf_counter()
    for start in range(0, N_TRAIN, CARL_SCORE_DATAPOINT_BATCH):
        end = min(start + CARL_SCORE_DATAPOINT_BATCH, N_TRAIN)
        count = end - start
        x0 = torch.from_numpy(images[start:end]).to(device)
        condition = torch.from_numpy(conditions[start:end]).to(device)
        t = torch.randint(
            0,
            T,
            (count, CARL_TRAIN_MC),
            generator=generator,
            device=device,
            dtype=torch.long,
        )
        noise = torch.randn(
            (count, CARL_TRAIN_MC, 3, 3, 3),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        flat_x0 = x0[:, None].expand(
            -1, CARL_TRAIN_MC, -1, -1, -1
        ).reshape(-1, 3, 3, 3)
        flat_condition = condition[:, None].expand(
            -1, CARL_TRAIN_MC, -1
        ).reshape(-1, condition.shape[-1])
        flat_t = t.reshape(-1)
        flat_noise = noise.reshape(-1, 3, 3, 3)
        xt = base.q_sample(flat_x0, flat_t, flat_noise, schedule)

        prediction = baseline_model(xt, flat_t, flat_condition)
        before = (
            (prediction.double() - flat_noise.double())
            .square()
            .flatten(1)
            .mean(1)
            .reshape(count, CARL_TRAIN_MC)
        )
        del prediction
        prediction = updated_model(xt, flat_t, flat_condition)
        after = (
            (prediction.double() - flat_noise.double())
            .square()
            .flatten(1)
            .mean(1)
            .reshape(count, CARL_TRAIN_MC)
        )
        decrease = before - after
        raw[start:end] = decrease.mean(dim=1).cpu().numpy()
        ratio[start:end] = (
            decrease / (before + after + CARL_EPS)
        ).mean(dim=1).cpu().numpy()
        if start == 0 or end == N_TRAIN or end % 5000 == 0:
            elapsed = time.perf_counter() - started
            eta = elapsed / end * (N_TRAIN - end)
            print(
                f"[gpu {device.index}] checkpoint score {end}/{N_TRAIN} "
                f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
                flush=True,
            )
    return raw, ratio


def score_query(
    query_id, checkpoint_lrs, dataset, images, conditions, schedule, device
):
    partial_dir = CARL_PARTIAL_DIR / f"q{query_id:02d}"
    partial_dir.mkdir(parents=True, exist_ok=True)
    partial_path = partial_dir / "state_v1.npz"
    sums = {
        "raw_uniform": np.zeros(N_TRAIN, dtype=np.float64),
        "ratio_uniform": np.zeros(N_TRAIN, dtype=np.float64),
        "raw_lr": np.zeros(N_TRAIN, dtype=np.float64),
        "ratio_lr": np.zeros(N_TRAIN, dtype=np.float64),
    }
    next_checkpoint = 0
    if partial_path.is_file():
        partial = np.load(partial_path, allow_pickle=False)
        next_checkpoint = int(partial["next_checkpoint"])
        for key in sums:
            sums[key][:] = partial[key]
        print(
            f"[gpu {device.index} q{query_id:02d}] resume checkpoint "
            f"{next_checkpoint + 1}",
            flush=True,
        )

    condition, query = query_condition(query_id, dataset, device)
    started = time.perf_counter()
    for checkpoint_index in range(
        next_checkpoint, len(CARL_CHECKPOINT_EPOCHS)
    ):
        epoch = CARL_CHECKPOINT_EPOCHS[checkpoint_index]
        checkpoint = load_checkpoint(epoch)
        checkpoint_lr = float(checkpoint_lrs[checkpoint_index])
        baseline_model = make_model(dataset, checkpoint["model_state"], device)
        updated_model = make_model(dataset, checkpoint["model_state"], device)
        config = checkpoint.get("config", {})
        optimizer = torch.optim.AdamW(
            updated_model.parameters(),
            lr=checkpoint_lr,
            betas=(
                float(config.get("adam_b1", ADAM_B1)),
                float(config.get("adam_b2", ADAM_B2)),
            ),
            eps=float(config.get("adam_eps", ADAM_EPS)),
            weight_decay=float(config.get("weight_decay", WEIGHT_DECAY)),
        )
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        for group in optimizer.param_groups:
            group["lr"] = checkpoint_lr
        clip_norm = float(config.get("grad_clip_norm", GRAD_CLIP))
        update_stats = four_reference_adamw_steps(
            updated_model,
            optimizer,
            query_id,
            condition,
            checkpoint_lr,
            clip_norm,
            device,
        )
        raw, ratio = paired_training_scores(
            baseline_model,
            updated_model,
            checkpoint_index,
            images,
            conditions,
            schedule,
            device,
        )
        sums["raw_uniform"] += raw
        sums["ratio_uniform"] += ratio
        sums["raw_lr"] += checkpoint_lr * raw
        sums["ratio_lr"] += checkpoint_lr * ratio
        with open(
            partial_dir / f"checkpoint_{checkpoint_index:02d}.json", "w"
        ) as handle:
            json.dump(
                {
                    "checkpoint_index": checkpoint_index,
                    "epoch": epoch,
                    "learning_rate": checkpoint_lr,
                    "clip_norm": clip_norm,
                    "updates": update_stats,
                    "raw_median": float(np.median(raw)),
                    "ratio_median": float(np.median(ratio)),
                },
                handle,
                indent=2,
            )

        temporary = partial_dir / "state_v1.tmp.npz"
        np.savez(
            temporary,
            next_checkpoint=np.asarray(checkpoint_index + 1, dtype=np.int64),
            **sums,
        )
        os.replace(temporary, partial_path)
        completed = checkpoint_index - next_checkpoint + 1
        remaining = len(CARL_CHECKPOINT_EPOCHS) - checkpoint_index - 1
        elapsed = time.perf_counter() - started
        eta = elapsed / completed * remaining
        print(
            f"[gpu {device.index} q{query_id:02d}] checkpoint="
            f"{checkpoint_index + 1:02d}/{len(CARL_CHECKPOINT_EPOCHS)} "
            f"epoch={epoch:03d} lr={checkpoint_lr:.3e} "
            f"raw_med={np.median(raw):+.3e} ratio_med={np.median(ratio):+.3e} "
            f"elapsed={elapsed/3600:.2f}h eta={eta/3600:.2f}h",
            flush=True,
        )
        del baseline_model, updated_model, optimizer, raw, ratio, checkpoint
        torch.cuda.empty_cache()

    normalizers = {
        "uniform": float(len(CARL_CHECKPOINT_EPOCHS)),
        "lr_weighted": float(sum(checkpoint_lrs)),
    }
    score_by_variant = {
        ("raw_decrease", "uniform"): sums["raw_uniform"] / normalizers["uniform"],
        ("difference_over_sum", "uniform"): sums["ratio_uniform"] / normalizers["uniform"],
        ("raw_decrease", "lr_weighted"): sums["raw_lr"] / normalizers["lr_weighted"],
        ("difference_over_sum", "lr_weighted"): sums["ratio_lr"] / normalizers["lr_weighted"],
    }
    for variant, score in score_by_variant.items():
        method = CARL_METHOD_BY_VARIANT[variant]
        output = ATTR_DIR / method / f"q{query_id:02d}"
        output.mkdir(parents=True, exist_ok=True)
        np.save(output / "scores.npy", score)
        with open(output / "metadata.json", "w") as handle:
            json.dump(
                {
                    "method": method,
                    "query": query,
                    "score_form": variant[0],
                    "checkpoint_weighting": variant[1],
                    "checkpoint_weight_normalizer": normalizers[variant[1]],
                    "checkpoint_epochs": list(CARL_CHECKPOINT_EPOCHS),
                    "checkpoint_learning_rates": list(checkpoint_lrs),
                    "parameter_source": "raw checkpoint",
                    "optimizer": "checkpoint AdamW m/v/step/param_groups",
                    "reference_source": "final EMA 1000-state trajectory and predicted noise",
                    "reference_updates_per_checkpoint": CARL_REFERENCE_STEPS,
                    "reference_batch_size": CARL_REFERENCE_BATCH_SIZE,
                    "training_mc": CARL_TRAIN_MC,
                    "paired_mc_before_after": True,
                    "positive_direction": "training loss decreases after reference learning",
                    "epsilon": CARL_EPS,
                },
                handle,
                indent=2,
            )
    print(f"[gpu {device.index} q{query_id:02d}] DONE", flush=True)


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
    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    images, conditions = dataset_arrays(dataset)
    schedule = base.make_linear_schedule(T, device=device)
    checkpoint_lrs = tuple(
        float(load_checkpoint(epoch)["learning_rate_at_checkpoint"])
        for epoch in CARL_CHECKPOINT_EPOCHS
    )
    query_ids = list(CARL_QUERY_IDS)[args.shard_index :: args.shard_count]
    print(
        f"[gpu {args.gpu}] checkpoint-AdamW reference learning qids={query_ids}",
        flush=True,
    )
    for query_id in query_ids:
        score_query(
            query_id,
            checkpoint_lrs,
            dataset,
            images,
            conditions,
            schedule,
            device,
        )


if __name__ == "__main__":
    main()
