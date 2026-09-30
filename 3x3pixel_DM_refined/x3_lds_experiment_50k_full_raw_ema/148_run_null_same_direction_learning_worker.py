"""Train one null-model copy per point along one noise direction and evaluate."""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import x3pixel_DM_training as base
from dataset_loader import ColorGridDataset
from null_same_direction_learning_config import *


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def atomic_torch_save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def make_model(dataset, state, device):
    model = base.CondEpsModel(
        in_ch=3,
        cond_dim=len(dataset.vocab),
        base_ch=BASE_CH,
        time_dim=TIME_DIM,
    ).to(device)
    model.load_state_dict(state, strict=True)
    return model.eval()


def checkpoint_learning_rate(checkpoint):
    if "learning_rate_at_checkpoint" in checkpoint:
        return float(checkpoint["learning_rate_at_checkpoint"])
    groups = checkpoint["optimizer_state"].get("param_groups", [])
    if groups and "lr" in groups[0]:
        return float(groups[0]["lr"])
    raise KeyError("null checkpoint has no saved learning rate")


def make_optimizer(model, checkpoint, learning_rate):
    config = checkpoint.get("config", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(
            float(config.get("adam_b1", ADAM_B1)),
            float(config.get("adam_b2", ADAM_B2)),
        ),
        eps=float(config.get("adam_eps", ADAM_EPS)),
        weight_decay=float(config.get("weight_decay", WEIGHT_DECAY)),
    )
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return optimizer, float(config.get("grad_clip_norm", GRAD_CLIP))


def fixed_direction(datapoint_index, device):
    generator = torch.Generator(device=device)
    generator.manual_seed(NSDL_DIRECTION_SEED_BASE + int(datapoint_index))
    return torch.randn(
        (3, 3, 3), generator=generator, device=device, dtype=torch.float32
    )


def evaluation_directions(datapoint_index, training_direction, device):
    generator = torch.Generator(device=device)
    generator.manual_seed(
        NSDL_RANDOM_DIRECTION_SEED_BASE + int(datapoint_index)
    )
    random_directions = torch.randn(
        (NSDL_RANDOM_DIRECTION_COUNT, 3, 3, 3),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    training_norm = training_direction.norm()
    random_directions = random_directions * (
        training_norm
        / random_directions.flatten(1).norm(dim=1).clamp_min(NSDL_EPS)
    )[:, None, None, None]
    directions = torch.cat(
        (
            training_direction.unsqueeze(0),
            (-training_direction).unsqueeze(0),
            random_directions,
        ),
        dim=0,
    )
    axis = training_direction.flatten() / training_norm.clamp_min(NSDL_EPS)
    cosine = (
        directions.flatten(1)
        / directions.flatten(1).norm(dim=1, keepdim=True).clamp_min(NSDL_EPS)
    ) @ axis
    return directions, cosine


@torch.no_grad()
def losses_by_direction(model, x0, condition, directions, schedule, device):
    timestamps = torch.tensor(
        NSDL_TIMESTAMPS, device=device, dtype=torch.long
    )
    losses = torch.zeros(
        len(directions), device=device, dtype=torch.float64
    )
    for direction_start in range(
        0, len(directions), NSDL_EVAL_DIRECTION_BATCH
    ):
        direction_end = min(
            direction_start + NSDL_EVAL_DIRECTION_BATCH, len(directions)
        )
        bank = directions[direction_start:direction_end]
        count = len(bank)
        total = torch.zeros(count, device=device, dtype=torch.float64)
        for timestamp_start in range(0, T, NSDL_UPDATE_BATCH_SIZE):
            t = timestamps[
                timestamp_start : timestamp_start + NSDL_UPDATE_BATCH_SIZE
            ]
            width = len(t)
            flat_t = t.unsqueeze(0).expand(count, -1).reshape(-1)
            flat_noise = bank[:, None].expand(
                -1, width, -1, -1, -1
            ).reshape(-1, 3, 3, 3)
            flat_x0 = x0.expand(count * width, -1, -1, -1)
            flat_condition = condition.expand(count * width, -1)
            xt = base.q_sample(flat_x0, flat_t, flat_noise, schedule)
            prediction = model(xt, flat_t, flat_condition)
            batch_loss = (
                (prediction.double() - flat_noise.double())
                .square()
                .flatten(1)
                .mean(1)
                .reshape(count, width)
                .sum(1)
            )
            total += batch_loss
        losses[direction_start:direction_end] = total / float(T)
    return losses


def four_updates(
    model,
    optimizer,
    clip_norm,
    x0,
    condition,
    direction,
    schedule,
    device,
):
    timestamps = torch.tensor(
        NSDL_TIMESTAMPS, device=device, dtype=torch.long
    )
    history = []
    for update in range(NSDL_UPDATE_COUNT):
        start = update * NSDL_UPDATE_BATCH_SIZE
        end = start + NSDL_UPDATE_BATCH_SIZE
        t = timestamps[start:end]
        noise = direction.unsqueeze(0).expand(len(t), -1, -1, -1)
        xt = base.q_sample(x0.expand(len(t), -1, -1, -1), t, noise, schedule)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(xt, t, condition.expand(len(t), -1))
        loss = F.mse_loss(prediction, noise)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), clip_norm
        )
        optimizer.step()
        history.append(
            {
                "update": update + 1,
                "timestamp_start": int(t[0]),
                "timestamp_end": int(t[-1]),
                "batch_size": len(t),
                "loss": float(loss.detach()),
                "gradient_norm_before_clip": float(gradient_norm),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
    return history


def summarize_group(before, after):
    decrease = before - after
    relative = decrease / (before + after + NSDL_EPS)
    return {
        "count": int(len(before)),
        "before_mean": float(before.mean()),
        "before_std": float(before.std(unbiased=False)),
        "after_mean": float(after.mean()),
        "after_std": float(after.std(unbiased=False)),
        "decrease_mean": float(decrease.mean()),
        "decrease_std": float(decrease.std(unbiased=False)),
        "relative_decrease_mean": float(relative.mean()),
        "fraction_improved": float((decrease > 0).double().mean()),
    }


def run_datapoint(datapoint_index, dataset, checkpoint, schedule, device):
    output_dir = NSDL_POINT_DIR / f"i{datapoint_index:05d}"
    done_path = output_dir / "result.json"
    if done_path.is_file():
        print(f"[gpu {device.index}] skip datapoint={datapoint_index}", flush=True)
        return
    image, condition = dataset[int(datapoint_index)]
    x0 = image.unsqueeze(0).to(device)
    condition = condition.unsqueeze(0).to(device)
    direction = fixed_direction(datapoint_index, device)
    directions, cosine = evaluation_directions(
        datapoint_index, direction, device
    )
    baseline = make_model(dataset, checkpoint["model_state"], device)
    updated = make_model(dataset, checkpoint["model_state"], device)
    learning_rate = checkpoint_learning_rate(checkpoint)
    optimizer, clip_norm = make_optimizer(
        updated, checkpoint, learning_rate
    )
    started = time.perf_counter()
    before = losses_by_direction(
        baseline, x0, condition, directions, schedule, device
    )
    updates = four_updates(
        updated,
        optimizer,
        clip_norm,
        x0,
        condition,
        direction,
        schedule,
        device,
    )
    after = losses_by_direction(
        updated, x0, condition, directions, schedule, device
    )
    decrease = before - after
    relative = decrease / (before + after + NSDL_EPS)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "direction_losses.npz",
        directions=directions.cpu().numpy(),
        cosine_to_training=cosine.cpu().numpy(),
        before=before.cpu().numpy(),
        after=after.cpu().numpy(),
        decrease=decrease.cpu().numpy(),
        relative_decrease=relative.cpu().numpy(),
    )
    atomic_torch_save(
        output_dir / "updated_model.pt",
        {
            "model_state": {
                key: value.detach().cpu()
                for key, value in updated.state_dict().items()
            },
            "optimizer_state": optimizer.state_dict(),
            "datapoint_index": int(datapoint_index),
            "updates": updates,
        },
    )
    result = {
        "datapoint_index": int(datapoint_index),
        "null_epoch": NSDL_NULL_EPOCH,
        "null_checkpoint": str(nsdl_checkpoint_path()),
        "learning_rate": learning_rate,
        "clip_norm": clip_norm,
        "direction_norm": float(direction.norm()),
        "training_timestamps": list(NSDL_TIMESTAMPS),
        "updates": updates,
        "same_direction": summarize_group(before[:1], after[:1]),
        "opposite_direction": summarize_group(before[1:2], after[1:2]),
        "random_directions": summarize_group(before[2:], after[2:]),
        "random_cosine_loss_decrease_correlation": float(
            np.corrcoef(
                cosine[2:].cpu().numpy(), decrease[2:].cpu().numpy()
            )[0, 1]
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(done_path, result)
    print(
        f"[gpu {device.index}] datapoint={datapoint_index} "
        f"same={result['same_direction']['decrease_mean']:+.6e} "
        f"opposite={result['opposite_direction']['decrease_mean']:+.6e} "
        f"random={result['random_directions']['decrease_mean']:+.6e}",
        flush=True,
    )


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
    checkpoint = torch.load(
        nsdl_checkpoint_path(), map_location="cpu", weights_only=False
    )
    schedule = base.make_linear_schedule(T, device=device)
    indices = nsdl_datapoint_indices()[args.shard_index :: args.shard_count]
    print(f"[gpu {args.gpu}] datapoints={list(indices)}", flush=True)
    for datapoint_index in indices:
        run_datapoint(
            datapoint_index, dataset, checkpoint, schedule, device
        )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
