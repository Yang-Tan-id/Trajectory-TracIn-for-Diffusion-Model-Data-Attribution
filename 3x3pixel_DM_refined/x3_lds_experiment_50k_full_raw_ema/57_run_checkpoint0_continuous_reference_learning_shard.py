"""Continuously learn a reference trajectory from saved checkpoint index zero."""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import x3pixel_DM_training as base
from checkpoint0_continuous_reference_learning_config import *
from dataset_loader import ColorGridDataset
from train_worker import lr_at
from importlib import import_module


shared = import_module("53_run_checkpoint_adamw_reference_learning_shard")


def atomic_torch_save(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def full_reference_loss(model, trajectory, targets, timestamps, condition, device):
    total = 0.0
    for batch_index in range(CARL_REFERENCE_STEPS):
        start = batch_index * CARL_REFERENCE_BATCH_SIZE
        end = start + CARL_REFERENCE_BATCH_SIZE
        x = torch.from_numpy(np.asarray(trajectory[start:end])).to(device)
        target = torch.from_numpy(np.asarray(targets[start:end])).to(device)
        t = torch.from_numpy(
            np.asarray(timestamps[start:end], dtype=np.int64)
        ).to(device)
        prediction = model(x, t, condition.expand(end - start, -1))
        total += float(F.mse_loss(prediction, target).item())
    return total / float(CARL_REFERENCE_STEPS)


def reference_update(
    model,
    optimizer,
    trajectory,
    targets,
    timestamps,
    condition,
    batch_index,
    clip_norm,
    device,
):
    start = batch_index * CARL_REFERENCE_BATCH_SIZE
    end = start + CARL_REFERENCE_BATCH_SIZE
    x = torch.from_numpy(np.asarray(trajectory[start:end])).to(device)
    target = torch.from_numpy(np.asarray(targets[start:end])).to(device)
    t = torch.from_numpy(
        np.asarray(timestamps[start:end], dtype=np.int64)
    ).to(device)
    optimizer.zero_grad(set_to_none=True)
    loss = F.mse_loss(
        model(x, t, condition.expand(end - start, -1)), target
    )
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    optimizer.step()
    return float(loss.detach().item()), float(gradient_norm)


def learn_reference(query_id, dataset, schedule, device):
    work_dir = C0RL_ROOT / f"q{query_id:02d}"
    work_dir.mkdir(parents=True, exist_ok=True)
    final_path = work_dir / "final_model.pt"
    resume_path = work_dir / "resume.pt"
    checkpoint = shared.load_checkpoint(C0RL_INITIAL_EPOCH)
    baseline_model = shared.make_model(
        dataset, checkpoint["model_state"], device
    )
    condition, query = shared.query_condition(query_id, dataset, device)
    query_dir = FLA_QUERY_DIR / f"q{query_id:02d}"
    trajectory = np.load(query_dir / "trajectory_xt_1000.npy", mmap_mode="r")
    targets = np.load(query_dir / "target_eps_ema_1000.npy", mmap_mode="r")
    timestamps = np.load(
        query_dir / "trajectory_t_1000.npy", mmap_mode="r"
    )
    initial_loss = full_reference_loss(
        baseline_model,
        trajectory,
        targets,
        timestamps,
        condition,
        device,
    )
    target_loss = C0RL_TARGET_FRACTION * initial_loss

    if final_path.is_file():
        payload = torch.load(
            final_path, map_location=device, weights_only=False
        )
        model = shared.make_model(dataset, payload["model_state"], device)
        print(
            f"[gpu {device.index} q{query_id:02d}] reuse final model "
            f"steps={payload['steps']} ratio={payload['final_loss']/initial_loss:.4f}",
            flush=True,
        )
        return baseline_model, model, query, payload

    model = shared.make_model(dataset, checkpoint["model_state"], device)
    config = checkpoint.get("config", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(checkpoint["learning_rate_at_checkpoint"]),
        betas=(
            float(config.get("adam_b1", ADAM_B1)),
            float(config.get("adam_b2", ADAM_B2)),
        ),
        eps=float(config.get("adam_eps", ADAM_EPS)),
        weight_decay=float(config.get("weight_decay", WEIGHT_DECAY)),
    )
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    clip_norm = float(config.get("grad_clip_norm", GRAD_CLIP))
    total_training_steps = EPOCHS * int(np.ceil(N_TRAIN / BATCH_SIZE))
    initial_global_step = int(checkpoint["global_step"])
    start_step = 0
    history = []
    if resume_path.is_file():
        resume = torch.load(
            resume_path, map_location=device, weights_only=False
        )
        model.load_state_dict(resume["model_state"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state"])
        start_step = int(resume["next_step"])
        history = list(resume.get("history", []))
        print(
            f"[gpu {device.index} q{query_id:02d}] resume step={start_step}",
            flush=True,
        )

    reached = False
    final_loss = initial_loss
    started = time.perf_counter()
    for step in range(start_step, C0RL_MAX_STEPS):
        learning_rate = float(
            lr_at(initial_global_step + step, total_training_steps)
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        batch_index = step % CARL_REFERENCE_STEPS
        batch_loss, gradient_norm = reference_update(
            model,
            optimizer,
            trajectory,
            targets,
            timestamps,
            condition,
            batch_index,
            clip_norm,
            device,
        )
        final_loss = full_reference_loss(
            model,
            trajectory,
            targets,
            timestamps,
            condition,
            device,
        )
        completed_steps = step + 1
        loss_fraction = final_loss / initial_loss
        record = {
            "step": completed_steps,
            "reference_batch": batch_index,
            "batch_loss_before_update": batch_loss,
            "full_reference_loss_after_update": final_loss,
            "loss_fraction": loss_fraction,
            "learning_rate": learning_rate,
            "gradient_norm_before_clip": gradient_norm,
        }
        history.append(record)
        print(
            f"[gpu {device.index} q{query_id:02d}] step="
            f"{completed_steps:03d}/{C0RL_MAX_STEPS} batch={batch_index} "
            f"lr={learning_rate:.3e} ref={final_loss:.7g} "
            f"fraction={loss_fraction:.3%} grad={gradient_norm:.4g} "
            f"elapsed={(time.perf_counter()-started)/60:.1f}m",
            flush=True,
        )
        reached = final_loss <= target_loss
        if reached or completed_steps % C0RL_RESUME_EVERY == 0:
            atomic_torch_save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "next_step": completed_steps,
                    "initial_loss": initial_loss,
                    "target_loss": target_loss,
                    "last_loss": final_loss,
                    "history": history,
                },
                resume_path,
            )
        if reached:
            break

    payload = {
        "model_state": model.state_dict(),
        "query_id": query_id,
        "initial_checkpoint_epoch": C0RL_INITIAL_EPOCH,
        "initial_global_step": initial_global_step,
        "steps": len(history),
        "initial_loss": initial_loss,
        "target_loss": target_loss,
        "target_fraction": C0RL_TARGET_FRACTION,
        "final_loss": final_loss,
        "final_loss_fraction": final_loss / initial_loss,
        "target_reached": reached,
        "max_steps": C0RL_MAX_STEPS,
        "optimizer": "checkpoint AdamW state with continued original LR schedule",
        "clip_norm": clip_norm,
        "reference_noise": None,
        "history": history,
    }
    atomic_torch_save(payload, final_path)
    print(
        f"[gpu {device.index} q{query_id:02d}] learning done "
        f"steps={payload['steps']} reached={reached} "
        f"fraction={payload['final_loss_fraction']:.3%}",
        flush=True,
    )
    return baseline_model, model, query, payload


def run_query(query_id, dataset, images, conditions, schedule, device):
    outputs = {
        "raw": ATTR_DIR / C0RL_METHOD_RAW / f"q{query_id:02d}",
        "ratio": ATTR_DIR / C0RL_METHOD_RATIO / f"q{query_id:02d}",
    }
    if all((path / "scores.npy").is_file() for path in outputs.values()):
        print(f"[gpu {device.index} q{query_id:02d}] skip completed", flush=True)
        return
    baseline_model, changed_model, query, learning = learn_reference(
        query_id, dataset, schedule, device
    )
    raw, ratio = shared.paired_training_scores(
        baseline_model,
        changed_model,
        0,
        images,
        conditions,
        schedule,
        device,
    )
    for label, method, score in (
        ("raw_decrease", C0RL_METHOD_RAW, raw),
        ("difference_over_sum", C0RL_METHOD_RATIO, ratio),
    ):
        if not np.isfinite(score).all():
            raise RuntimeError(f"q{query_id:02d} {label} has non-finite scores")
        output = ATTR_DIR / method / f"q{query_id:02d}"
        output.mkdir(parents=True, exist_ok=True)
        np.save(output / "scores.npy", score)
        with open(output / "metadata.json", "w") as handle:
            json.dump(
                {
                    "method": method,
                    "query": query,
                    "score_form": label,
                    "initial_checkpoint_epoch": C0RL_INITIAL_EPOCH,
                    "training_mc": CARL_TRAIN_MC,
                    "paired_mc_before_after": True,
                    "positive_direction": "loss decreases from initial to final model",
                    "epsilon": CARL_EPS,
                    "learning": {
                        key: value
                        for key, value in learning.items()
                        if key not in ("model_state", "history")
                    },
                },
                handle,
                indent=2,
            )
    print(f"[gpu {device.index} q{query_id:02d}] scores saved", flush=True)


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
    images, conditions = shared.dataset_arrays(dataset)
    schedule = base.make_linear_schedule(T, device=device)
    query_ids = list(CARL_QUERY_IDS)[args.shard_index :: args.shard_count]
    print(f"[gpu {args.gpu}] continuous checkpoint-0 qids={query_ids}", flush=True)
    for query_id in query_ids:
        run_query(
            query_id, dataset, images, conditions, schedule, device
        )


if __name__ == "__main__":
    main()
