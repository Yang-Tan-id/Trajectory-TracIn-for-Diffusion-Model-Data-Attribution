"""Train one 49k model after removing a query-specific top-1000 set."""

import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import x3pixel_DM_training as base
from dataset_loader import ColorGridDataset
from exp_config import *
from train_worker import ema_update, format_seconds, lr_at, set_seed


RESUME_EVERY_EPOCHS = 10


def checkpoint_payload(model, ema_model, optimizer, dataset, job, epoch, global_step, lr):
    model_device = next(model.parameters()).device
    payload = {
        "model_state": model.state_dict(),
        "ema_model_state": ema_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "eta": float(lr),
        "learning_rate_at_checkpoint": float(lr),
        "T": int(T),
        "cond_dim": int(len(dataset.vocab)),
        "vocab": dataset.vocab,
        "grid_size": 3,
        "base_ch": int(BASE_CH),
        "time_dim": int(TIME_DIM),
        "family": job["family"],
        "unprompted": bool(job["family"] == "unprompted"),
        "train_indices": np.load(job["kept_indices_path"]).astype(np.int64),
        "removal_job": job,
        "config": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": PEAK_LR,
            "lr_schedule": "cosine_warmup",
            "lr_warmup_ratio": WARMUP_RATIO,
            "weight_decay": WEIGHT_DECAY,
            "adam_b1": ADAM_B1,
            "adam_b2": ADAM_B2,
            "adam_eps": ADAM_EPS,
            "grad_clip_norm": GRAD_CLIP,
            "ema_decay": EMA_DECAY,
            "timesteps": T,
            "base_channels": BASE_CH,
            "time_emb_dim": TIME_DIM,
        },
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state"] = torch.cuda.get_rng_state(model_device)
    return payload


def atomic_torch_save(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()

    job_dir = Path(args.job_dir)
    with open(job_dir / "job.json") as handle:
        job = json.load(handle)
    final_path = Path(job["model_dir"]) / f"epoch_{EPOCHS:04d}.pt"
    if final_path.is_file():
        print(f"[skip] final checkpoint exists: {final_path}", flush=True)
        return

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    set_seed(int(job["train_seed"]))
    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    kept = np.load(job["kept_indices_path"]).astype(np.int64)
    if kept.shape != (N_TRAIN - int(job["topk"]),):
        raise ValueError(f"unexpected kept-index shape: {kept.shape}")
    train_dataset = Subset(dataset, kept.tolist())
    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    x, condition = dataset[0]
    model = base.CondEpsModel(
        in_ch=int(x.shape[0]), cond_dim=int(condition.numel()),
        base_ch=BASE_CH, time_dim=TIME_DIM,
    ).to(device)
    ema_model = copy.deepcopy(model).to(device).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    schedule = base.make_linear_schedule(T, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=PEAK_LR,
        betas=(ADAM_B1, ADAM_B2), eps=ADAM_EPS, weight_decay=WEIGHT_DECAY,
    )

    start_epoch = 1
    global_step = 0
    resume_path = Path(job["model_dir"]) / "resume.pt"
    if resume_path.is_file():
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(resume["model_state"], strict=True)
        ema_model.load_state_dict(resume["ema_model_state"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state"])
        start_epoch = int(resume["epoch"]) + 1
        global_step = int(resume["global_step"])
        random.setstate(resume["python_random_state"])
        np.random.set_state(resume["numpy_random_state"])
        torch.set_rng_state(resume["torch_rng_state"].cpu())
        if torch.cuda.is_available() and "cuda_rng_state" in resume:
            torch.cuda.set_rng_state(resume["cuda_rng_state"].cpu(), device=device)
        print(f"[resume] epoch={start_epoch} global_step={global_step}", flush=True)

    total_steps = EPOCHS * len(loader)
    started = time.perf_counter()
    family = job["family"]
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        losses = []
        for x0, cond in loader:
            x0 = x0.to(device, non_blocking=True)
            cond = cond.to(device, non_blocking=True)
            if family == "unprompted":
                cond = torch.zeros_like(cond)
            lr = lr_at(global_step, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            batch = x0.shape[0]
            t = torch.randint(0, T, (batch,), device=device, dtype=torch.long)
            noise = torch.randn_like(x0)
            xt = base.q_sample(x0, t, noise, schedule)
            loss = F.mse_loss(model(xt, t, cond), noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            ema_update(ema_model, model)
            global_step += 1
            losses.append(float(loss.detach().item()))

        elapsed = time.perf_counter() - started
        if epoch == start_epoch or epoch % 10 == 0 or epoch == EPOCHS:
            completed_here = epoch - start_epoch + 1
            eta = elapsed / max(completed_here, 1) * (EPOCHS - epoch)
            print(
                f"[topk/{job['method_tag']}/q{int(job['query_id']):02d}/{family}] "
                f"epoch={epoch:03d}/{EPOCHS} loss={np.mean(losses):.6f} lr={lr:.3e} "
                f"elapsed={format_seconds(elapsed)} eta={format_seconds(eta)}",
                flush=True,
            )
        if epoch % RESUME_EVERY_EPOCHS == 0 and epoch < EPOCHS:
            atomic_torch_save(
                checkpoint_payload(model, ema_model, optimizer, dataset, job, epoch, global_step, lr),
                resume_path,
            )

    payload = checkpoint_payload(
        model, ema_model, optimizer, dataset, job, EPOCHS, global_step, lr
    )
    atomic_torch_save(payload, final_path)
    print(f"[done] saved {final_path}", flush=True)


if __name__ == "__main__":
    main()
