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



def format_seconds(sec):
    sec = max(0, int(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lr_at(step, total_steps):
    warm = int(math.ceil(total_steps * WARMUP_RATIO))
    if warm > 0 and step < warm:
        return PEAK_LR * float(step + 1) / float(warm)
    if total_steps <= warm:
        return PEAK_LR
    p = (step - warm) / max(1, total_steps - warm)
    p = min(max(p, 0.0), 1.0)
    return 0.5 * PEAK_LR * (1.0 + math.cos(math.pi * p))


@torch.no_grad()
def ema_update(ema_model, model):
    for pe, p in zip(ema_model.parameters(), model.parameters()):
        pe.mul_(EMA_DECAY).add_(p, alpha=1.0 - EMA_DECAY)


def out_dir_for(kind, family, lds_seed=None, subset_id=None):
    if kind == "base":
        return MODEL_DIR / "base" / family
    return MODEL_DIR / "subsets" / f"seed_{int(lds_seed):02d}" / f"subset_{int(subset_id):02d}" / family


def save_ckpt(path, model, ema_model, opt, ds, epoch, global_step, family, train_indices, eta):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "ema_model_state": ema_model.state_dict(),
        "optimizer_state": opt.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "eta": float(eta),
        "learning_rate_at_checkpoint": float(eta),
        "T": int(T),
        "cond_dim": int(len(ds.vocab)),
        "vocab": ds.vocab,
        "grid_size": 3,
        "base_ch": int(BASE_CH),
        "time_dim": int(TIME_DIM),
        "family": family,
        "unprompted": bool(family == "unprompted"),
        "train_indices": None if train_indices is None else np.asarray(train_indices, dtype=np.int64),
        "config": {
            "batch_size": BATCH_SIZE, "epochs": EPOCHS, "learning_rate": PEAK_LR,
            "lr_schedule": "cosine_warmup", "lr_warmup_ratio": WARMUP_RATIO,
            "weight_decay": WEIGHT_DECAY, "adam_b1": ADAM_B1, "adam_b2": ADAM_B2,
            "adam_eps": ADAM_EPS, "grad_clip_norm": GRAD_CLIP, "ema_decay": EMA_DECAY,
            "timesteps": T, "base_channels": BASE_CH, "time_emb_dim": TIME_DIM,
        }
    }
    torch.save(payload, path)


def train_one(kind, family, seed, gpu, mask_path=None, lds_seed=None, subset_id=None):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    ds = ColorGridDataset(str(BASE_CSV), grid_size=3)
    train_indices = None
    train_ds = ds
    save_every = BASE_SAVE_EVERY_EPOCHS

    if kind != "base":
        train_indices = np.load(mask_path).astype(np.int64)
        train_ds = Subset(ds, train_indices.tolist())
        save_every = SUBSET_SAVE_EVERY_EPOCHS

    loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        drop_last=False, pin_memory=torch.cuda.is_available(),
    )

    x, c = ds[0]
    model = base.CondEpsModel(
        in_ch=int(x.shape[0]), cond_dim=int(c.numel()),
        base_ch=BASE_CH, time_dim=TIME_DIM,
    ).to(device)
    ema_model = copy.deepcopy(model).to(device).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    sched = base.make_linear_schedule(T, device=device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=PEAK_LR,
        betas=(ADAM_B1, ADAM_B2), eps=ADAM_EPS, weight_decay=WEIGHT_DECAY,
    )

    total_steps = EPOCHS * len(loader)
    global_step = 0
    out_dir = out_dir_for(kind, family, lds_seed, subset_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "kind": kind, "family": family, "seed": int(seed), "gpu": int(gpu),
        "base_csv": str(BASE_CSV), "mask_path": mask_path,
        "lds_seed": lds_seed, "subset_id": subset_id,
        "n_train": len(train_ds),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    model_t0 = time.perf_counter()

    train_t0 = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for x0, cond in loader:
            x0 = x0.to(device, non_blocking=True)
            cond = cond.to(device, non_blocking=True)
            if family == "unprompted":
                cond = torch.zeros_like(cond)

            lr = lr_at(global_step, total_steps)
            for pg in opt.param_groups:
                pg["lr"] = lr

            b = x0.shape[0]
            t = torch.randint(0, T, (b,), device=device, dtype=torch.long)
            noise = torch.randn_like(x0)
            xt = base.q_sample(x0, t, noise, sched)
            pred = model(xt, t, cond)
            loss = F.mse_loss(pred, noise)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            ema_update(ema_model, model)

            global_step += 1
            losses.append(float(loss.detach().item()))

        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            elapsed = time.perf_counter() - model_t0
            sec_per_epoch = elapsed / float(epoch)
            eta_sec = sec_per_epoch * float(EPOCHS - epoch)
            print(
                f"[{kind}/{family}] epoch={epoch:03d}/{EPOCHS} "
                f"loss={np.mean(losses):.6f} lr={lr:.3e} | "
                f"elapsed={format_seconds(elapsed)} | "
                f"eta={format_seconds(eta_sec)}",
                flush=True,
            )

        if epoch % save_every == 0 or epoch == EPOCHS:
            save_ckpt(
                out_dir / f"epoch_{epoch:04d}.pt",
                model, ema_model, opt, ds, epoch, global_step,
                family, train_indices, lr,
            )

    print(
        f"[done] {kind}/{family} -> {out_dir} | "
        f"elapsed={format_seconds(time.perf_counter() - model_t0)}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["base", "subset"], required=True)
    ap.add_argument("--family", choices=["prompted", "unprompted"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--mask-path", default=None)
    ap.add_argument("--lds-seed", type=int, default=None)
    ap.add_argument("--subset-id", type=int, default=None)
    a = ap.parse_args()
    train_one(a.kind, a.family, a.seed, a.gpu, a.mask_path, a.lds_seed, a.subset_id)


if __name__ == "__main__":
    main()
