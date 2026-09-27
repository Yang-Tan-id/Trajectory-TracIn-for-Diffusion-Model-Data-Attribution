"""Train one continuous five-stage model for the staged 50k experiment."""

import argparse
import copy
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import x3pixel_DM_training as base
from dataset_loader import ColorGridDataset
from staged_lds_config import *
from train_worker import ema_update, format_seconds, lr_at, set_seed


def save_checkpoint(path, model, ema_model, optimizer, dataset, epoch, step, lr, kind, mask_id):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "ema_model_state": ema_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": int(epoch),
            "global_step": int(step),
            "eta": float(lr),
            "learning_rate_at_checkpoint": float(lr),
            "T": int(T),
            "cond_dim": len(dataset.vocab),
            "vocab": dataset.vocab,
            "grid_size": 3,
            "base_ch": int(BASE_CH),
            "time_dim": int(TIME_DIM),
            "family": STAGED_FAMILY,
            "unprompted": False,
            "kind": kind,
            "mask_id": mask_id,
            "staged_training": True,
            "config": {
                "stages": STAGE_COUNT,
                "stage_size": STAGE_SIZE,
                "stage_epochs": STAGE_EPOCHS,
                "batch_size": BATCH_SIZE,
                "epochs": STAGED_EPOCHS,
                "learning_rate": PEAK_LR,
                "lr_schedule": "continuous_cosine_warmup",
                "lr_warmup_ratio": WARMUP_RATIO,
                "weight_decay": WEIGHT_DECAY,
                "adam_b1": ADAM_B1,
                "adam_b2": ADAM_B2,
                "adam_eps": ADAM_EPS,
                "grad_clip_norm": GRAD_CLIP,
                "ema_decay": EMA_DECAY,
            },
        },
        path,
    )


def train(kind, mask_id, gpu):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    set_seed(TRAIN_SEED)
    stages = np.load(STAGED_PARTITION_DIR / "stages.npy")
    membership = None
    if kind == "subset":
        membership = np.load(STAGED_MASK_DIR / "membership.npy", mmap_mode="r")[mask_id]
    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    x, c = dataset[0]
    model = base.CondEpsModel(int(x.shape[0]), int(c.numel()), BASE_CH, TIME_DIM).to(device)
    ema_model = copy.deepcopy(model).to(device).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=PEAK_LR, betas=(ADAM_B1, ADAM_B2),
        eps=ADAM_EPS, weight_decay=WEIGHT_DECAY,
    )
    sched = base.make_linear_schedule(T, device=device)
    examples_per_stage = STAGE_SIZE if kind == "base" else STAGED_LDS_PER_STAGE
    steps_per_epoch = math.ceil(examples_per_stage / BATCH_SIZE)
    total_steps = STAGED_EPOCHS * steps_per_epoch
    global_step = 0
    started = time.perf_counter()
    for stage_id, stage_indices in enumerate(stages):
        indices = np.asarray(stage_indices, dtype=np.int64)
        if membership is not None:
            indices = indices[np.asarray(membership[indices], dtype=bool)]
        if len(indices) != examples_per_stage:
            raise RuntimeError(f"stage {stage_id} has {len(indices)} examples")
        loader = DataLoader(
            Subset(dataset, indices.tolist()), batch_size=BATCH_SIZE, shuffle=True,
            num_workers=0, drop_last=False, pin_memory=torch.cuda.is_available(),
        )
        for local_epoch in range(1, STAGE_EPOCHS + 1):
            epoch = stage_id * STAGE_EPOCHS + local_epoch
            losses = []
            model.train()
            for x0, cond in loader:
                x0 = x0.to(device, non_blocking=True)
                cond = cond.to(device, non_blocking=True)
                lr = lr_at(global_step, total_steps)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                t = torch.randint(0, T, (x0.shape[0],), device=device)
                noise = torch.randn_like(x0)
                pred = model(base.q_sample(x0, t, noise, sched), t, cond)
                loss = F.mse_loss(pred, noise)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                ema_update(ema_model, model)
                global_step += 1
                losses.append(float(loss.detach()))
            if local_epoch == 1 or epoch % 10 == 0 or local_epoch == STAGE_EPOCHS:
                elapsed = time.perf_counter() - started
                eta = elapsed / epoch * (STAGED_EPOCHS - epoch)
                print(
                    f"[{kind} mask={mask_id} gpu={gpu}] stage={stage_id+1}/{STAGE_COUNT} "
                    f"epoch={epoch}/{STAGED_EPOCHS} loss={np.mean(losses):.6f} lr={lr:.3e} "
                    f"elapsed={format_seconds(elapsed)} eta={format_seconds(eta)}",
                    flush=True,
                )
            if kind == "base" and epoch % STAGED_SAVE_EVERY == 0:
                save_checkpoint(
                    staged_base_checkpoint(epoch), model, ema_model, optimizer,
                    dataset, epoch, global_step, lr, kind, mask_id,
                )
            elif kind == "subset" and epoch == STAGED_EPOCHS:
                save_checkpoint(
                    staged_subset_checkpoint(mask_id), model, ema_model, optimizer,
                    dataset, epoch, global_step, lr, kind, mask_id,
                )
    print(f"[done] {kind} mask={mask_id}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("base", "subset"), required=True)
    parser.add_argument("--mask-id", type=int)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if args.kind == "subset" and args.mask_id is None:
        parser.error("--mask-id is required for subset training")
    train(args.kind, args.mask_id, args.gpu)


if __name__ == "__main__":
    main()
