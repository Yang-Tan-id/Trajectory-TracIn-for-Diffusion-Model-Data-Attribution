import json
import math
import random
from pathlib import Path

import numpy as np
import torch

import x3pixel_DM_training as base
from dataset_loader import ColorGridDataset
from exp_config import *


def load_model(family, param_source="ema"):
    d = MODEL_DIR / "base" / family
    p = d / f"epoch_{EPOCHS:04d}.pt"
    ck = torch.load(p, map_location="cpu")
    ds = ColorGridDataset(str(BASE_CSV), grid_size=3)
    x, c = ds[0]
    model = base.CondEpsModel(
        in_ch=int(x.shape[0]), cond_dim=int(c.numel()),
        base_ch=BASE_CH, time_dim=TIME_DIM,
    )
    key = "ema_model_state" if param_source == "ema" else "model_state"
    model.load_state_dict(ck[key], strict=True)
    return model.eval(), ck, ds


def prompt_groups(vocab):
    labs = sorted(vocab, key=vocab.get)
    groups = {
        "shape_color": [x for x in labs if x.startswith("shape_color_")],
        "background_color": [x for x in labs if x.startswith("background_color_")],
        "shape": [x for x in labs if x.startswith("shape_") and not x.startswith("shape_color_")],
    }
    if any(len(v) == 0 for v in groups.values()):
        raise RuntimeError(f"Could not form prompt groups from vocab: {groups}")
    return groups


def cond_from_labels(labels, ds):
    v = torch.zeros(len(ds.vocab), dtype=torch.float32)
    for x in labels:
        v[ds.vocab[x]] = 1.0
    return v.unsqueeze(0)


def main():
    QUERY_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    prompted_model, ck, ds = load_model("prompted", "ema")
    unprompted_model, _, _ = load_model("unprompted", "ema")
    prompted_model = prompted_model.to(device)
    unprompted_model = unprompted_model.to(device)
    sched = base.make_linear_schedule(T, device=device)

    groups = prompt_groups(ds.vocab)
    rnd = random.Random(QUERY_PROMPT_SEED)
    records = []

    # 3 prompted initial noises x 4 random prompts = 12 queries.
    qid = 0
    for init_seed in PROMPTED_INITIAL_SEEDS:
        for prompt_rep in range(RANDOM_PROMPTS_PER_PROMPTED_INITIAL):
            labels = [
                rnd.choice(groups["shape_color"]),
                rnd.choice(groups["background_color"]),
                rnd.choice(groups["shape"]),
            ]
            cond = cond_from_labels(labels, ds).to(device)
            save_steps = np.linspace(0, DDIM_STEPS - 1, TRAJ_SNAPSHOTS, dtype=np.int64).tolist()
            traj = base.ddim_sample(
                model=prompted_model, sched=sched, cond=cond,
                shape=(1, 3, 3, 3), seed=int(init_seed), steps=DDIM_STEPS,
                eta=0.0, device=str(device), save_steps=save_steps,
            )
            ts = np.linspace(T - 1, 0, DDIM_STEPS, dtype=np.int64)
            t_seq = np.asarray([int(ts[k]) for k in save_steps], dtype=np.int64)
            out = QUERY_DIR / f"q{qid:02d}"
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / "trajectory_xt.npy", np.stack([x.detach().cpu().numpy() for x in traj], axis=0))
            np.save(out / "trajectory_t.npy", t_seq)
            np.save(out / "final_state.npy", traj[-1].detach().cpu().numpy())
            rec = {
                "query_id": qid, "family": "prompted", "initial_seed": int(init_seed),
                "prompt_rep": prompt_rep, "labels": labels, "dir": str(out),
            }
            with open(out / "query.json", "w") as f:
                json.dump(rec, f, indent=2)
            records.append(rec)
            qid += 1

    # 4 unprompted initial noises = 4 queries.
    zero_cond = torch.zeros((1, len(ds.vocab)), device=device)
    for init_seed in UNPROMPTED_INITIAL_SEEDS:
        save_steps = np.linspace(0, DDIM_STEPS - 1, TRAJ_SNAPSHOTS, dtype=np.int64).tolist()
        traj = base.ddim_sample(
            model=unprompted_model, sched=sched, cond=zero_cond,
            shape=(1, 3, 3, 3), seed=int(init_seed), steps=DDIM_STEPS,
            eta=0.0, device=str(device), save_steps=save_steps,
        )
        ts = np.linspace(T - 1, 0, DDIM_STEPS, dtype=np.int64)
        t_seq = np.asarray([int(ts[k]) for k in save_steps], dtype=np.int64)
        out = QUERY_DIR / f"q{qid:02d}"
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "trajectory_xt.npy", np.stack([x.detach().cpu().numpy() for x in traj], axis=0))
        np.save(out / "trajectory_t.npy", t_seq)
        np.save(out / "final_state.npy", traj[-1].detach().cpu().numpy())
        rec = {
            "query_id": qid, "family": "unprompted", "initial_seed": int(init_seed),
            "prompt_rep": None, "labels": [], "dir": str(out),
        }
        with open(out / "query.json", "w") as f:
            json.dump(rec, f, indent=2)
        records.append(rec)
        qid += 1

    assert len(records) == 16
    with open(QUERY_DIR / "manifest.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"[done] saved {len(records)} queries -> {QUERY_DIR}")


if __name__ == "__main__":
    main()
