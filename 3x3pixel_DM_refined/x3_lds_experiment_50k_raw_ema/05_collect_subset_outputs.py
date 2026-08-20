"""
Evaluate every subset checkpoint on every query for LDS observed responses.

Produces:
  lds/observed_simple_loss.npy  shape [16, 192]
  lds/observed_traj_ref.npy     shape [16, 192]

Each query automatically uses the matching prompted/unprompted subset model.
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import x3pixel_DM_training as base
from dataset_loader import ColorGridDataset
from exp_config import *


def model_path(mask_rec, family):
    return (
        MODEL_DIR / "subsets" / f"seed_{int(mask_rec['lds_seed']):02d}"
        / f"subset_{int(mask_rec['subset_id']):02d}" / family
        / f"epoch_{EPOCHS:04d}.pt"
    )


def build(path, source, device, ds):
    ck = torch.load(path, map_location=device)
    m = base.CondEpsModel(3, len(ds.vocab), BASE_CH, TIME_DIM).to(device)
    m.load_state_dict(ck["model_state" if source=="raw" else "ema_model_state"], strict=True)
    return m.eval()


def cond_for(rec, ds, device):
    c = torch.zeros((1, len(ds.vocab)), device=device)
    for lab in rec["labels"]:
        c[0, ds.vocab[lab]] = 1
    return c


@torch.no_grad()
def simple_loss_metric(model, x0, cond, sched, qid):
    vals = []
    for tval in LDS_EVAL_TIMESTEPS:
        t = torch.tensor([int(tval)], device=x0.device, dtype=torch.long)
        for mc in range(LDS_EVAL_MC):
            gen = torch.Generator(device=str(x0.device))
            gen.manual_seed(90000000 + qid*10000 + int(tval)*10 + mc)
            noise = torch.randn(x0.shape, generator=gen, device=x0.device, dtype=x0.dtype)
            xt = base.q_sample(x0, t, noise, sched)
            vals.append(F.mse_loss(model(xt, t, cond), noise).item())
    return float(np.mean(vals))


@torch.no_grad()
def traj_ref_metric(model, traj, t_seq, cond, ref_model):
    vals = []
    for si, tval in enumerate(t_seq.tolist()):
        xt = torch.from_numpy(traj[si]).to(next(model.parameters()).device, dtype=torch.float32)
        t = torch.tensor([int(tval)], device=xt.device, dtype=torch.long)
        vals.append((model(xt,t,cond) - ref_model(xt,t,cond)).pow(2).sum().item())
    return float(np.mean(vals))


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ds = ColorGridDataset(str(BASE_CSV), grid_size=3)
    sched = base.make_linear_schedule(T, device=device)

    with open(MASK_DIR / "manifest.json") as f:
        masks = json.load(f)
    with open(QUERY_DIR / "manifest.json") as f:
        qs = json.load(f)

    simple = np.zeros((len(qs), len(masks)), dtype=np.float64)
    trajref = np.zeros_like(simple)
    collect_t0 = time.perf_counter()
    total_pairs = len(qs) * len(masks)
    done_pairs = 0

    # Cache full reference models per family.
    ref_models = {}
    for family in FAMILIES:
        p = MODEL_DIR / "base" / family / f"epoch_{EPOCHS:04d}.pt"
        ref_models[family] = build(p, "ema", device, ds)

    for qi, q in enumerate(qs):
        family = q["family"]
        cond = cond_for(q, ds, device)
        x0 = torch.from_numpy(np.load(Path(q["dir"]) / "final_state.npy")).to(device=device, dtype=torch.float32)
        traj = np.load(Path(q["dir"]) / "trajectory_xt.npy")
        t_seq = np.load(Path(q["dir"]) / "trajectory_t.npy")

        for mi, mr in enumerate(masks):
            p = model_path(mr, family)
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing subset checkpoint {p}. "
                    "If SUBSET_TRAIN_FAMILIES omitted this family, add it and train first."
                )
            m = build(p, "ema", device, ds)
            simple[qi, mi] = simple_loss_metric(m, x0, cond, sched, qi)
            trajref[qi, mi] = traj_ref_metric(m, traj, t_seq, cond, ref_models[family])
            del m
            done_pairs += 1
            if mi == 0 or (mi + 1) % 16 == 0 or (mi + 1) == len(masks):
                elapsed = time.perf_counter() - collect_t0
                eta = elapsed / float(done_pairs) * float(total_pairs - done_pairs)
                print(
                    f"[LDS collect] q{qi:02d} subset {mi+1}/{len(masks)} | "
                    f"overall {done_pairs}/{total_pairs} "
                    f"({100.0*done_pairs/total_pairs:.1f}%) | "
                    f"elapsed={elapsed/3600:.2f}h | eta≈{eta/3600:.2f}h",
                    flush=True,
                )

        print(f"[query {qi:02d}] subset outputs complete", flush=True)

    LDS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(LDS_DIR / "observed_simple_loss.npy", simple)
    np.save(LDS_DIR / "observed_traj_ref.npy", trajref)
    print("[done] observed LDS responses saved")


if __name__ == "__main__":
    main()
