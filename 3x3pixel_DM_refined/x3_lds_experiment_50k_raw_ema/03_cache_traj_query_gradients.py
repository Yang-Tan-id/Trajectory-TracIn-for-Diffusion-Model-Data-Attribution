"""
Cache 4096-D CountSketch projections of Traj-TracIn query gradients for all
16 queries, for BOTH target definitions:
  - reference checkpoint
  - next checkpoint

This is a cache/diagnostic stage. Exact Traj scoring can still use full
gradients in memory; this file prevents query-gradient work from being lost.
"""
import json
import math
import re
from pathlib import Path

import numpy as np
import torch

import x3pixel_DM_training as base
from dataset_loader import ColorGridDataset
from exp_config import *


def natural_key(p):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", p.name)]


def ckpts(family):
    return sorted((MODEL_DIR / "base" / family).glob("epoch_*.pt"), key=natural_key)


def build_model(path, source, device):
    ck = torch.load(path, map_location=device)
    ds = ColorGridDataset(str(BASE_CSV), grid_size=3)
    x, c = ds[0]
    m = base.CondEpsModel(3, len(ds.vocab), BASE_CH, TIME_DIM).to(device)
    m.load_state_dict(ck["model_state" if source == "raw" else "ema_model_state"], strict=True)
    m.eval()
    return m, ds


def cond_for(rec, ds, device):
    c = torch.zeros((1, len(ds.vocab)), device=device)
    for lab in rec["labels"]:
        c[0, ds.vocab[lab]] = 1.0
    return c


def sketch(grads, dim, seed):
    gen = torch.Generator(device=grads[0].device)
    gen.manual_seed(int(seed))
    out = torch.zeros(dim, device=grads[0].device)
    for ti, g in enumerate(grads):
        flat = g.reshape(-1).float()
        idx = torch.randint(0, dim, (flat.numel(),), generator=gen, device=flat.device)
        sign = torch.randint(0, 2, (flat.numel(),), generator=gen, device=flat.device) * 2 - 1
        out.index_add_(0, idx, sign.float() * flat)
    return out / math.sqrt(dim)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(QUERY_DIR / "manifest.json") as f:
        queries = json.load(f)

    for rec in queries:
        family = rec["family"]
        paths = ckpts(family)
        ref_path = paths[-1]
        traj = np.load(Path(rec["dir"]) / "trajectory_xt.npy")
        t_seq = np.load(Path(rec["dir"]) / "trajectory_t.npy")

        _, ds = build_model(ref_path, TRACIN_PARAM_SOURCE, device)
        cond = cond_for(rec, ds, device)

        for target_mode in ("reference", "next"):
            feats = []
            meta = []
            for ci, cur_path in enumerate(paths):
                if target_mode == "next" and ci + 1 >= len(paths):
                    continue

                model, _ = build_model(cur_path, TRACIN_PARAM_SOURCE, device)
                target_path = ref_path if target_mode == "reference" else paths[ci + 1]
                target, _ = build_model(target_path, TRACIN_PARAM_SOURCE, device)
                active = [p for p in model.parameters() if p.requires_grad]

                for si, tval in enumerate(t_seq.tolist()):
                    xt = torch.from_numpy(traj[si]).to(device=device, dtype=torch.float32)
                    t = torch.tensor([int(tval)], device=device, dtype=torch.long)
                    with torch.no_grad():
                        eps_target = target(xt, t, cond).detach()
                    eps = model(xt, t, cond)
                    f = (eps - eps_target).pow(2).sum()
                    g = torch.autograd.grad(f, active, allow_unused=False)
                    ph = sketch(g, QUERY_GRAD_CACHE_DIM, seed=10_000_000 + rec["query_id"]*100000 + ci*1000 + si)
                    feats.append(ph.detach().cpu().numpy().astype(np.float32))
                    meta.append((ci, si, int(tval)))

                del model, target

            out = ATTR_DIR / "query_grad_cache" / target_mode
            out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out / f"q{rec['query_id']:02d}.npz",
                query_features=np.stack(feats, axis=0),
                meta=np.asarray(meta, dtype=np.int32),
                query_id=np.asarray(rec["query_id"], dtype=np.int32),
                family=np.asarray(family),
                target=np.asarray(target_mode),
                proj_dim=np.asarray(QUERY_GRAD_CACHE_DIM, dtype=np.int32),
            )
            print(f"[saved] q{rec['query_id']:02d} {target_mode}: {len(feats)} terms")


if __name__ == "__main__":
    main()
