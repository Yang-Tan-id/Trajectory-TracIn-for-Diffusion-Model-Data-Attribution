"""
Collect LDS observed responses for BOTH raw and EMA evaluation models.

Outputs [100, 192]:
  observed_simple_loss_ema.npy
  observed_simple_loss_raw.npy
  observed_traj_ref_ema.npy
  observed_traj_ref_raw.npy
  observed_endpoint_deviation_ema.npy
  observed_endpoint_deviation_raw.npy
  observed_trajectory_state_mse_ema.npy
  observed_trajectory_state_mse_raw.npy

For endpoint/trajectory-state metrics, each subset model regenerates a full
1000-step deterministic DDIM trajectory from the SAME x_T as the query and
we compare the same 100 saved trajectory states.
"""

import json
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
        MODEL_DIR / "subsets"
        / f"seed_{int(mask_rec['lds_seed']):02d}"
        / f"subset_{int(mask_rec['subset_id']):02d}"
        / family
        / f"epoch_{EPOCHS:04d}.pt"
    )


def build(path, source, device, ds):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = base.CondEpsModel(3, len(ds.vocab), BASE_CH, TIME_DIM).to(device)
    state = ck["model_state" if source == "raw" else "ema_model_state"]
    m.load_state_dict(state, strict=True)
    return m.eval()


def cond_for(rec, ds, device):
    c = torch.zeros((1, len(ds.vocab)), device=device)
    if rec["family"] == "unprompted":
        return c
    for lab in rec.get("labels", []):
        c[0, ds.vocab[lab]] = 1.0
    return c


@torch.no_grad()
def simple_loss_metric(model, x0, cond, sched, qid):
    vals = []
    for tval in LDS_EVAL_TIMESTEPS:
        tval = int(tval)
        t = torch.tensor([tval], device=x0.device, dtype=torch.long)
        for mc in range(int(LDS_EVAL_MC)):
            gen = torch.Generator(device=str(x0.device))
            gen.manual_seed(90000000 + qid * 10000 + tval * 10 + mc)
            noise = torch.randn(
                x0.shape,
                generator=gen,
                device=x0.device,
                dtype=x0.dtype,
            )
            xt = base.q_sample(x0, t, noise, sched)
            vals.append(F.mse_loss(model(xt, t, cond), noise).item())
    return float(np.mean(vals))


@torch.no_grad()
def traj_ref_metric(model, traj, t_seq, cond, ref_model):
    vals = []
    device = next(model.parameters()).device
    for si, tval in enumerate(t_seq.tolist()):
        xt = torch.from_numpy(traj[si]).to(device=device, dtype=torch.float32)
        t = torch.tensor([int(tval)], device=device, dtype=torch.long)
        vals.append(
            (model(xt, t, cond) - ref_model(xt, t, cond))
            .pow(2)
            .sum()
            .item()
        )
    return float(np.mean(vals))


@torch.no_grad()
def regenerated_trajectory_metrics(
    model,
    sched,
    cond,
    x_T,
    x0_ref,
    ref_traj,
    device,
):
    save_steps = np.linspace(
        0,
        DDIM_STEPS - 1,
        TRAJ_SNAPSHOTS,
        dtype=int,
    ).tolist()

    subset_traj = base.ddim_sample(
        model=model,
        sched=sched,
        cond=cond,
        shape=tuple(x_T.shape),
        seed=0,
        steps=DDIM_STEPS,
        eta=0.0,
        device=str(device),
        x_T=x_T,
        save_steps=save_steps,
    )

    if len(subset_traj) != len(ref_traj):
        raise RuntimeError(
            f"trajectory mismatch subset={len(subset_traj)} ref={len(ref_traj)}"
        )

    state_mses = []
    for k in range(len(ref_traj)):
        x_ref = torch.from_numpy(ref_traj[k]).to(
            device=device,
            dtype=torch.float32,
        )
        state_mses.append(
            F.mse_loss(subset_traj[k], x_ref).item()
        )

    trajectory_state_mse = float(np.mean(state_mses))
    endpoint_deviation = float(
        F.mse_loss(subset_traj[-1], x0_ref).item()
    )

    return endpoint_deviation, trajectory_state_mse


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)

    ds = ColorGridDataset(str(BASE_CSV), grid_size=3)
    sched = base.make_linear_schedule(T, device=device)

    with open(MASK_DIR / "manifest.json") as f:
        masks = json.load(f)
    with open(QUERY_DIR / "manifest.json") as f:
        qs = json.load(f)

    shape = (len(qs), len(masks))
    arrays = {
        "simple_loss_ema": np.zeros(shape, dtype=np.float64),
        "simple_loss_raw": np.zeros(shape, dtype=np.float64),
        "traj_ref_ema": np.zeros(shape, dtype=np.float64),
        "traj_ref_raw": np.zeros(shape, dtype=np.float64),
        "endpoint_deviation_ema": np.zeros(shape, dtype=np.float64),
        "endpoint_deviation_raw": np.zeros(shape, dtype=np.float64),
        "trajectory_state_mse_ema": np.zeros(shape, dtype=np.float64),
        "trajectory_state_mse_raw": np.zeros(shape, dtype=np.float64),
    }

    ref_models = {}
    for family in FAMILIES:
        p = MODEL_DIR / "base" / family / f"epoch_{EPOCHS:04d}.pt"
        ref_models[(family, "ema")] = build(p, "ema", device, ds)
        ref_models[(family, "raw")] = build(p, "raw", device, ds)

    total_pairs = len(qs) * len(masks)
    done_pairs = 0
    t0 = time.perf_counter()

    for qi, q in enumerate(qs):
        family = q["family"]
        cond = cond_for(q, ds, device)
        qdir = Path(q["dir"])

        x0 = torch.from_numpy(np.load(qdir / "final_state.npy")).to(
            device=device,
            dtype=torch.float32,
        )
        traj = np.load(qdir / "trajectory_xt.npy")
        t_seq = np.load(qdir / "trajectory_t.npy")
        x_T = torch.from_numpy(traj[0]).to(
            device=device,
            dtype=torch.float32,
        )

        for mi, mr in enumerate(masks):
            p = model_path(mr, family)
            if not p.exists():
                raise FileNotFoundError(f"Missing subset checkpoint: {p}")

            for source in ("ema", "raw"):
                m = build(p, source, device, ds)

                arrays[f"simple_loss_{source}"][qi, mi] = simple_loss_metric(
                    m, x0, cond, sched, qi
                )

                arrays[f"traj_ref_{source}"][qi, mi] = traj_ref_metric(
                    m,
                    traj,
                    t_seq,
                    cond,
                    ref_models[(family, source)],
                )

                endpoint_val, trajstate_val = regenerated_trajectory_metrics(
                    model=m,
                    sched=sched,
                    cond=cond,
                    x_T=x_T,
                    x0_ref=x0,
                    ref_traj=traj,
                    device=device,
                )

                arrays[f"endpoint_deviation_{source}"][qi, mi] = endpoint_val
                arrays[f"trajectory_state_mse_{source}"][qi, mi] = trajstate_val

                del m

            done_pairs += 1

            if mi == 0 or (mi + 1) % 16 == 0 or (mi + 1) == len(masks):
                elapsed = time.perf_counter() - t0
                eta = elapsed / max(done_pairs, 1) * (total_pairs - done_pairs)

                print(
                    f"[LDS collect] q{qi:02d} subset {mi+1}/{len(masks)} | "
                    f"simple ema/raw="
                    f"{arrays['simple_loss_ema'][qi,mi]:.4f}/"
                    f"{arrays['simple_loss_raw'][qi,mi]:.4f} | "
                    f"traj ema/raw="
                    f"{arrays['traj_ref_ema'][qi,mi]:.4f}/"
                    f"{arrays['traj_ref_raw'][qi,mi]:.4f} | "
                    f"endpoint ema/raw="
                    f"{arrays['endpoint_deviation_ema'][qi,mi]:.4f}/"
                    f"{arrays['endpoint_deviation_raw'][qi,mi]:.4f} | "
                    f"state ema/raw="
                    f"{arrays['trajectory_state_mse_ema'][qi,mi]:.4f}/"
                    f"{arrays['trajectory_state_mse_raw'][qi,mi]:.4f} | "
                    f"overall {done_pairs}/{total_pairs} "
                    f"({100.0*done_pairs/total_pairs:.1f}%) | "
                    f"elapsed={elapsed/3600:.2f}h | eta≈{eta/3600:.2f}h",
                    flush=True,
                )

        LDS_DIR.mkdir(parents=True, exist_ok=True)
        for metric, arr in arrays.items():
            np.save(LDS_DIR / f"observed_{metric}.npy", arr)

        print(f"[saved partial] q{qi:02d}", flush=True)

    for metric, arr in arrays.items():
        np.save(LDS_DIR / f"observed_{metric}.npy", arr)

    print("[done] all raw/EMA observed LDS responses saved", flush=True)


if __name__ == "__main__":
    main()
