"""Collect per-timestamp regenerated-trajectory MSE for existing LDS subsets."""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dataset_loader import ColorGridDataset
from exp_config import *
import x3pixel_DM_training as base


def load_collect_module():
    path = Path(__file__).with_name("05_collect_subset_outputs.py")
    spec = importlib.util.spec_from_file_location("collect_subset_outputs", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def trajectory_mse_per_t(model, sched, cond, x_T, ref_traj, device):
    save_steps = np.linspace(0, DDIM_STEPS - 1, TRAJ_SNAPSHOTS, dtype=int).tolist()
    generated = base.ddim_sample(
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
    if len(generated) != len(ref_traj):
        raise RuntimeError(f"trajectory mismatch {len(generated)} != {len(ref_traj)}")
    return np.asarray([
        F.mse_loss(
            generated[k],
            torch.from_numpy(ref_traj[k]).to(device=device, dtype=torch.float32),
        ).item()
        for k in range(len(ref_traj))
    ], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--query-shard-index", type=int, required=True)
    ap.add_argument("--query-shard-count", type=int, required=True)
    ap.add_argument("--query-ids", default="0-9")
    args = ap.parse_args()

    start_q, end_q = (int(x) for x in args.query_ids.split("-"))
    query_ids = list(range(start_q, end_q + 1))
    assigned = query_ids[args.query_shard_index :: args.query_shard_count]
    device = torch.device(f"cuda:{args.gpu}")
    collect = load_collect_module()
    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    sched = base.make_linear_schedule(T, device=device)
    with open(MASK_DIR / "manifest.json") as handle:
        masks = json.load(handle)
    with open(QUERY_DIR / "manifest.json") as handle:
        queries = json.load(handle)

    output_dir = LDS_DIR / "observed_trajectory_per_t"
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(assigned) * len(masks) * 2
    completed = 0
    started = time.perf_counter()

    for qid in assigned:
        output = output_dir / f"q{qid:02d}.npz"
        if output.is_file():
            print(f"[skip] {output}", flush=True)
            continue
        query = queries[qid]
        cond = collect.cond_for(query, dataset, device)
        query_dir = Path(query["dir"])
        ref_traj = np.load(query_dir / "trajectory_xt.npy")
        trajectory_t = np.load(query_dir / "trajectory_t.npy")
        x_T = torch.from_numpy(ref_traj[0]).to(device=device, dtype=torch.float32)
        values = {
            source: np.empty((len(masks), len(ref_traj)), dtype=np.float64)
            for source in ("ema", "raw")
        }
        for mi, mask in enumerate(masks):
            checkpoint = collect.model_path(mask, query["family"])
            for source in ("ema", "raw"):
                model = collect.build(checkpoint, source, device, dataset)
                values[source][mi] = trajectory_mse_per_t(
                    model, sched, cond, x_T, ref_traj, device
                )
                del model
                completed += 1
            if mi == 0 or (mi + 1) % 16 == 0 or mi + 1 == len(masks):
                elapsed = time.perf_counter() - started
                eta = elapsed / max(completed, 1) * max(total - completed, 0)
                print(
                    f"[per-t gpu={args.gpu}] q{qid:02d} subset={mi+1}/{len(masks)} "
                    f"elapsed={elapsed/3600:.2f}h eta~{eta/3600:.2f}h",
                    flush=True,
                )
        temp = output.with_suffix(".tmp.npz")
        np.savez(
            temp,
            query_id=np.asarray(qid),
            trajectory_t=trajectory_t,
            mse_ema=values["ema"],
            mse_raw=values["raw"],
        )
        temp.replace(output)
        print(f"[saved] {output}", flush=True)


if __name__ == "__main__":
    main()
