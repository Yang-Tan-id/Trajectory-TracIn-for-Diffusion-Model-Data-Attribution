"""Collect observed LDS responses for a query shard of staged subset models."""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

import x3pixel_DM_training as base
from dataset_loader import ColorGridDataset
from staged_lds_config import *


_spec = importlib.util.spec_from_file_location("original_collector", Path(__file__).with_name("05_collect_subset_outputs.py"))
original = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(original)
METRICS = (
    "simple_loss_ema", "simple_loss_raw", "traj_ref_ema", "traj_ref_raw",
    "endpoint_deviation_ema", "endpoint_deviation_raw",
    "trajectory_state_mse_ema", "trajectory_state_mse_raw",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--query-shard-index", type=int, required=True)
    parser.add_argument("--query-shard-count", type=int, default=4)
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    sched = base.make_linear_schedule(T, device=device)
    with open(STAGED_QUERY_DIR / "manifest.json") as handle:
        queries = json.load(handle)
    checkpoint = staged_base_checkpoint(STAGED_EPOCHS)
    refs = {source: original.build(checkpoint, source, device, dataset) for source in ("ema", "raw")}
    shard_dir = STAGED_LDS_DIR / "observed_query_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    query_positions = range(args.query_shard_index, len(queries), args.query_shard_count)
    started = time.perf_counter()
    for query_position in query_positions:
        query = queries[query_position]
        qid = int(query["query_id"])
        out = shard_dir / f"q{qid:02d}.npz"
        if out.is_file():
            print(f"[skip] {out}", flush=True)
            continue
        values = {metric: np.zeros(STAGED_LDS_MASK_COUNT, dtype=np.float64) for metric in METRICS}
        cond = original.cond_for(query, dataset, device)
        query_dir = Path(query["dir"])
        x0 = torch.from_numpy(np.load(query_dir / "final_state.npy")).to(device=device, dtype=torch.float32)
        trajectory = np.load(query_dir / "trajectory_xt.npy")
        t_seq = np.load(query_dir / "trajectory_t.npy")
        x_T = torch.from_numpy(trajectory[0]).to(device=device, dtype=torch.float32)
        for mask_id in range(STAGED_LDS_MASK_COUNT):
            path = staged_subset_checkpoint(mask_id)
            if not path.is_file():
                raise FileNotFoundError(path)
            for source in ("ema", "raw"):
                model = original.build(path, source, device, dataset)
                values[f"simple_loss_{source}"][mask_id] = original.simple_loss_metric(model, x0, cond, sched, qid)
                values[f"traj_ref_{source}"][mask_id] = original.traj_ref_metric(model, trajectory, t_seq, cond, refs[source])
                endpoint, trajectory_mse = original.regenerated_trajectory_metrics(
                    model, sched, cond, x_T, x0, trajectory, device,
                )
                values[f"endpoint_deviation_{source}"][mask_id] = endpoint
                values[f"trajectory_state_mse_{source}"][mask_id] = trajectory_mse
                del model
            if mask_id == 0 or (mask_id + 1) % 16 == 0 or mask_id + 1 == STAGED_LDS_MASK_COUNT:
                print(
                    f"[observed gpu={args.gpu}] q{qid:02d} subset={mask_id+1}/{STAGED_LDS_MASK_COUNT} "
                    f"elapsed={(time.perf_counter()-started)/3600:.2f}h",
                    flush=True,
                )
        temporary = out.with_suffix(".tmp.npz")
        np.savez(temporary, query_id=np.asarray(qid), **values)
        temporary.replace(out)
        print(f"[saved] {out}", flush=True)


if __name__ == "__main__":
    main()
