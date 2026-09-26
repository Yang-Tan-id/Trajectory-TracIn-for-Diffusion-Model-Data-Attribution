"""Evaluate one removal model with the query's identical x_T and prompt."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

import x3pixel_DM_training as base
from dataset_loader import ColorGridDataset
from exp_config import *


def build_model(path, source, device, dataset):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = base.CondEpsModel(3, len(dataset.vocab), BASE_CH, TIME_DIM).to(device)
    key = "model_state" if source == "raw" else "ema_model_state"
    model.load_state_dict(checkpoint[key], strict=True)
    return model.eval()


def condition_for(query, dataset, device):
    condition = torch.zeros((1, len(dataset.vocab)), device=device)
    if query["family"] == "prompted":
        for label in query["labels"]:
            condition[0, dataset.vocab[label]] = 1.0
    return condition


def trajectory(model, schedule, condition, x_T, device):
    save_steps = np.linspace(0, DDIM_STEPS - 1, TRAJ_SNAPSHOTS, dtype=np.int64).tolist()
    values = base.ddim_sample(
        model=model,
        sched=schedule,
        cond=condition,
        shape=tuple(x_T.shape),
        seed=0,
        steps=DDIM_STEPS,
        eta=0.0,
        device=str(device),
        x_T=x_T,
        save_steps=save_steps,
    )
    return np.stack([value.detach().cpu().numpy() for value in values], axis=0)


def atomic_json(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    job_dir = Path(args.job_dir)
    result_path = job_dir / "evaluation.json"
    evaluation_version = 2
    if result_path.is_file():
        with open(result_path) as handle:
            existing = json.load(handle)
        if int(existing.get("evaluation_version", 0)) == evaluation_version:
            print(f"[skip] evaluation exists: {result_path}", flush=True)
            return
    with open(job_dir / "job.json") as handle:
        job = json.load(handle)
    with open(QUERY_DIR / "manifest.json") as handle:
        queries = json.load(handle)
    query = queries[int(job["query_id"])]

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    condition = condition_for(query, dataset, device)
    query_dir = Path(query["dir"])
    reference_traj = np.load(query_dir / "trajectory_xt.npy")
    x_T = torch.from_numpy(reference_traj[0]).to(device=device, dtype=torch.float32)
    source = job["eval_param_source"]
    if source != "ema":
        raise ValueError(f"top-k removal evaluation must use EMA, got {source!r}")
    removal_path = Path(job["model_dir"]) / f"epoch_{EPOCHS:04d}.pt"
    removal_model = build_model(removal_path, source, device, dataset)
    schedule = base.make_linear_schedule(T, device=device)

    removal_traj = trajectory(removal_model, schedule, condition, x_T, device)
    if removal_traj.shape != reference_traj.shape:
        raise ValueError(
            f"trajectory shape mismatch removal={removal_traj.shape} "
            f"reference={reference_traj.shape}"
        )
    delta = removal_traj.astype(np.float64) - reference_traj.astype(np.float64)
    per_snapshot_mse = np.mean(delta ** 2, axis=tuple(range(1, delta.ndim)))
    endpoint_delta = delta[-1]

    np.save(job_dir / "removal_trajectory.npy", removal_traj)
    np.save(job_dir / "per_snapshot_mse.npy", per_snapshot_mse)
    np.save(job_dir / "endpoint_delta.npy", endpoint_delta)
    payload = {
        "evaluation_version": evaluation_version,
        "job_id": int(job["job_id"]),
        "query_id": int(job["query_id"]),
        "family": query["family"],
        "method_tag": job["method_tag"],
        "method": job["method"],
        "lambda": job["lambda"],
        "score_param_source": job["score_param_source"],
        "eval_param_source": source,
        "topk": int(job["topk"]),
        "ranking": job["ranking"],
        "same_initial_noise": True,
        "initial_seed": int(query["initial_seed"]),
        "same_prompt": True,
        "labels": query.get("labels", []),
        "trajectory_mse": float(np.mean(delta ** 2)),
        "trajectory_rmse": float(np.sqrt(np.mean(delta ** 2))),
        "trajectory_max_abs": float(np.max(np.abs(delta))),
        "endpoint_mse": float(np.mean(endpoint_delta ** 2)),
        "endpoint_rmse": float(np.sqrt(np.mean(endpoint_delta ** 2))),
        "endpoint_l2": float(np.linalg.norm(endpoint_delta.reshape(-1))),
        "endpoint_max_abs": float(np.max(np.abs(endpoint_delta))),
        "reference": "saved_base_ema_query_trajectory",
    }
    atomic_json(payload, result_path)
    print(
        f"[evaluated] {job['method_tag']} q{int(job['query_id']):02d} "
        f"trajectory_mse={payload['trajectory_mse']:.8g} "
        f"endpoint_mse={payload['endpoint_mse']:.8g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
