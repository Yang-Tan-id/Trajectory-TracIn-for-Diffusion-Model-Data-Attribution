from __future__ import annotations

"""
Compare sampler trajectory outputs against a reference model trajectory.

This script reads trajectories saved by DM___sampler.py:

  <run_dir>/run_info.json
  <run_dir>/sample_000/trajectory.npy

It aligns trajectories by saved diffusion timestep, computes pixel-space
differences to a reference trajectory, and saves a CSV/JSON plus curve plots.

Example
-------
python DM_compare_sample_trajectories.py \
  --base-run samples/cifar10_checkpoints_horse_automobile/result_horse_seed0 \
  --target-run samples/cifar__horse_automobile__horse__seed_0__remove_top5000__endpoint_das/result_horse_seed0:endpoint_das_top5000 \
  --target-run samples/cifar__horse_automobile__horse__seed_0__remove_top5000__traj_tracin/result_horse_seed0:traj_tracin_top5000 \
  --outdir trajectory_comparisons/horse_seed0_top5000
"""

import argparse
import csv
import json
import os
import re
from typing import Dict, List, Sequence, Tuple

import numpy as np


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def sanitize_label(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "trajectory"


def resolve_run_and_sample_dir(path: str, sample_index: int) -> Tuple[str, str]:
    path = os.path.abspath(path)
    if os.path.isfile(os.path.join(path, "trajectory.npy")):
        sample_dir = path
        run_dir = os.path.dirname(path)
    else:
        run_dir = path
        sample_dir = os.path.join(run_dir, f"sample_{int(sample_index):03d}")

    if not os.path.isfile(os.path.join(sample_dir, "trajectory.npy")):
        raise FileNotFoundError(f"Missing trajectory.npy under {sample_dir}")
    if not os.path.isfile(os.path.join(run_dir, "run_info.json")):
        raise FileNotFoundError(f"Missing run_info.json under {run_dir}")
    return run_dir, sample_dir


def load_sampler_trajectory(path: str, sample_index: int = 0) -> Dict[str, object]:
    run_dir, sample_dir = resolve_run_and_sample_dir(path, sample_index)
    run_info = load_json(os.path.join(run_dir, "run_info.json"))
    traj = np.asarray(np.load(os.path.join(sample_dir, "trajectory.npy")), dtype=np.float32)
    if traj.ndim != 4:
        raise ValueError(f"Expected trajectory.npy shape (K,H,W,C), got {traj.shape} in {sample_dir}")

    saved_timesteps = [int(t) for t in run_info.get("saved_timesteps", [])]
    if len(saved_timesteps) != traj.shape[0]:
        png_ts = []
        for name in os.listdir(sample_dir):
            m = re.match(r"t_(\d+)\.png$", name)
            if m:
                png_ts.append(int(m.group(1)))
        if len(png_ts) == traj.shape[0]:
            saved_timesteps = sorted(png_ts, reverse=True)
        else:
            raise ValueError(
                f"Could not infer timesteps for {sample_dir}: "
                f"len(saved_timesteps)={len(saved_timesteps)}, trajectory length={traj.shape[0]}"
            )

    # DM___sampler.py writes trajectory.npy in sorted(saved.keys(), reverse=True) order.
    timesteps = sorted(saved_timesteps, reverse=True)
    if len(timesteps) != traj.shape[0]:
        raise ValueError(f"Timestep count {len(timesteps)} does not match trajectory length {traj.shape[0]}")

    return {
        "run_dir": run_dir,
        "sample_dir": sample_dir,
        "run_info": run_info,
        "timesteps": np.asarray(timesteps, dtype=np.int32),
        "trajectory": traj,
    }


def parse_target_run_arg(text: str) -> Tuple[str, str]:
    if ":" in text:
        path, label = text.rsplit(":", 1)
        return path, sanitize_label(label)
    path = text
    label = sanitize_label(os.path.basename(os.path.dirname(path)) or os.path.basename(path))
    return path, label


def align_by_timestep(
    base_ts: np.ndarray,
    base_traj: np.ndarray,
    target_ts: np.ndarray,
    target_traj: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_map = {int(t): i for i, t in enumerate(base_ts.tolist())}
    target_map = {int(t): i for i, t in enumerate(target_ts.tolist())}
    common = sorted(set(base_map.keys()) & set(target_map.keys()), reverse=True)
    if not common:
        raise ValueError("No common saved timesteps between base and target trajectories.")

    base_aligned = np.stack([base_traj[base_map[t]] for t in common], axis=0)
    target_aligned = np.stack([target_traj[target_map[t]] for t in common], axis=0)
    return np.asarray(common, dtype=np.int32), base_aligned, target_aligned


def compute_metrics(base: np.ndarray, target: np.ndarray) -> Dict[str, np.ndarray]:
    diff = target.astype(np.float32) - base.astype(np.float32)
    mse = np.mean(diff ** 2, axis=(1, 2, 3))
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(diff), axis=(1, 2, 3))
    linf = np.max(np.abs(diff), axis=(1, 2, 3))
    denom = np.maximum(mse, 1e-12)
    psnr = 10.0 * np.log10(1.0 / denom)
    return {
        "mse": mse.astype(np.float64),
        "rmse": rmse.astype(np.float64),
        "mae": mae.astype(np.float64),
        "linf": linf.astype(np.float64),
        "psnr": psnr.astype(np.float64),
    }


def write_metrics_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["label", "timestep", "mse", "rmse", "mae", "linf", "psnr"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_curves(out_path: str, series: Sequence[Dict[str, object]], title: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax_mse, ax_rmse = axes

    for item in series:
        label = item["label"]
        timesteps = item["timesteps"]
        metrics = item["metrics"]
        ax_mse.plot(timesteps, metrics["mse"], marker="o", markersize=2.5, linewidth=1.6, label=label)
        ax_rmse.plot(timesteps, metrics["rmse"], marker="o", markersize=2.5, linewidth=1.6, label=label)

    ax_mse.set_ylabel("MSE to original")
    ax_rmse.set_ylabel("RMSE to original")
    ax_rmse.set_xlabel("Diffusion timestep")
    ax_mse.set_title(title)
    ax_mse.grid(True, alpha=0.25)
    ax_rmse.grid(True, alpha=0.25)
    ax_mse.legend(loc="best")
    ax_mse.invert_xaxis()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare saved sampler trajectories against a base trajectory.")
    parser.add_argument("--base-run", required=True, help="Original model run dir or sample_000 dir.")
    parser.add_argument(
        "--target-run",
        action="append",
        required=True,
        help="Target run dir or sample dir. Optional label syntax: path:label",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--base-label", type=str, default="original")
    parser.add_argument("--outdir", type=str, default="./trajectory_comparisons")
    parser.add_argument("--title", type=str, default="Trajectory difference to original model")
    args = parser.parse_args()

    base = load_sampler_trajectory(args.base_run, sample_index=args.sample_index)
    base_ts = base["timesteps"]
    base_traj = base["trajectory"]

    series = []
    csv_rows: List[Dict[str, object]] = []
    summary = {
        "base_run": base["run_dir"],
        "base_sample_dir": base["sample_dir"],
        "base_label": args.base_label,
        "targets": [],
    }

    for target_arg in args.target_run:
        target_path, label = parse_target_run_arg(target_arg)
        target = load_sampler_trajectory(target_path, sample_index=args.sample_index)
        common_ts, base_aligned, target_aligned = align_by_timestep(
            base_ts=base_ts,
            base_traj=base_traj,
            target_ts=target["timesteps"],
            target_traj=target["trajectory"],
        )
        metrics = compute_metrics(base_aligned, target_aligned)
        series.append({"label": label, "timesteps": common_ts, "metrics": metrics})

        target_summary = {
            "label": label,
            "run_dir": target["run_dir"],
            "sample_dir": target["sample_dir"],
            "num_common_timesteps": int(len(common_ts)),
            "mean_mse": float(np.mean(metrics["mse"])),
            "max_mse": float(np.max(metrics["mse"])),
            "mean_rmse": float(np.mean(metrics["rmse"])),
            "max_rmse": float(np.max(metrics["rmse"])),
            "final_timestep": int(common_ts[-1]),
            "final_mse": float(metrics["mse"][-1]),
            "final_rmse": float(metrics["rmse"][-1]),
        }
        summary["targets"].append(target_summary)

        for i, t in enumerate(common_ts.tolist()):
            csv_rows.append(
                {
                    "label": label,
                    "timestep": int(t),
                    "mse": float(metrics["mse"][i]),
                    "rmse": float(metrics["rmse"][i]),
                    "mae": float(metrics["mae"][i]),
                    "linf": float(metrics["linf"][i]),
                    "psnr": float(metrics["psnr"][i]),
                }
            )

    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "trajectory_diff_metrics.csv")
    json_path = os.path.join(args.outdir, "trajectory_diff_summary.json")
    plot_path = os.path.join(args.outdir, "trajectory_diff_curves.png")

    write_metrics_csv(csv_path, csv_rows)
    save_json(json_path, summary)
    plot_curves(plot_path, series, title=args.title)

    print("=" * 88)
    print("Trajectory comparison complete")
    print(f"base_run : {base['run_dir']}")
    for target in summary["targets"]:
        print(
            f"target   : {target['label']} | "
            f"mean_mse={target['mean_mse']:.6g} | final_mse={target['final_mse']:.6g}"
        )
    print(f"csv      : {csv_path}")
    print(f"summary  : {json_path}")
    print(f"plot     : {plot_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
