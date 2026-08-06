from __future__ import annotations

"""
Run LDS evaluation for CIFAR diffusion attribution scores.

For each random subset S_j, this script:
  1. retrains a CIFAR model on S_j,
  2. computes the true counterfactual output f(theta*(S_j)),
  3. predicts that output with sum_i_in_S_j tau_i,
  4. reports Spearman(predicted, true) as LDS.

The default target function is the trajectory-noise objective:

  f_noise(theta; x_ref) =
      mean_k w_k || eps_theta(x_ref_k, t_k) - eps_theta_ref(x_ref_k, t_k) ||^2

Example
-------
CUDA_VISIBLE_DEVICES=0 python3 LDS/DM_cifar_lds.py \
  --score-file "\
attribution_results/traj_tracein/cifar2_traj_attr_cifar10_horse_automobile_from_sample_range_1_2000,\
attribution_results/traj_tracein/cifar2_traj_attr_cifar10_horse_automobile_from_sample_range_2001_4000,\
attribution_results/traj_tracein/cifar2_traj_attr_cifar10_horse_automobile_from_sample_range_4001_6000,\
attribution_results/traj_tracein/cifar2_traj_attr_cifar10_horse_automobile_from_sample_range_6001_8000,\
attribution_results/traj_tracein/cifar2_traj_attr_cifar10_horse_automobile_from_sample_range_8001_10000" \
  --base-checkpoint "models/cifar10_checkpoints_horse_automobile/seed_0_epoch_0200.ckpt" \
  --class-names "horse,automobile" \
  --subset-size 5000 \
  --m 10 \
  --subset-seed 0 \
  --prompt "horse" \
  --target-function noise_trajectory \
  --trajectory-reduction sum \
  --prediction-sign -1 \
  --run-tag traj_sum_horse \
  --epochs 200 \
  --prefer-device gpu \
  --num-devices 1 \
  --no-use-data-parallel \
  --log-every 10 \
  --progress-bar \
  --save-every-epochs 1 \
  --keep-last-k 0

CUDA_VISIBLE_DEVICES=0 python3 LDS/DM_cifar_lds.py \
  --score-file "\
attribution_results/traj_tracein/cifar2_traj_attr_cifar10_horse_automobile_from_samplee_model_horse_automobile__ckpt_seed_0_epoch_0200_horse_automobile_range_1_2500,\
attribution_results/traj_tracein/cifar2_traj_attr_cifar10_horse_automobile_from_samplee_model_horse_automobile__ckpt_seed_0_epoch_0200_horse_automobile_range_2501_5000,\
attribution_results/traj_tracein/cifar2_traj_attr_cifar10_horse_automobile_from_samplee_model_horse_automobile__ckpt_seed_0_epoch_0200_horse_automobile_range_5001_7500,\
attribution_results/traj_tracein/cifar2_traj_attr_cifar10_horse_automobile_from_samplee_model_horse_automobile__ckpt_seed_0_epoch_0200_horse_automobile_range_7501_10000" \
  --base-checkpoint "models/cifar10_checkpoints_horse_automobile/seed_0_epoch_0200.ckpt" \
  --class-names "horse,automobile" \
  --subset-size 5000 \
  --m 100 \
  --subset-seed 0 \
  --prompt "horse,automobile" \
  --target-function noise_trajectory \
  --trajectory-reduction sum \
  --prediction-sign -1 \
  --out-root ./LDS/runs \
  --run-name traj_sum_m100_horse_automobile_seed0_model_0200 \
  --epochs 200 \
  --prefer-device gpu \
  --num-devices 1 \
  --no-use-data-parallel \
  --log-every 10 \
  --progress-bar \
  --save-every-epochs 200 \
  --keep-last-k 1
  
CUDA_VISIBLE_DEVICES=1 python3 LDS/DM_cifar_lds.py \
  --score-file "\
attribution_results/traj_tracein/cifar2_traj_attr_cifar10_horse_automobile_from_samplee_model_horse_automobile__ckpt_seed_0_epoch_0200_horse_automobile_range_1_2500,\
attribution_results/traj_tracein/cifar2_traj_attr_cifar10_horse_automobile_from_samplee_model_horse_automobile__ckpt_seed_0_epoch_0200_horse_automobile_range_2501_5000,\
attribution_results/traj_tracein/cifar2_traj_attr_cifar10_horse_automobile_from_samplee_model_horse_automobile__ckpt_seed_0_epoch_0200_horse_automobile_range_5001_7500,\
attribution_results/traj_tracein/cifar2_traj_attr_cifar10_horse_automobile_from_samplee_model_horse_automobile__ckpt_seed_0_epoch_0200_horse_automobile_range_7501_10000" \
  --base-checkpoint "models/cifar10_checkpoints_horse_automobile/seed_0_epoch_0200.ckpt" \
  --class-names "horse,automobile" \
  --subset-size 5000 \
  --m 100 \
  --subset-seed 0 \
  --prompt "horse,automobile" \
  --target-function simple_loss \
  --simple-loss-num-mc 16 \
  --simple-loss-mc-seed 0 \
  --prediction-sign -1 \
  --out-root ./LDS/runs \
  --run-name traj_simple_mc16_seed0_m100_horse_automobile_model_0200 \
  --epochs 200 \
  --prefer-device gpu \
  --num-devices 1 \
  --no-use-data-parallel \
  --log-every 10 \
  --progress-bar \
  --save-every-epochs 200 \
  --keep-last-k 1  
  
  
CUDA_VISIBLE_DEVICES=2 python3 LDS/DM_cifar_lds.py \
  --score-file "attribution_results/endpoint_das/cifar2_endpoint_das_horse_automobile_from_sample_model_horse_automobile__ckpt_seed_0_epoch_0200_horse_automobile_range_1_10000" \
  --base-checkpoint "models/cifar10_checkpoints_horse_automobile/seed_0_epoch_0200.ckpt" \
  --class-names "horse,automobile" \
  --subset-size 5000 \
  --m 100 \
  --subset-seed 0 \
  --prompt "horse,automobile" \
  --target-function noise_trajectory \
  --trajectory-reduction sum \
  --prediction-sign -1 \
  --out-root ./LDS/runs \
  --run-name das_traj_sum_m100_horse_automobile_seed0_model_0200 \
  --epochs 200 \
  --prefer-device gpu \
  --num-devices 1 \
  --no-use-data-parallel \
  --log-every 10 \
  --progress-bar \
  --save-every-epochs 200 \
  --keep-last-k 1

CUDA_VISIBLE_DEVICES=3 python3 LDS/DM_cifar_lds.py \
  --score-file "attribution_results/endpoint_das/cifar2_endpoint_das_horse_automobile_from_sample_model_horse_automobile__ckpt_seed_0_epoch_0200_horse_automobile_range_1_10000" \
  --base-checkpoint "models/cifar10_checkpoints_horse_automobile/seed_0_epoch_0200.ckpt" \
  --class-names "horse,automobile" \
  --subset-size 5000 \
  --m 100 \
  --subset-seed 0 \
  --prompt "horse,automobile" \
  --target-function simple_loss \
  --simple-loss-num-mc 16 \
  --simple-loss-mc-seed 0 \
  --prediction-sign -1 \
  --out-root ./LDS/runs \
  --run-name das_simple_mc16_seed0_m100_horse_automobile_model_0200 \
  --epochs 200 \
  --prefer-device gpu \
  --num-devices 1 \
  --no-use-data-parallel \
  --log-every 10 \
  --progress-bar \
  --save-every-epochs 200 \
  --keep-last-k 1

"""
import argparse
import contextlib
import csv
import json
import math
import os
import pickle
import re
import sys
import time
from dataclasses import asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(THIS_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import jax
import jax.numpy as jnp

from DM__training_CIFAR10_pixel import TrainConfig, train
from DM___sampler import CIFARAdapter
from DM_counterfactual_retrain_from_attribution import (
    build_filtered_index_to_cifar_row_map,
    combine_attribution_scores,
    load_base_config,
    parse_class_names,
    selected_indices_to_exclude_indices,
)


def sanitize_tag(text: Optional[str], default: str = "unknown") -> str:
    if text is None or str(text).strip() == "":
        return default
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or default


def resolve_path(path: Optional[str], *, must_exist: bool = False) -> Optional[str]:
    if path is None:
        return None
    expanded = os.path.expanduser(path)
    candidates = [expanded]
    if not os.path.isabs(expanded):
        candidates.append(os.path.join(PROJECT_DIR, expanded))
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if os.path.exists(candidate):
            return candidate
    resolved = os.path.abspath(candidates[-1] if len(candidates) > 1 else expanded)
    if must_exist:
        raise FileNotFoundError(f"Path not found: {path} (also tried relative to {PROJECT_DIR})")
    return resolved


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        "subset_id",
        "subset_seed",
        "subset_size",
        "prediction_subset",
        "prediction_sign",
        "pred_sum_tau",
        "true_f",
        "checkpoint",
        "subset_dir",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def rank_average_ties(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j - 1)
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def spearman_corr(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"Spearman inputs must have same shape, got {x.shape} and {y.shape}")
    if x.size < 2:
        return float("nan")
    rx = rank_average_ties(x)
    ry = rank_average_ties(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt(np.sum(rx ** 2) * np.sum(ry ** 2)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(rx * ry) / denom)


def resolve_score_inputs(score_file: str) -> List[str]:
    if "," in score_file and not os.path.exists(score_file):
        return [resolve_path(p, must_exist=True) for p in (x.strip() for x in score_file.split(",")) if p]
    return [resolve_path(score_file, must_exist=True)]


def parse_int_csv(text: Optional[str], *, name: str) -> Optional[List[int]]:
    if text is None:
        return None
    values = []
    for raw in str(text).split(","):
        raw = raw.strip()
        if not raw:
            continue
        values.append(int(raw))
    if not values:
        raise ValueError(f"{name} must contain at least one integer.")
    return values


def infer_score_metadata(score_inputs: Sequence[str]) -> Dict[str, object]:
    meta: Dict[str, object] = {"score_inputs": [os.path.abspath(p) for p in score_inputs]}
    first = score_inputs[0]
    if os.path.isdir(first):
        for name in ("traj_attr_result.json", "run_config.json", "run_info.json"):
            path = os.path.join(first, name)
            if os.path.isfile(path):
                try:
                    payload = load_json(path)
                except Exception:
                    continue
                if name == "traj_attr_result.json":
                    cfg = payload.get("config", {})
                    meta["score_run_config"] = cfg
                    meta["query_objective"] = payload.get("query_objective")
                    if payload.get("query_objective") is None:
                        saved_projection = payload.get("trajectory_projection", {})
                        meta["trajectory_projection"] = {
                            "seed": int(cfg.get("seed", 0)),
                            "m_proj": int(cfg.get("m_proj", 1)),
                            "num_checkpoint_projection_sets": max(
                                1, len(payload.get("used_checkpoints", []))
                            ),
                            "snapshot_timesteps": payload.get("snapshot_timesteps_used"),
                            "projection_seed_formula": (
                                "seed + 1000 + 100000 * checkpoint_index + snapshot_index"
                            ),
                            **saved_projection,
                        }
                else:
                    meta[name[:-5]] = payload
    return meta


def infer_attribution_sample_dir(score_inputs: Sequence[str]) -> Optional[str]:
    first = score_inputs[0]
    candidates = []
    if os.path.isdir(first):
        candidates.extend(
            [
                os.path.join(first, "traj_attr_result.json"),
                os.path.join(first, "run_config.json"),
                os.path.join(first, "run_info.json"),
            ]
        )
    for path in candidates:
        if not os.path.isfile(path):
            continue
        payload = load_json(path)
        cfg = payload.get("config", payload)
        value = cfg.get("attribution_sample_dir")
        if value:
            return value
        sample_meta = payload.get("attribution_sample_meta", {})
        seed_info = sample_meta.get("seed_info", {})
        manifest = sample_meta.get("manifest", {})
        if manifest.get("checkpoint"):
            meta_dir = sample_meta.get("seed_dir")
            if meta_dir:
                return os.path.dirname(meta_dir)
        if seed_info:
            meta_dir = sample_meta.get("seed_dir")
            if meta_dir:
                return os.path.dirname(meta_dir)
    return None


def infer_prompt(score_inputs: Sequence[str]) -> Optional[str]:
    first = score_inputs[0]
    candidates = []
    if os.path.isdir(first):
        candidates.extend(
            [
                os.path.join(first, "traj_attr_result.json"),
                os.path.join(first, "run_config.json"),
                os.path.join(first, "run_info.json"),
            ]
        )
    for path in candidates:
        if not os.path.isfile(path):
            continue
        payload = load_json(path)
        cfg = payload.get("config", payload)
        for key in ("query", "prompt"):
            if cfg.get(key):
                return str(cfg[key])
        sample_meta = payload.get("attribution_sample_meta", {})
        seed_info = sample_meta.get("seed_info", {})
        manifest = sample_meta.get("manifest", {})
        for source in (seed_info, manifest):
            if source.get("prompt"):
                return str(source["prompt"])
    return None


def find_seed_dir(sample_root: str, sample_seed: Optional[int]) -> str:
    sample_root = os.path.abspath(sample_root)
    if os.path.isfile(os.path.join(sample_root, "trajectory_xt.npy")):
        return sample_root
    if sample_seed is not None:
        candidate = os.path.join(sample_root, f"seed_{int(sample_seed):06d}")
        if os.path.isfile(os.path.join(candidate, "trajectory_xt.npy")):
            return candidate
        raise FileNotFoundError(f"Missing trajectory_xt.npy under {candidate}")
    candidates = []
    for name in sorted(os.listdir(sample_root)):
        path = os.path.join(sample_root, name)
        if name.startswith("seed_") and os.path.isfile(os.path.join(path, "trajectory_xt.npy")):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No seed_* directory with trajectory_xt.npy under {sample_root}")
    return candidates[0]


def load_trajectory_target(
    sample_root: str,
    sample_seed: Optional[int],
    sample_index: int,
    max_steps: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    seed_dir = find_seed_dir(sample_root, sample_seed)
    trajectory_path = os.path.join(seed_dir, "trajectory_xt.npy")
    t_path = os.path.join(seed_dir, "trajectory_t.npy")
    pos_path = os.path.join(seed_dir, "trajectory_pos.npy")
    trajectory = np.asarray(np.load(trajectory_path), dtype=np.float32)
    if trajectory.ndim != 5:
        raise ValueError(f"Expected trajectory_xt.npy shape (K,B,H,W,C), got {trajectory.shape}")
    if sample_index < 0 or sample_index >= trajectory.shape[1]:
        raise IndexError(f"sample_index={sample_index} out of range for batch size {trajectory.shape[1]}")
    if not os.path.isfile(t_path):
        raise FileNotFoundError(f"Missing trajectory_t.npy: {t_path}")
    t_seq = np.asarray(np.load(t_path), dtype=np.int32).reshape(-1)
    if len(t_seq) != trajectory.shape[0]:
        raise ValueError(f"trajectory_t length {len(t_seq)} does not match trajectory length {trajectory.shape[0]}")

    if max_steps is not None and int(max_steps) > 0 and int(max_steps) < len(t_seq):
        pick = np.linspace(0, len(t_seq) - 1, int(max_steps), dtype=np.int32)
        trajectory = trajectory[pick]
        t_seq = t_seq[pick]
    xt = trajectory[:, sample_index : sample_index + 1]

    meta = {
        "seed_dir": seed_dir,
        "trajectory_xt_path": trajectory_path,
        "trajectory_t_path": t_path,
        "trajectory_pos_path": pos_path if os.path.isfile(pos_path) else None,
        "sample_index": int(sample_index),
        "trajectory_shape_loaded": list(xt.shape),
        "timesteps": [int(t) for t in t_seq.tolist()],
        "seed_info": load_json(os.path.join(seed_dir, "seed_info.json"))
        if os.path.isfile(os.path.join(seed_dir, "seed_info.json"))
        else {},
        "manifest": load_json(os.path.join(os.path.dirname(seed_dir), "manifest.json"))
        if os.path.isfile(os.path.join(os.path.dirname(seed_dir), "manifest.json"))
        else {},
    }
    return xt, t_seq, meta


def latest_checkpoint(checkpoint_dir: str) -> str:
    paths = []
    for name in os.listdir(checkpoint_dir):
        if not name.endswith(".ckpt") or "_epoch_" not in name:
            continue
        try:
            ep = int(name.split("_epoch_")[-1].split(".ckpt")[0])
        except Exception:
            continue
        paths.append((ep, os.path.join(checkpoint_dir, name)))
    if not paths:
        raise FileNotFoundError(f"No *_epoch_*.ckpt found in {checkpoint_dir}")
    paths.sort(key=lambda item: item[0])
    return paths[-1][1]


def build_score_vector(indices: np.ndarray, scores: np.ndarray) -> Dict[int, float]:
    return {int(i): float(s) for i, s in zip(indices.tolist(), scores.tolist())}


def make_subset_indices(
    rng: np.random.Generator,
    universe: np.ndarray,
    subset_size: int,
) -> np.ndarray:
    if subset_size <= 0:
        raise ValueError(f"subset_size must be positive, got {subset_size}")
    if subset_size > len(universe):
        raise ValueError(f"subset_size={subset_size} exceeds universe size {len(universe)}")
    return np.asarray(rng.choice(universe, size=subset_size, replace=False), dtype=np.int64)


def sum_scores(indices: Iterable[int], score_map: Dict[int, float], sign: float) -> float:
    return float(sign * sum(score_map.get(int(i), 0.0) for i in indices))


def plot_scatter(
    path: str,
    pred: np.ndarray,
    true: np.ndarray,
    title: str,
    xlabel: str = "Predicted sum of attribution scores",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warning] matplotlib unavailable; skipping scatter plot ({exc})")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(pred, true, s=34, alpha=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("True counterfactual f")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


class CompactTrainLogWriter:
    def __init__(self, log_f, terminal, prefix: str, progress_bar: bool):
        self.log_f = log_f
        self.terminal = terminal
        self.prefix = prefix
        self._buffer = ""
        self.progress_bar = bool(progress_bar)
        self._pbar = None
        self._last_step = 0
        self._tqdm = None
        if self.progress_bar:
            try:
                from tqdm.auto import tqdm
                self._tqdm = tqdm
            except Exception:
                self.progress_bar = False

    def write(self, text: str) -> int:
        self.log_f.write(text)
        self.log_f.flush()
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._maybe_emit(line)
        return len(text)

    def flush(self) -> None:
        self.log_f.flush()
        if self._buffer:
            self._maybe_emit(self._buffer)
            self._buffer = ""
        if self._pbar is not None:
            self._pbar.refresh()

    def close(self) -> None:
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None

    def _maybe_emit(self, line: str) -> None:
        if self._handle_progress_line(line):
            return
        keep_markers = (
            "Loaded ",
            "steps_per_epoch=",
            "[epoch ",
            "Saved checkpoint",
            "Training finished.",
            "Traceback",
            "Error",
            "ERROR",
        )
        if any(marker in line for marker in keep_markers):
            print(f"{self.prefix}{line}", file=self.terminal, flush=True)

    def _handle_progress_line(self, line: str) -> bool:
        match = re.search(r"epoch=(\d+)/(\d+)\s+step=(\d+)/(\d+)\s+loss=([0-9.eE+-]+)", line)
        if match is None:
            return False

        if not self.progress_bar or self._tqdm is None:
            print(f"{self.prefix}{line}", file=self.terminal, flush=True)
            return True

        epoch, total_epochs, step, total_steps, loss = match.groups()
        step_i = int(step)
        total_i = int(total_steps)
        if self._pbar is None:
            self._pbar = self._tqdm(
                total=total_i,
                initial=0,
                desc=self.prefix.strip(),
                dynamic_ncols=True,
                file=self.terminal,
                leave=True,
            )
        if total_i != self._pbar.total:
            self._pbar.total = total_i
        delta = max(0, step_i - self._last_step)
        if delta:
            self._pbar.update(delta)
            self._last_step = step_i
        self._pbar.set_postfix(epoch=f"{epoch}/{total_epochs}", loss=loss)
        return True


def run_train_with_optional_logging(
    cfg: TrainConfig,
    log_path: str,
    quiet: bool,
    prefix: str,
    progress_bar: bool,
) -> None:
    if not quiet:
        train(cfg)
        return
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as log_f:
        compact_writer = CompactTrainLogWriter(
            log_f,
            sys.__stdout__,
            prefix=prefix,
            progress_bar=progress_bar,
        )
        try:
            with contextlib.redirect_stdout(compact_writer), contextlib.redirect_stderr(compact_writer):
                train(cfg)
        finally:
            compact_writer.close()


class CifarTargetEvaluator:
    def __init__(
        self,
        code_file: str,
        base_checkpoint: str,
        prompt: str,
        prefer_device: str,
        data_root: Optional[str],
        target_function: str,
        sample_root: Optional[str],
        sample_seed: Optional[int],
        sample_index: int,
        max_trajectory_steps: Optional[int],
        trajectory_reduction: str,
        trajectory_projection: Optional[Dict[str, object]],
        simple_loss_timesteps: Sequence[int],
        simple_loss_noise_seeds: Optional[Sequence[int]],
        simple_loss_num_mc: int,
        simple_loss_mc_seed: int,
    ):
        self.code_file = code_file
        self.base_checkpoint = base_checkpoint
        self.prompt = prompt
        self.prefer_device = prefer_device
        self.data_root = data_root
        self.target_function = target_function
        self.sample_root = sample_root
        self.sample_seed = sample_seed
        self.sample_index = sample_index
        self.max_trajectory_steps = max_trajectory_steps
        self.trajectory_reduction = trajectory_reduction
        self.trajectory_projection = trajectory_projection
        self.simple_loss_timesteps = [int(t) for t in simple_loss_timesteps]
        self.simple_loss_noise_seeds = None if simple_loss_noise_seeds is None else [int(s) for s in simple_loss_noise_seeds]
        self.simple_loss_num_mc = int(simple_loss_num_mc)
        self.simple_loss_mc_seed = int(simple_loss_mc_seed)
        if not self.simple_loss_timesteps:
            raise ValueError("simple_loss_timesteps must contain at least one timestep.")
        if self.simple_loss_num_mc <= 0:
            raise ValueError("simple_loss_num_mc must be positive.")

        self.base_adapter = CIFARAdapter(
            code_file=code_file,
            checkpoint=base_checkpoint,
            prefer_device=prefer_device,
            cifar_data_root=data_root,
        )
        self.base_adapter.setup()
        self.device = self.base_adapter.device
        self.cond = self.base_adapter.make_condition(prompt=prompt, batch_size=1)

        self.xt_ref = None
        self.t_seq = None
        self.target_meta: Dict[str, object] = {}
        if target_function in ("noise_trajectory", "projected_trajectory", "simple_loss", "trajectory_state_mse"):
            if sample_root is None:
                raise ValueError("--attribution-sample-dir is required for target function evaluation.")
            self.xt_ref, self.t_seq, self.target_meta = load_trajectory_target(
                sample_root=sample_root,
                sample_seed=sample_seed,
                sample_index=sample_index,
                max_steps=max_trajectory_steps if target_function != "simple_loss" else None,
            )
            if target_function == "simple_loss":
                self.xt_ref = self.xt_ref[-1:]
                self.t_seq = self.t_seq[-1:]
            self.xt_ref = jax.device_put(jnp.asarray(self.xt_ref, dtype=jnp.float32), self.device)
            self.t_seq = np.asarray(self.t_seq, dtype=np.int32)

        self._noise_fn = jax.jit(self._noise_objective)
        self._projected_fn = jax.jit(self._projected_trajectory_objective)
        self._simple_loss_fn = jax.jit(self._simple_loss_objective)

        if target_function == "projected_trajectory":
            if not trajectory_projection:
                raise ValueError(
                    "projected_trajectory requires Traj TracIn metadata from "
                    "traj_attr_result.json in --score-file."
                )
            expected_timesteps = trajectory_projection.get("snapshot_timesteps")
            if expected_timesteps is not None:
                expected = np.asarray(expected_timesteps, dtype=np.int32)
                if not np.array_equal(expected, self.t_seq):
                    raise ValueError(
                        "Saved LDS trajectory timesteps do not match the Traj TracIn "
                        f"attribution timesteps: LDS={self.t_seq.tolist()} versus "
                        f"attribution={expected.tolist()}."
                    )

    def _make_adapter(self, checkpoint: str) -> CIFARAdapter:
        adapter = CIFARAdapter(
            code_file=self.code_file,
            checkpoint=checkpoint,
            prefer_device=self.prefer_device,
            cifar_data_root=self.data_root,
        )
        adapter.setup()
        return adapter

    def _eps_apply(self, adapter: CIFARAdapter, params, x, t):
        cond = self.cond
        return adapter.model.apply({"params": params}, x, t, cond, train=False)

    def _noise_objective(self, target_params, base_params, xt_refs, t_seq):
        def one_step(x, t_scalar):
            t = jnp.full((x.shape[0],), t_scalar, dtype=jnp.int32)
            eps_target = self._eps_apply(self._target_adapter_for_jit, target_params, x, t)
            eps_base = self._eps_apply(self.base_adapter, base_params, x, t)
            sq = (eps_target - eps_base) ** 2
            if self.trajectory_reduction in ("sum", "snapshot_mean"):
                return jnp.sum(sq)
            return jnp.mean(sq)

        vals = jax.vmap(one_step)(xt_refs, t_seq)
        if self.trajectory_reduction == "sum":
            return jnp.sum(vals)
        return jnp.mean(vals)

    def _projected_trajectory_objective(self, target_params, base_params, xt_refs, t_seq):
        """F(target) - F(base), using the exact projections from Traj TracIn."""
        projection_seed = int(self.trajectory_projection["seed"])
        m_proj = int(self.trajectory_projection["m_proj"])
        num_checkpoint_sets = int(
            self.trajectory_projection["num_checkpoint_projection_sets"]
        )

        def one_step(snapshot_index, x, t_scalar):
            t = jnp.full((x.shape[0],), t_scalar, dtype=jnp.int32)
            eps_target = self._eps_apply(self._target_adapter_for_jit, target_params, x, t)
            eps_base = self._eps_apply(self.base_adapter, base_params, x, t)
            delta = eps_target - eps_base
            projected = jnp.array(0.0, dtype=delta.dtype)
            for checkpoint_index in range(num_checkpoint_sets):
                key = jax.random.PRNGKey(
                    projection_seed
                    + 1000
                    + 100000 * checkpoint_index
                    + snapshot_index
                )
                projection_keys = jax.random.split(key, m_proj)
                checkpoint_projection = jnp.array(0.0, dtype=delta.dtype)
                for projection_key in projection_keys:
                    bits = jax.random.randint(
                        projection_key, delta.shape, 0, 2, dtype=jnp.int32
                    )
                    rademacher = (bits * 2 - 1).astype(delta.dtype)
                    checkpoint_projection = checkpoint_projection + jnp.sum(
                        delta * rademacher
                    )
                projected = projected + checkpoint_projection / float(m_proj)
            return projected

        snapshot_indices = jnp.arange(xt_refs.shape[0], dtype=jnp.int32)
        vals = jax.vmap(one_step)(snapshot_indices, xt_refs, t_seq)
        # Traj TracIn always applies 1/K over snapshots and sums checkpoints.
        return jnp.mean(vals)

    def _simple_loss_objective(self, target_params, x0_ref, t_values, rng_keys):
        x0 = x0_ref[0]
        schedule = self.base_adapter.schedule

        def one_loss(t_scalar, rng):
            t = jnp.full((x0.shape[0],), t_scalar, dtype=jnp.int32)
            noise = jax.random.normal(rng, x0.shape, dtype=x0.dtype)
            xt = (
                self.base_adapter.module.extract(schedule.sqrt_alphas_cumprod, t, x0.shape) * x0
                + self.base_adapter.module.extract(schedule.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
            )
            pred = self._eps_apply(self._target_adapter_for_jit, target_params, xt, t)
            return jnp.mean((pred - noise) ** 2)

        losses = jax.vmap(one_loss)(t_values, rng_keys)
        return jnp.mean(losses)

    def _sample_model_space_trajectory(self, adapter: CIFARAdapter, seed: int, timesteps_to_save: Sequence[int]) -> np.ndarray:
        """Generate model-space x_t snapshots for the target checkpoint without saving them."""
        rng = jax.random.PRNGKey(int(seed))
        shape = adapter.sample_shape(batch_size=1)
        cond_y = self.cond if getattr(adapter.cfg, "class_cond", True) else None
        betas = adapter.schedule.betas
        alphas = adapter.schedule.alphas
        alphas_cumprod = adapter.schedule.alphas_cumprod
        t_seq = jnp.arange(adapter.cfg.timesteps - 1, -1, -1, dtype=jnp.int32)

        @jax.jit
        def sample_scan(init_rng):
            init_x = jax.random.normal(init_rng, shape, dtype=jnp.float32)

            def body_fn(carry, i):
                x, loop_rng = carry
                t = jnp.full((shape[0],), i, dtype=jnp.int32)
                pred = adapter.model.apply(
                    {"params": adapter.state.ema_params},
                    x,
                    t,
                    cond_y,
                    train=False,
                )
                x0_pred = pred if adapter.cfg.predict_x0 else adapter.predict_x0_from_eps(x, t, pred)
                eps = pred if not adapter.cfg.predict_x0 else (
                    x - jnp.sqrt(alphas_cumprod[i]) * x0_pred
                ) / jnp.sqrt(1.0 - alphas_cumprod[i])

                alpha_t = alphas[i]
                abar_t = alphas_cumprod[i]
                beta_t = betas[i]
                coef1 = 1.0 / jnp.sqrt(alpha_t)
                coef2 = beta_t / jnp.sqrt(1.0 - abar_t)
                mean = coef1 * (x - coef2 * eps)

                loop_rng, step_rng = jax.random.split(loop_rng)
                noise = jax.random.normal(step_rng, shape, dtype=x.dtype)
                next_x = jax.lax.cond(
                    i > 0,
                    lambda _: mean + jnp.sqrt(beta_t) * noise,
                    lambda _: mean,
                    operand=None,
                )
                return (next_x, loop_rng), x

            (_final_x, _), xt_seq = jax.lax.scan(body_fn, (init_x, init_rng), t_seq)
            return xt_seq

        xt_seq_np = np.asarray(sample_scan(rng), dtype=np.float32)
        saved = []
        for timestep in timesteps_to_save:
            t_value = int(timestep)
            if t_value < 0 or t_value >= int(adapter.cfg.timesteps):
                raise ValueError(f"Cannot save timestep {t_value}; valid range is [0, {int(adapter.cfg.timesteps) - 1}]")
            seq_idx = int(adapter.cfg.timesteps) - 1 - t_value
            saved.append(xt_seq_np[seq_idx])
        return np.stack(saved, axis=0).astype(np.float32)

    def _trajectory_state_mse(self, target_adapter: CIFARAdapter) -> Tuple[float, Dict[str, object]]:
        if self.sample_seed is None:
            raise ValueError("trajectory_state_mse requires --attribution-sample-seed/seed metadata.")
        assert self.xt_ref is not None
        assert self.t_seq is not None
        target_xt = self._sample_model_space_trajectory(
            target_adapter,
            seed=int(self.sample_seed),
            timesteps_to_save=self.t_seq,
        )
        ref_xt = np.asarray(jax.device_get(self.xt_ref), dtype=np.float32)
        if target_xt.shape != ref_xt.shape:
            raise ValueError(f"Generated trajectory shape {target_xt.shape} does not match reference {ref_xt.shape}")
        sq = (target_xt.astype(np.float64) - ref_xt.astype(np.float64)) ** 2
        per_snapshot_mean = np.mean(sq, axis=tuple(range(1, sq.ndim)))
        per_snapshot_sum = np.sum(sq, axis=tuple(range(1, sq.ndim)))
        if self.trajectory_reduction == "sum":
            value = float(np.sum(per_snapshot_sum))
        elif self.trajectory_reduction == "snapshot_mean":
            value = float(np.mean(per_snapshot_sum))
        else:
            value = float(np.mean(per_snapshot_mean))
        details = {
            "target_function": "trajectory_state_mse",
            "num_trajectory_steps": int(len(self.t_seq)),
            "trajectory_reduction": self.trajectory_reduction,
            "definition": "Generate target-checkpoint trajectory from the same seed and average x_t MSE to the saved reference trajectory.",
            "sample_seed": int(self.sample_seed),
            "per_snapshot_mean_min": float(np.min(per_snapshot_mean)),
            "per_snapshot_mean_mean": float(np.mean(per_snapshot_mean)),
            "per_snapshot_mean_max": float(np.max(per_snapshot_mean)),
            "per_snapshot_sum_min": float(np.min(per_snapshot_sum)),
            "per_snapshot_sum_mean": float(np.mean(per_snapshot_sum)),
            "per_snapshot_sum_max": float(np.max(per_snapshot_sum)),
        }
        return value, details

    def evaluate(self, checkpoint: str) -> Tuple[float, Dict[str, object]]:
        target_adapter = self._make_adapter(checkpoint)
        # Store the target adapter for the jitted closures. The module/model structure is
        # identical across checkpoints; params are the dynamic inputs.
        self._target_adapter_for_jit = target_adapter
        target_params = target_adapter.state.ema_params
        base_params = self.base_adapter.state.ema_params

        if self.target_function == "noise_trajectory":
            value = self._noise_fn(
                target_params,
                base_params,
                self.xt_ref,
                jax.device_put(jnp.asarray(self.t_seq, dtype=jnp.int32), self.device),
            )
            details = {
                "target_function": "noise_trajectory",
                "num_trajectory_steps": int(len(self.t_seq)),
                "trajectory_reduction": self.trajectory_reduction,
            }
        elif self.target_function == "projected_trajectory":
            value = self._projected_fn(
                target_params,
                base_params,
                self.xt_ref,
                jax.device_put(jnp.asarray(self.t_seq, dtype=jnp.int32), self.device),
            )
            details = {
                "target_function": "projected_trajectory",
                "num_trajectory_steps": int(len(self.t_seq)),
                "trajectory_reduction": "mean_over_snapshots",
                "trajectory_projection": self.trajectory_projection,
                "definition": "F(subset_checkpoint) - F(full_checkpoint)",
            }
        elif self.target_function == "simple_loss":
            rng = np.random.default_rng(self.simple_loss_mc_seed)
            if self.simple_loss_noise_seeds is None:
                t_values_np = rng.choice(
                    np.asarray(self.simple_loss_timesteps, dtype=np.int32),
                    size=self.simple_loss_num_mc,
                    replace=True,
                )
                key_seeds = rng.integers(0, np.iinfo(np.int32).max, size=self.simple_loss_num_mc, dtype=np.int32)
            else:
                t_values_np = []
                key_seeds = []
                for t_value in self.simple_loss_timesteps:
                    for seed in self.simple_loss_noise_seeds:
                        t_values_np.append(int(t_value))
                        key_seeds.append(int(seed))
                t_values_np = np.asarray(t_values_np, dtype=np.int32)
                key_seeds = np.asarray(key_seeds, dtype=np.int32)
            rng_keys = [jax.random.PRNGKey(int(seed)) for seed in key_seeds.tolist()]
            t_values = jax.device_put(jnp.asarray(t_values_np, dtype=jnp.int32), self.device)
            rng_keys = jax.device_put(jnp.stack(rng_keys, axis=0), self.device)
            value = self._simple_loss_fn(target_params, self.xt_ref, t_values, rng_keys)
            details = {
                "target_function": "simple_loss",
                "simple_loss_timestep_candidates": [int(t) for t in self.simple_loss_timesteps],
                "simple_loss_sampled_timesteps": [int(t) for t in t_values_np.tolist()],
                "simple_loss_noise_key_seeds": [int(s) for s in key_seeds.tolist()],
                "simple_loss_num_mc_terms": int(len(t_values_np)),
                "simple_loss_mc_seed": int(self.simple_loss_mc_seed),
                "simple_loss_manual_grid": self.simple_loss_noise_seeds is not None,
            }
        elif self.target_function == "trajectory_state_mse":
            value_float, details = self._trajectory_state_mse(target_adapter)
            return value_float, details
        else:
            raise ValueError(f"Unknown target_function={self.target_function!r}")

        return float(value), details


def main():
    parser = argparse.ArgumentParser(description="Compute LDS for CIFAR attribution scores.")
    parser.add_argument(
        "--score-file",
        required=True,
        help="Attribution result directory, or comma-separated result directories, containing scores.npy and indices.",
    )
    parser.add_argument("--base-checkpoint", required=True, help="Original/reference CIFAR checkpoint.")
    parser.add_argument("--code-file", default="DM__training_CIFAR10_pixel.py")
    parser.add_argument("--subset-size", type=int, required=True, help="Number of scored examples kept in each S_j.")
    parser.add_argument("--m", type=int, required=True, help="Number of random subsets/counterfactual models.")
    parser.add_argument("--subset-seed", type=int, default=0)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--attribution-sample-dir", type=str, default=None)
    parser.add_argument("--attribution-sample-seed", type=int, default=None)
    parser.add_argument("--attribution-sample-index", type=int, default=None)
    parser.add_argument("--max-trajectory-steps", type=int, default=None)
    parser.add_argument(
        "--target-function",
        choices=["noise_trajectory", "projected_trajectory", "simple_loss", "trajectory_state_mse"],
        default="noise_trajectory",
    )
    parser.add_argument(
        "--trajectory-reduction",
        choices=["mean", "sum", "snapshot_mean"],
        default="mean",
        help=(
            "How to reduce f_noise. snapshot_mean computes mean_k sum_pixels, "
            "matching uniform trajectory weights w_k=1/K."
        ),
    )
    parser.add_argument("--simple-loss-timestep", type=int, default=0)
    parser.add_argument("--simple-loss-noise-seed", type=int, default=0)
    parser.add_argument(
        "--simple-loss-timesteps",
        type=str,
        default=None,
        help="Optional comma-separated candidate timesteps for simple-loss MC, e.g. 100,500,900. Defaults to all diffusion timesteps.",
    )
    parser.add_argument(
        "--simple-loss-noise-seeds",
        type=str,
        default=None,
        help="Optional comma-separated noise seeds. When set, uses the full timestep x seed grid for debugging.",
    )
    parser.add_argument("--simple-loss-num-mc", type=int, default=16)
    parser.add_argument("--simple-loss-mc-seed", type=int, default=0)
    parser.add_argument(
        "--subset-universe",
        choices=["score", "all_filtered"],
        default="score",
        help="score: sample/train only among scored indices. all_filtered: sample from full class-filtered CIFAR set; unscored kept items contribute score 0.",
    )
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--class-names", type=str, default=None, help="Comma-separated class subset override.")
    parser.add_argument("--out-root", type=str, default="./LDS/runs")
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional exact output run directory name under --out-root. Overrides the auto-generated long name.",
    )
    parser.add_argument("--duplicate-policy", choices=["max", "sum", "mean"], default="max")
    parser.add_argument(
        "--prediction-subset",
        choices=["kept", "removed"],
        default="kept",
        help="Use sum of scores over kept examples (LDS paper convention) or removed examples (if scores encode removal effect).",
    )
    parser.add_argument(
        "--prediction-sign",
        type=float,
        default=1.0,
        help="Multiplier applied to the attribution-score sum. Use -1 if the score direction is opposite to f.",
    )
    parser.add_argument("--prefer-device", choices=["auto", "cpu", "gpu"], default="gpu")
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-devices", type=int, default=None, help="Limit training to this many visible devices.")
    parser.add_argument(
        "--use-data-parallel",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override TrainConfig.use_data_parallel. Use --no-use-data-parallel to avoid pmap compile/debug issues.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Per-subset training step log interval. Smaller values make compact progress update more often.",
    )
    parser.add_argument(
        "--save-every-epochs",
        type=int,
        default=None,
        help="Checkpoint interval for each subset. Defaults to the final epoch so LDS can always evaluate true_f.",
    )
    parser.add_argument("--keep-last-k", type=int, default=1)
    parser.add_argument("--use-tqdm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--quiet-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Redirect verbose per-subset training logs to subset train.log and keep terminal LDS progress compact.",
    )
    parser.add_argument(
        "--progress-bar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show a compact tqdm progress bar for each subset while full logs are written to train.log.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write subsets/config but skip retraining and LDS.")
    args = parser.parse_args()

    simple_loss_timesteps_arg = parse_int_csv(args.simple_loss_timesteps, name="--simple-loss-timesteps")
    simple_loss_noise_seeds = parse_int_csv(args.simple_loss_noise_seeds, name="--simple-loss-noise-seeds")
    if args.simple_loss_timesteps is None and args.simple_loss_noise_seeds is not None:
        simple_loss_timesteps_arg = [int(args.simple_loss_timestep)]
    if simple_loss_noise_seeds is None and args.simple_loss_timesteps is None and (
        args.simple_loss_timestep != 0 or args.simple_loss_noise_seed != 0
    ):
        simple_loss_timesteps_arg = [int(args.simple_loss_timestep)]
        simple_loss_noise_seeds = [int(args.simple_loss_noise_seed)]

    args.base_checkpoint = resolve_path(args.base_checkpoint, must_exist=True)
    args.code_file = resolve_path(args.code_file, must_exist=True)
    args.data_root = resolve_path(args.data_root, must_exist=True) if args.data_root is not None else None
    args.attribution_sample_dir = (
        resolve_path(args.attribution_sample_dir, must_exist=True)
        if args.attribution_sample_dir is not None
        else None
    )
    args.out_root = resolve_path(args.out_root, must_exist=False)

    score_inputs = resolve_score_inputs(args.score_file)
    score_meta = infer_score_metadata(score_inputs)
    score_run_config = score_meta.get("score_run_config", {})
    if args.attribution_sample_seed is None:
        configured_sample_seed = score_run_config.get("attribution_sample_seed")
        if configured_sample_seed is not None:
            args.attribution_sample_seed = int(configured_sample_seed)
    if args.attribution_sample_index is None:
        args.attribution_sample_index = int(
            score_run_config.get("attribution_sample_index", 0)
        )
    prompt = args.prompt or infer_prompt(score_inputs)
    if prompt is None:
        raise ValueError("--prompt was not provided and could not be inferred from the score file.")
    sample_dir = args.attribution_sample_dir or infer_attribution_sample_dir(score_inputs)
    if args.target_function in ("noise_trajectory", "projected_trajectory", "simple_loss", "trajectory_state_mse") and sample_dir is None:
        raise ValueError(
            "--attribution-sample-dir was not provided and could not be inferred from the score file."
        )
    if sample_dir is not None:
        sample_dir = resolve_path(sample_dir, must_exist=True)

    cfg = load_base_config(args.base_checkpoint)
    cfg.resume_from = None
    cfg.exclude_ranges = None
    if args.data_root is not None:
        cfg.data_root = args.data_root
    elif cfg.data_root is not None:
        cfg.data_root = resolve_path(cfg.data_root, must_exist=True)
    if args.class_names is not None:
        cfg.class_names = parse_class_names(args.class_names)
    if args.train_seed is not None:
        cfg.seed = int(args.train_seed)
    if args.prefer_device is not None:
        cfg.prefer_device = args.prefer_device
    if args.epochs is not None:
        cfg.epochs = int(args.epochs)
    if args.batch_size is not None:
        cfg.batch_size = int(args.batch_size)
    if args.num_devices is not None:
        cfg.num_devices = int(args.num_devices)
    if args.use_data_parallel is not None:
        cfg.use_data_parallel = bool(args.use_data_parallel)
    if args.log_every is not None:
        cfg.log_every = int(args.log_every)
    if args.save_every_epochs is not None:
        cfg.save_every_epochs = int(args.save_every_epochs)
    else:
        cfg.save_every_epochs = max(1, int(cfg.epochs))
    if args.keep_last_k is not None:
        cfg.keep_last_k = int(args.keep_last_k)
    cfg.use_tqdm = bool(args.use_tqdm)

    simple_loss_timesteps = (
        [int(t) for t in simple_loss_timesteps_arg]
        if simple_loss_timesteps_arg is not None
        else list(range(int(cfg.timesteps)))
    )

    all_indices, all_scores, sources = combine_attribution_scores(
        score_inputs,
        duplicate_policy=args.duplicate_policy,
    )
    if len(all_indices) == 0:
        raise RuntimeError("No attribution scores loaded.")
    score_map = build_score_vector(all_indices, all_scores)

    filtered_to_cifar_rows = build_filtered_index_to_cifar_row_map(
        data_root=cfg.data_root,
        batch_names=cfg.batch_names,
        class_names=cfg.class_names,
    )
    all_filtered_indices = np.arange(len(filtered_to_cifar_rows), dtype=np.int64)
    if args.subset_universe == "score":
        universe = np.asarray(sorted(set(int(i) for i in all_indices.tolist())), dtype=np.int64)
    else:
        universe = all_filtered_indices

    if len(universe) == 0:
        raise RuntimeError("Subset universe is empty.")
    if args.subset_size > len(universe):
        raise ValueError(f"--subset-size {args.subset_size} exceeds universe size {len(universe)}")

    score_tag = sanitize_tag(os.path.basename(os.path.abspath(score_inputs[0])), "score")
    model_parent = sanitize_tag(os.path.basename(os.path.dirname(os.path.abspath(args.base_checkpoint))), "model")
    model_ckpt = sanitize_tag(os.path.basename(args.base_checkpoint).replace(".ckpt", ""), "ckpt")
    input_tag = sanitize_tag(prompt, "input")
    run_tag = sanitize_tag(args.run_tag, "") if args.run_tag else None
    name_parts = [
        "lds",
        model_parent,
        model_ckpt,
        score_tag,
        f"input_{input_tag}",
        f"m{int(args.m)}",
        f"subset{int(args.subset_size)}",
        f"seed{int(args.subset_seed)}",
    ]
    if run_tag:
        name_parts.append(run_tag)
    run_name = sanitize_tag(args.run_name, "") if args.run_name else "__".join(name_parts)
    out_dir = os.path.abspath(os.path.join(args.out_root, run_name))
    models_dir = os.path.join(out_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    rng = np.random.default_rng(int(args.subset_seed))
    subsets = []
    filtered_set = set(int(i) for i in all_filtered_indices.tolist())
    for subset_id in range(int(args.m)):
        subset_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        subset_rng = np.random.default_rng(subset_seed)
        kept = make_subset_indices(subset_rng, universe, int(args.subset_size))
        kept_set = set(int(i) for i in kept.tolist())
        excluded = np.asarray(sorted(filtered_set - kept_set), dtype=np.int64)
        prediction_indices = kept if args.prediction_subset == "kept" else excluded
        pred_sum_tau = sum_scores(prediction_indices, score_map, sign=float(args.prediction_sign))
        subset_dir = os.path.join(models_dir, f"subset_{subset_id:04d}")
        subsets.append(
            {
                "subset_id": int(subset_id),
                "subset_seed": subset_seed,
                "subset_dir": subset_dir,
                "kept_indices": kept,
                "excluded_indices": excluded,
                "prediction_indices_kind": args.prediction_subset,
                "pred_sum_tau": pred_sum_tau,
            }
        )

    config_payload = {
        "run_name": run_name,
        "out_dir": out_dir,
        "score_inputs": [os.path.abspath(p) for p in score_inputs],
        "score_sources": sources,
        "score_metadata": score_meta,
        "base_checkpoint": os.path.abspath(args.base_checkpoint),
        "code_file": os.path.abspath(args.code_file),
        "prompt": prompt,
        "attribution_sample_dir": None if sample_dir is None else os.path.abspath(sample_dir),
        "attribution_sample_seed": args.attribution_sample_seed,
        "attribution_sample_index": int(args.attribution_sample_index),
        "target_function": args.target_function,
        "trajectory_reduction": args.trajectory_reduction,
        "simple_loss_timestep_candidates": [int(t) for t in simple_loss_timesteps],
        "simple_loss_noise_seeds": None if simple_loss_noise_seeds is None else [int(s) for s in simple_loss_noise_seeds],
        "simple_loss_num_mc": int(args.simple_loss_num_mc),
        "simple_loss_mc_seed": int(args.simple_loss_mc_seed),
        "simple_loss_manual_grid": simple_loss_noise_seeds is not None,
        "m": int(args.m),
        "subset_size": int(args.subset_size),
        "subset_seed": int(args.subset_seed),
        "subset_universe": args.subset_universe,
        "prediction_subset": args.prediction_subset,
        "prediction_sign": float(args.prediction_sign),
        "universe_size": int(len(universe)),
        "num_combined_scores": int(len(all_scores)),
        "class_names": None if cfg.class_names is None else list(cfg.class_names),
        "train_config_template": asdict(cfg),
        "dry_run": bool(args.dry_run),
    }
    save_json(os.path.join(out_dir, "lds_config.json"), config_payload)
    np.save(os.path.join(out_dir, "score_indices.npy"), all_indices.astype(np.int64))
    np.save(os.path.join(out_dir, "scores.npy"), all_scores.astype(np.float64))
    np.save(os.path.join(out_dir, "subset_universe.npy"), universe.astype(np.int64))

    print("=" * 92)
    print("CIFAR LDS setup")
    print(f"out_dir             : {out_dir}")
    print(f"base_checkpoint     : {args.base_checkpoint}")
    print(f"score_inputs        : {score_inputs}")
    print(f"prompt              : {prompt}")
    print(f"target_function     : {args.target_function}")
    print(f"subset_universe     : {args.subset_universe} ({len(universe)} items)")
    print(f"m, subset_size      : {args.m}, {args.subset_size}")
    print(f"dry_run             : {args.dry_run}")
    print("=" * 92)

    for item in subsets:
        os.makedirs(item["subset_dir"], exist_ok=True)
        np.save(os.path.join(item["subset_dir"], "kept_attribution_indices.npy"), item["kept_indices"])
        np.save(os.path.join(item["subset_dir"], "excluded_attribution_indices.npy"), item["excluded_indices"])
        save_json(
            os.path.join(item["subset_dir"], "subset_metadata.json"),
            {
                "subset_id": item["subset_id"],
                "subset_seed": item["subset_seed"],
                "subset_size": int(len(item["kept_indices"])),
                "num_excluded_from_universe": int(len(item["excluded_indices"])),
                "prediction_indices_kind": item["prediction_indices_kind"],
                "pred_sum_tau": float(item["pred_sum_tau"]),
            },
        )

    if args.dry_run:
        print("[dry-run] wrote LDS config and subset files; skipped retraining/evaluation.")
        return

    evaluator = CifarTargetEvaluator(
        code_file=args.code_file,
        base_checkpoint=args.base_checkpoint,
        prompt=prompt,
        prefer_device=args.prefer_device,
        data_root=cfg.data_root,
        target_function=args.target_function,
        sample_root=sample_dir,
        sample_seed=args.attribution_sample_seed,
        sample_index=int(args.attribution_sample_index),
        max_trajectory_steps=args.max_trajectory_steps,
        trajectory_reduction=args.trajectory_reduction,
        trajectory_projection=score_meta.get("trajectory_projection"),
        simple_loss_timesteps=simple_loss_timesteps,
        simple_loss_noise_seeds=simple_loss_noise_seeds,
        simple_loss_num_mc=int(args.simple_loss_num_mc),
        simple_loss_mc_seed=int(args.simple_loss_mc_seed),
    )
    config_payload["target_metadata"] = evaluator.target_meta
    save_json(os.path.join(out_dir, "lds_config.json"), config_payload)

    rows: List[Dict[str, object]] = []
    t0 = time.time()
    for item in subsets:
        subset_id = int(item["subset_id"])
        subset_dir = item["subset_dir"]
        subset_start = time.time()
        print(
            f"[subset {subset_id + 1:03d}/{len(subsets):03d}] "
            f"train start | kept={len(item['kept_indices'])} | "
            f"log={os.path.join(subset_dir, 'train.log') if args.quiet_train else 'terminal'}",
            flush=True,
        )
        subset_cfg = TrainConfig(**asdict(cfg))
        subset_cfg.exclude_indices = selected_indices_to_exclude_indices(
            item["excluded_indices"],
            filtered_to_cifar_rows,
        )
        subset_cfg.checkpoint_dir = subset_dir
        subset_cfg.wandb_run_name = f"{run_name}__subset_{subset_id:04d}"

        save_json(
            os.path.join(subset_dir, "train_config.json"),
            {
                "train_config": asdict(subset_cfg),
                "num_excluded_rows": int(sum(len(v) for v in subset_cfg.exclude_indices.values())),
            },
        )
        run_train_with_optional_logging(
            subset_cfg,
            log_path=os.path.join(subset_dir, "train.log"),
            quiet=bool(args.quiet_train),
            prefix=f"[subset {subset_id + 1:03d}/{len(subsets):03d}] ",
            progress_bar=bool(args.progress_bar),
        )
        ckpt = latest_checkpoint(subset_dir)
        true_f, target_details = evaluator.evaluate(ckpt)

        row = {
            "subset_id": subset_id,
            "subset_seed": int(item["subset_seed"]),
            "subset_size": int(len(item["kept_indices"])),
            "prediction_subset": args.prediction_subset,
            "prediction_sign": float(args.prediction_sign),
            "pred_sum_tau": float(item["pred_sum_tau"]),
            "true_f": float(true_f),
            "checkpoint": os.path.abspath(ckpt),
            "subset_dir": os.path.abspath(subset_dir),
        }
        rows.append(row)
        save_json(
            os.path.join(subset_dir, "target_value.json"),
            {
                **row,
                "target_details": target_details,
            },
        )
        write_csv(os.path.join(out_dir, "lds_results.csv"), rows)

        pred = np.asarray([r["pred_sum_tau"] for r in rows], dtype=np.float64)
        true = np.asarray([r["true_f"] for r in rows], dtype=np.float64)
        partial_lds = spearman_corr(pred, true)
        elapsed = time.time() - subset_start
        print(
            f"[subset {subset_id + 1:03d}/{len(subsets):03d}] "
            f"done | true_f={row['true_f']:.6g} | pred_sum_tau={row['pred_sum_tau']:.6g} | "
            f"partial_LDS={partial_lds:.6g} | elapsed={elapsed / 60.0:.1f}m",
            flush=True,
        )

    pred = np.asarray([r["pred_sum_tau"] for r in rows], dtype=np.float64)
    true = np.asarray([r["true_f"] for r in rows], dtype=np.float64)
    lds = spearman_corr(pred, true)
    summary = {
        "run_name": run_name,
        "out_dir": out_dir,
        "lds_spearman": float(lds),
        "lds_percent": float(100.0 * lds) if not math.isnan(lds) else float("nan"),
        "m": int(len(rows)),
        "subset_size": int(args.subset_size),
        "subset_seed": int(args.subset_seed),
        "target_function": args.target_function,
        "trajectory_reduction": args.trajectory_reduction,
        "prediction_subset": args.prediction_subset,
        "prediction_sign": float(args.prediction_sign),
        "elapsed_sec": float(time.time() - t0),
        "results_csv": os.path.abspath(os.path.join(out_dir, "lds_results.csv")),
        "scatter_plot": os.path.abspath(os.path.join(out_dir, "lds_scatter.png")),
        "rows": rows,
    }
    save_json(os.path.join(out_dir, "lds_summary.json"), summary)
    write_csv(os.path.join(out_dir, "lds_results.csv"), rows)
    plot_scatter(
        os.path.join(out_dir, "lds_scatter.png"),
        pred,
        true,
        title=f"LDS={lds:.4f} ({100.0 * lds:.2f}%)",
    )

    if np.any(all_scores < 0):
        squared_score_map = build_score_vector(all_indices, np.square(all_scores))
        squared_rows = []
        for row, item in zip(rows, subsets):
            prediction_indices = (
                item["kept_indices"]
                if args.prediction_subset == "kept"
                else item["excluded_indices"]
            )
            squared_row = dict(row)
            squared_row["pred_sum_tau"] = sum_scores(
                prediction_indices,
                squared_score_map,
                sign=float(args.prediction_sign),
            )
            squared_rows.append(squared_row)

        squared_pred = np.asarray(
            [row["pred_sum_tau"] for row in squared_rows],
            dtype=np.float64,
        )
        squared_lds = spearman_corr(squared_pred, true)
        squared_csv = os.path.join(out_dir, "lds_results_squared_scores.csv")
        squared_png = os.path.join(out_dir, "lds_scatter_squared_scores.png")
        write_csv(squared_csv, squared_rows)
        plot_scatter(
            squared_png,
            squared_pred,
            true,
            title=f"Squared-score LDS={squared_lds:.4f} ({100.0 * squared_lds:.2f}%)",
            xlabel="Predicted sum of squared attribution scores",
        )
        print(f"squared csv  : {squared_csv}")
        print(f"squared plot : {squared_png}")

    print("=" * 92)
    print("LDS complete")
    print(f"LDS Spearman : {lds:.6f}")
    print(f"LDS (%)      : {100.0 * lds:.3f}")
    print(f"summary      : {os.path.join(out_dir, 'lds_summary.json')}")
    print(f"csv          : {os.path.join(out_dir, 'lds_results.csv')}")
    print(f"scatter      : {os.path.join(out_dir, 'lds_scatter.png')}")
    print("=" * 92)


if __name__ == "__main__":
    main()
