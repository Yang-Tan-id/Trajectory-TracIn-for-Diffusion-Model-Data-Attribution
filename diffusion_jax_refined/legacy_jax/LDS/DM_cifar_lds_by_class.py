#!/usr/bin/env python
"""
Class-conditional CIFAR LDS wrapper.

This is intentionally separate from DM_cifar_lds.py.  It reuses the same
retrain/evaluate machinery, but samples each LDS subset from one requested
class at a time (for CIFAR2, e.g. horse vs automobile) and writes one LDS run
per class plus a small comparison summary.

Example from diffusion_jax_refined/cifar2:

    python ../legacy_jax/LDS/DM_cifar_lds_by_class.py \
      --config-module lds.CONFIG \
      --classes horse,automobile \
      --subset-size 2500 \
      --m 100

Or run from legacy_jax with explicit paths, mirroring DM_cifar_lds.py.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from DM_cifar_lds import (
    CifarTargetEvaluator,
    build_score_vector,
    combine_attribution_scores,
    infer_attribution_sample_dir,
    infer_prompt,
    infer_score_metadata,
    latest_checkpoint,
    load_base_config,
    parse_class_names,
    plot_scatter,
    resolve_path,
    resolve_score_inputs,
    save_json,
    spearman_corr,
    sum_scores,
    write_csv,
    run_train_with_optional_logging,
)
from DM_counterfactual_retrain_from_attribution import (
    build_filtered_index_to_cifar_row_map,
    load_cifar_label_names,
    selected_indices_to_exclude_indices,
)


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def sanitize_tag(text: Optional[str], default: str = "unknown") -> str:
    if text is None or str(text).strip() == "":
        return default
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or default


def split_csv(text: Optional[str]) -> List[str]:
    if text is None:
        return []
    return [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]


def parse_int_list(text: Optional[str]) -> Optional[List[int]]:
    if text is None or str(text).strip() == "":
        return None
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def parse_command_value(command: Sequence[str], flag: str) -> Optional[str]:
    for i, token in enumerate(command):
        if token == flag and i + 1 < len(command):
            return command[i + 1]
        prefix = flag + "="
        if token.startswith(prefix):
            return token[len(prefix):]
    return None


def load_defaults_from_config_module(module_name: Optional[str]) -> Dict[str, object]:
    if not module_name:
        return {}
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    module = importlib.import_module(module_name)
    command = list(module.COMMANDS["lds"])
    defaults: Dict[str, object] = {}
    for flag, key in (
        ("--score-file", "score_file"),
        ("--base-checkpoint", "base_checkpoint"),
        ("--data-root", "data_root"),
        ("--class-names", "class_names"),
        ("--subset-size", "subset_size"),
        ("--m", "m"),
        ("--subset-seed", "subset_seed"),
        ("--prompt", "prompt"),
        ("--target-function", "target_function"),
        ("--trajectory-reduction", "trajectory_reduction"),
        ("--prediction-sign", "prediction_sign"),
        ("--out-root", "out_root"),
        ("--run-name", "run_name"),
        ("--epochs", "epochs"),
        ("--prefer-device", "prefer_device"),
        ("--num-devices", "num_devices"),
        ("--save-every-epochs", "save_every_epochs"),
        ("--keep-last-k", "keep_last_k"),
    ):
        value = parse_command_value(command, flag)
        if value is not None:
            defaults[key] = value
    defaults["use_data_parallel"] = "--no-use-data-parallel" not in command
    defaults["command_cwd"] = getattr(module, "COMMAND_CWD", None)
    return defaults


def class_filtered_universe(
    *,
    class_name: str,
    data_root: str,
    batch_names: Optional[Sequence[str]],
    configured_class_names: Optional[Sequence[str]],
    scored_indices: np.ndarray,
    subset_universe: str,
) -> Tuple[np.ndarray, int]:
    label_names = load_cifar_label_names(data_root)
    name_to_id = {name: i for i, name in enumerate(label_names)}
    if class_name not in name_to_id:
        raise ValueError(f"Unknown class {class_name!r}; available labels: {label_names}")
    class_id = int(name_to_id[class_name])

    mapping = build_filtered_index_to_cifar_row_map(
        data_root=data_root,
        batch_names=batch_names,
        class_names=configured_class_names,
    )
    class_indices = np.asarray(
        [i for i, (_, _, label) in enumerate(mapping) if int(label) == class_id],
        dtype=np.int64,
    )
    if subset_universe == "score":
        scored_set = set(int(i) for i in scored_indices.tolist())
        class_indices = np.asarray(
            [int(i) for i in class_indices.tolist() if int(i) in scored_set],
            dtype=np.int64,
        )
    return class_indices, len(mapping)


def make_subset_indices(
    rng: np.random.Generator,
    universe: np.ndarray,
    subset_size: int,
) -> np.ndarray:
    if subset_size <= 0:
        raise ValueError(f"subset_size must be positive, got {subset_size}")
    if subset_size > len(universe):
        raise ValueError(f"subset_size={subset_size} exceeds class universe size {len(universe)}")
    return np.asarray(rng.choice(universe, size=subset_size, replace=False), dtype=np.int64)


def write_comparison_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        "class_name",
        "lds_spearman",
        "lds_percent",
        "m",
        "subset_size",
        "class_universe_size",
        "num_combined_scores",
        "results_csv",
        "scatter_plot",
        "run_dir",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def run_one_class(
    args: argparse.Namespace,
    *,
    class_name: str,
    class_index: int,
    score_inputs: Sequence[str],
    all_indices: np.ndarray,
    all_scores: np.ndarray,
    sources: Sequence[Dict[str, object]],
    score_meta: Dict[str, object],
    prompt: str,
    sample_dir: Optional[str],
) -> Dict[str, object]:
    t0 = time.time()
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
        parse_int_list(args.simple_loss_timesteps)
        if args.simple_loss_timesteps is not None
        else list(range(int(cfg.timesteps)))
    )
    simple_loss_noise_seeds = parse_int_list(args.simple_loss_noise_seeds)

    score_map = build_score_vector(all_indices, all_scores)
    universe, filtered_size = class_filtered_universe(
        class_name=class_name,
        data_root=cfg.data_root,
        batch_names=cfg.batch_names,
        configured_class_names=cfg.class_names,
        scored_indices=all_indices,
        subset_universe=args.subset_universe,
    )
    if len(universe) == 0:
        raise RuntimeError(f"Class universe is empty for {class_name!r}.")
    if args.subset_size > len(universe):
        raise ValueError(
            f"--subset-size {args.subset_size} exceeds {class_name} universe size {len(universe)}"
        )

    base_run_name = sanitize_tag(args.run_name, "class_lds") if args.run_name else "class_lds"
    run_name = f"{base_run_name}__class_{sanitize_tag(class_name, 'class')}"
    out_dir = os.path.abspath(os.path.join(args.out_root, run_name))
    models_dir = os.path.join(out_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    rng = np.random.default_rng(int(args.subset_seed) + class_index * 1_000_003)
    subsets = []
    for subset_id in range(int(args.m)):
        subset_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        subset_rng = np.random.default_rng(subset_seed)
        kept = make_subset_indices(subset_rng, universe, int(args.subset_size))
        kept_set = set(int(i) for i in kept.tolist())
        class_excluded = np.asarray(
            sorted(set(int(i) for i in universe.tolist()) - kept_set),
            dtype=np.int64,
        )
        prediction_indices = kept if args.prediction_subset == "kept" else class_excluded
        pred_sum_tau = sum_scores(prediction_indices, score_map, sign=float(args.prediction_sign))
        subsets.append(
            {
                "subset_id": int(subset_id),
                "subset_seed": subset_seed,
                "subset_dir": os.path.join(models_dir, f"subset_{subset_id:04d}"),
                "kept_indices": kept,
                "class_excluded_indices": class_excluded,
                "prediction_indices_kind": args.prediction_subset,
                "pred_sum_tau": pred_sum_tau,
            }
        )

    config_payload = {
        "run_name": run_name,
        "out_dir": out_dir,
        "class_name": class_name,
        "score_inputs": [os.path.abspath(p) for p in score_inputs],
        "score_sources": list(sources),
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
        "simple_loss_noise_seeds": simple_loss_noise_seeds,
        "simple_loss_num_mc": int(args.simple_loss_num_mc),
        "simple_loss_mc_seed": int(args.simple_loss_mc_seed),
        "m": int(args.m),
        "subset_size": int(args.subset_size),
        "subset_seed": int(args.subset_seed),
        "subset_universe": args.subset_universe,
        "class_universe_size": int(len(universe)),
        "filtered_dataset_size": int(filtered_size),
        "prediction_subset": args.prediction_subset,
        "prediction_sign": float(args.prediction_sign),
        "num_combined_scores": int(len(all_scores)),
        "class_names": None if cfg.class_names is None else list(cfg.class_names),
        "train_config_template": asdict(cfg),
        "dry_run": bool(args.dry_run),
    }
    save_json(os.path.join(out_dir, "lds_class_config.json"), config_payload)
    np.save(os.path.join(out_dir, "score_indices.npy"), all_indices.astype(np.int64))
    np.save(os.path.join(out_dir, "scores.npy"), all_scores.astype(np.float64))
    np.save(os.path.join(out_dir, "class_subset_universe.npy"), universe.astype(np.int64))

    print("=" * 92)
    print(f"CIFAR class LDS setup | class={class_name}")
    print(f"out_dir              : {out_dir}")
    print(f"class_universe_size  : {len(universe)}")
    print(f"subset_size          : {args.subset_size}")
    print(f"m                    : {args.m}")
    print(f"prediction_subset    : {args.prediction_subset}")
    print(f"prediction_sign      : {args.prediction_sign}")
    print(f"dry_run              : {args.dry_run}")
    print("=" * 92)

    for item in subsets:
        subset_dir = item["subset_dir"]
        os.makedirs(subset_dir, exist_ok=True)
        np.save(os.path.join(subset_dir, "kept_attribution_indices.npy"), item["kept_indices"])
        np.save(os.path.join(subset_dir, "class_excluded_attribution_indices.npy"), item["class_excluded_indices"])
        save_json(
            os.path.join(subset_dir, "subset_metadata.json"),
            {
                "subset_id": item["subset_id"],
                "subset_seed": int(item["subset_seed"]),
                "class_name": class_name,
                "subset_size": int(len(item["kept_indices"])),
                "class_universe_size": int(len(universe)),
                "num_excluded_from_class_universe": int(len(item["class_excluded_indices"])),
                "prediction_indices_kind": item["prediction_indices_kind"],
                "pred_sum_tau": float(item["pred_sum_tau"]),
            },
        )

    if args.dry_run:
        summary = {
            "class_name": class_name,
            "out_dir": out_dir,
            "m": int(args.m),
            "subset_size": int(args.subset_size),
            "class_universe_size": int(len(universe)),
            "dry_run": True,
        }
        save_json(os.path.join(out_dir, "lds_class_summary.json"), summary)
        return summary

    evaluator = CifarTargetEvaluator(
        code_file=args.code_file,
        base_checkpoint=args.base_checkpoint,
        prompt=prompt,
        prefer_device=args.prefer_device,
        data_root=cfg.data_root,
        target_function=args.target_function,
        sample_root=sample_dir,
        sample_seed=args.attribution_sample_seed,
        sample_index=args.attribution_sample_index,
        max_trajectory_steps=args.max_trajectory_steps,
        trajectory_reduction=args.trajectory_reduction,
        simple_loss_timesteps=simple_loss_timesteps,
        simple_loss_noise_seeds=simple_loss_noise_seeds,
        simple_loss_num_mc=args.simple_loss_num_mc,
        simple_loss_mc_seed=args.simple_loss_mc_seed,
    )

    rows = []
    filtered_to_cifar_rows = build_filtered_index_to_cifar_row_map(
        data_root=cfg.data_root,
        batch_names=cfg.batch_names,
        class_names=cfg.class_names,
    )
    for subset_id, item in enumerate(subsets):
        subset_dir = item["subset_dir"]
        class_excluded_set = set(int(i) for i in item["class_excluded_indices"].tolist())
        non_class_set = set(range(filtered_size)) - set(int(i) for i in universe.tolist())
        train_excluded = sorted(non_class_set | class_excluded_set)
        subset_cfg = type(cfg)(**asdict(cfg))
        subset_cfg.checkpoint_dir = subset_dir
        subset_cfg.resume_from = None
        subset_cfg.exclude_indices = selected_indices_to_exclude_indices(
            train_excluded,
            filtered_to_cifar_rows,
        )
        subset_cfg.exclude_ranges = None
        subset_cfg.seed = int(item["subset_seed"])
        run_train_with_optional_logging(
            subset_cfg,
            log_path=os.path.join(subset_dir, "train.log"),
            quiet=bool(args.quiet_train),
            prefix=f"[{class_name} subset {subset_id + 1:03d}/{len(subsets):03d}] ",
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
        pred = np.asarray([r["pred_sum_tau"] for r in rows], dtype=np.float64)
        true = np.asarray([r["true_f"] for r in rows], dtype=np.float64)
        partial_lds = spearman_corr(pred, true)
        print(
            f"[{class_name} subset {subset_id + 1}/{len(subsets)}] done | "
            f"true_f={row['true_f']:.6g} | pred_sum_tau={row['pred_sum_tau']:.6g} | "
            f"partial_LDS={partial_lds:.6g}"
        )

    pred = np.asarray([r["pred_sum_tau"] for r in rows], dtype=np.float64)
    true = np.asarray([r["true_f"] for r in rows], dtype=np.float64)
    lds = spearman_corr(pred, true)
    summary = {
        "class_name": class_name,
        "run_name": run_name,
        "out_dir": out_dir,
        "lds_spearman": float(lds),
        "lds_percent": float(100.0 * lds) if not math.isnan(lds) else float("nan"),
        "m": int(len(rows)),
        "subset_size": int(args.subset_size),
        "subset_seed": int(args.subset_seed),
        "class_universe_size": int(len(universe)),
        "target_function": args.target_function,
        "trajectory_reduction": args.trajectory_reduction,
        "prediction_subset": args.prediction_subset,
        "prediction_sign": float(args.prediction_sign),
        "elapsed_sec": float(time.time() - t0),
        "results_csv": os.path.abspath(os.path.join(out_dir, "lds_results.csv")),
        "scatter_plot": os.path.abspath(os.path.join(out_dir, "lds_scatter.png")),
        "rows": rows,
    }
    save_json(os.path.join(out_dir, "lds_class_summary.json"), summary)
    write_csv(os.path.join(out_dir, "lds_results.csv"), rows)
    plot_scatter(
        os.path.join(out_dir, "lds_scatter.png"),
        pred,
        true,
        title=f"{class_name} LDS={lds:.4f} ({100.0 * lds:.2f}%)",
    )
    print("=" * 92)
    print(f"Class LDS complete: {class_name}")
    print(f"LDS Spearman : {lds:.6f}")
    print(f"LDS (%)      : {100.0 * lds:.3f}")
    print(f"summary      : {os.path.join(out_dir, 'lds_class_summary.json')}")
    print(f"csv          : {os.path.join(out_dir, 'lds_results.csv')}")
    print(f"scatter      : {os.path.join(out_dir, 'lds_scatter.png')}")
    print("=" * 92)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CIFAR LDS separately for each requested class.")
    parser.add_argument("--config-module", type=str, default=None, help="Optional Python module with COMMANDS['lds'], e.g. lds.CONFIG from a dataset dir.")
    parser.add_argument("--classes", type=str, default="horse,automobile", help="Comma-separated classes to compare.")
    parser.add_argument("--score-file", type=str, default=None)
    parser.add_argument("--base-checkpoint", type=str, default=None)
    parser.add_argument("--code-file", default="DM__training_CIFAR10_pixel.py")
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--subset-seed", type=int, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--attribution-sample-dir", type=str, default=None)
    parser.add_argument("--attribution-sample-seed", type=int, default=None)
    parser.add_argument("--attribution-sample-index", type=int, default=0)
    parser.add_argument("--max-trajectory-steps", type=int, default=None)
    parser.add_argument("--target-function", choices=["noise_trajectory", "simple_loss"], default=None)
    parser.add_argument("--trajectory-reduction", choices=["mean", "sum"], default=None)
    parser.add_argument("--simple-loss-timesteps", type=str, default=None)
    parser.add_argument("--simple-loss-noise-seeds", type=str, default=None)
    parser.add_argument("--simple-loss-num-mc", type=int, default=16)
    parser.add_argument("--simple-loss-mc-seed", type=int, default=0)
    parser.add_argument("--subset-universe", choices=["score", "all_filtered"], default="score")
    parser.add_argument("--prediction-subset", choices=["kept", "removed"], default="kept")
    parser.add_argument("--prediction-sign", type=float, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--class-names", type=str, default=None)
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--duplicate-policy", choices=["max", "sum", "mean"], default="max")
    parser.add_argument("--prefer-device", choices=["auto", "cpu", "gpu"], default=None)
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-devices", type=int, default=None)
    parser.add_argument("--use-data-parallel", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every-epochs", type=int, default=None)
    parser.add_argument("--keep-last-k", type=int, default=None)
    parser.add_argument("--quiet-train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-bar", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-tqdm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    defaults = load_defaults_from_config_module(args.config_module)
    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    if args.score_file is None:
        raise ValueError("--score-file is required unless --config-module provides it.")
    if args.base_checkpoint is None:
        raise ValueError("--base-checkpoint is required unless --config-module provides it.")
    if args.data_root is None:
        raise ValueError("--data-root is required unless --config-module provides it.")
    if args.subset_size is None:
        raise ValueError("--subset-size is required unless --config-module provides it.")
    if args.m is None:
        raise ValueError("--m is required unless --config-module provides it.")
    args.subset_size = int(args.subset_size)
    args.m = int(args.m)
    if args.subset_seed is None:
        args.subset_seed = 0
    args.subset_seed = int(args.subset_seed)
    if args.target_function is None:
        args.target_function = "noise_trajectory"
    if args.trajectory_reduction is None:
        args.trajectory_reduction = "sum"
    if args.prediction_sign is None:
        args.prediction_sign = 1.0
    args.prediction_sign = float(args.prediction_sign)
    if args.attribution_sample_seed is not None:
        args.attribution_sample_seed = int(args.attribution_sample_seed)
    args.attribution_sample_index = int(args.attribution_sample_index)
    args.simple_loss_num_mc = int(args.simple_loss_num_mc)
    args.simple_loss_mc_seed = int(args.simple_loss_mc_seed)
    if args.epochs is not None:
        args.epochs = int(args.epochs)
    if args.batch_size is not None:
        args.batch_size = int(args.batch_size)
    if args.num_devices is not None:
        args.num_devices = int(args.num_devices)
    if args.train_seed is not None:
        args.train_seed = int(args.train_seed)
    if args.save_every_epochs is not None:
        args.save_every_epochs = int(args.save_every_epochs)
    if args.keep_last_k is not None:
        args.keep_last_k = int(args.keep_last_k)
    if args.log_every is not None:
        args.log_every = int(args.log_every)
    if args.out_root is None:
        args.out_root = "./LDS/class_runs"

    args.base_checkpoint = resolve_path(args.base_checkpoint, must_exist=True)
    args.code_file = resolve_path(args.code_file, must_exist=True)
    args.data_root = resolve_path(args.data_root, must_exist=True)
    args.out_root = resolve_path(args.out_root, must_exist=False)
    sample_dir = args.attribution_sample_dir
    if sample_dir is not None:
        sample_dir = resolve_path(sample_dir, must_exist=True)

    score_inputs = resolve_score_inputs(args.score_file)
    score_meta = infer_score_metadata(score_inputs)
    prompt = args.prompt or infer_prompt(score_inputs)
    if prompt is None:
        raise ValueError("--prompt was not provided and could not be inferred from scores.")
    if sample_dir is None:
        sample_dir = infer_attribution_sample_dir(score_inputs)
        if sample_dir is not None:
            sample_dir = resolve_path(sample_dir, must_exist=True)
    if args.target_function in ("noise_trajectory", "simple_loss") and sample_dir is None:
        raise ValueError("--attribution-sample-dir is required or must be inferable from scores.")

    all_indices, all_scores, sources = combine_attribution_scores(
        score_inputs,
        duplicate_policy=args.duplicate_policy,
    )
    if len(all_indices) == 0:
        raise RuntimeError("No attribution scores loaded.")

    class_names = split_csv(args.classes)
    if not class_names:
        raise ValueError("--classes must include at least one class.")

    summaries = []
    for class_index, class_name in enumerate(class_names):
        summaries.append(
            run_one_class(
                args,
                class_name=class_name,
                class_index=class_index,
                score_inputs=score_inputs,
                all_indices=all_indices,
                all_scores=all_scores,
                sources=sources,
                score_meta=score_meta,
                prompt=prompt,
                sample_dir=sample_dir,
            )
        )

    comparison_dir = os.path.abspath(os.path.join(args.out_root, sanitize_tag(args.run_name, "class_lds")))
    os.makedirs(comparison_dir, exist_ok=True)
    comparison_json = os.path.join(comparison_dir, "lds_class_comparison.json")
    comparison_csv = os.path.join(comparison_dir, "lds_class_comparison.csv")
    save_json(comparison_json, {"classes": summaries})
    write_comparison_csv(comparison_csv, summaries)
    print("=" * 92)
    print("Class LDS comparison complete")
    for row in summaries:
        if row.get("dry_run"):
            print(
                f"{row['class_name']}: dry-run | "
                f"class_universe={row['class_universe_size']} | subset={row['subset_size']}"
            )
        else:
            print(
                f"{row['class_name']}: LDS={row['lds_spearman']:.6f} "
                f"({row['lds_percent']:.3f}%) | class_universe={row['class_universe_size']}"
            )
    print(f"comparison json: {comparison_json}")
    print(f"comparison csv : {comparison_csv}")
    print("=" * 92)


if __name__ == "__main__":
    main()
