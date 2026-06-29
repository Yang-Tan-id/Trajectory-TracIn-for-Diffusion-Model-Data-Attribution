#!/usr/bin/env python
"""
Train one CIFAR subset model per class and compare the target f values.

For CIFAR2, this answers: train one model on horse only, one model on
automobile only, then evaluate the same LDS target function f for each model.

Example from diffusion_jax_refined/cifar2:

    EXPERIMENT_TAG=experiment1_42 python ../legacy_jax/LDS/DM_cifar_class_f.py \
      --config-module lds.CONFIG \
      --classes horse,automobile \
      --run-name experiment1_42_cifar2_class_f
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import re
import sys
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence

from DM_cifar_lds import (
    CifarTargetEvaluator,
    infer_attribution_sample_dir,
    infer_prompt,
    latest_checkpoint,
    load_base_config,
    parse_class_names,
    resolve_path,
    resolve_score_inputs,
    run_train_with_optional_logging,
    save_json,
)
from DM_counterfactual_retrain_from_attribution import (
    build_filtered_index_to_cifar_row_map,
    load_cifar_label_names,
    selected_indices_to_exclude_indices,
)


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
        ("--prompt", "prompt"),
        ("--target-function", "target_function"),
        ("--trajectory-reduction", "trajectory_reduction"),
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
    return defaults


def class_indices(
    *,
    class_name: str,
    data_root: str,
    batch_names: Optional[Sequence[str]],
    configured_class_names: Optional[Sequence[str]],
) -> tuple[List[int], int]:
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
    keep = [i for i, (_, _, label) in enumerate(mapping) if int(label) == class_id]
    return keep, len(mapping)


def write_summary_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        "class_name",
        "f",
        "num_kept",
        "num_excluded",
        "checkpoint",
        "subset_dir",
        "target_function",
        "trajectory_reduction",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one class-only CIFAR model per class and compare f.")
    parser.add_argument("--config-module", type=str, default=None, help="Optional module with COMMANDS['lds'], e.g. lds.CONFIG")
    parser.add_argument("--classes", type=str, default="horse,automobile")
    parser.add_argument("--score-file", type=str, default=None, help="Optional; used only to infer prompt/sample dir if not provided.")
    parser.add_argument("--base-checkpoint", type=str, default=None)
    parser.add_argument("--code-file", default="DM__training_CIFAR10_pixel.py")
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
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--class-names", type=str, default=None, help="Outer filtered dataset classes, e.g. horse,automobile")
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
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

    if args.base_checkpoint is None:
        raise ValueError("--base-checkpoint is required unless --config-module provides it.")
    if args.data_root is None:
        raise ValueError("--data-root is required unless --config-module provides it.")
    if args.target_function is None:
        args.target_function = "noise_trajectory"
    if args.trajectory_reduction is None:
        args.trajectory_reduction = "sum"
    if args.out_root is None:
        args.out_root = "./LDS/class_f"

    args.base_checkpoint = resolve_path(args.base_checkpoint, must_exist=True)
    args.code_file = resolve_path(args.code_file, must_exist=True)
    args.data_root = resolve_path(args.data_root, must_exist=True)
    args.out_root = resolve_path(args.out_root, must_exist=False)

    score_inputs = resolve_score_inputs(args.score_file) if args.score_file else []
    prompt = args.prompt or (infer_prompt(score_inputs) if score_inputs else None)
    if prompt is None:
        raise ValueError("--prompt is required unless it can be inferred from --score-file.")

    sample_dir = args.attribution_sample_dir
    if sample_dir is None and score_inputs:
        sample_dir = infer_attribution_sample_dir(score_inputs)
    if sample_dir is not None:
        sample_dir = resolve_path(sample_dir, must_exist=True)
    if args.target_function in ("noise_trajectory", "simple_loss") and sample_dir is None:
        raise ValueError("--attribution-sample-dir is required for target f evaluation.")

    cfg_template = load_base_config(args.base_checkpoint)
    cfg_template.resume_from = None
    cfg_template.exclude_ranges = None
    cfg_template.data_root = args.data_root
    if args.class_names is not None:
        cfg_template.class_names = parse_class_names(args.class_names)
    if args.train_seed is not None:
        cfg_template.seed = int(args.train_seed)
    if args.prefer_device is not None:
        cfg_template.prefer_device = args.prefer_device
    if args.epochs is not None:
        cfg_template.epochs = int(args.epochs)
    if args.batch_size is not None:
        cfg_template.batch_size = int(args.batch_size)
    if args.num_devices is not None:
        cfg_template.num_devices = int(args.num_devices)
    if args.use_data_parallel is not None:
        cfg_template.use_data_parallel = bool(args.use_data_parallel)
    if args.log_every is not None:
        cfg_template.log_every = int(args.log_every)
    if args.save_every_epochs is not None:
        cfg_template.save_every_epochs = int(args.save_every_epochs)
    else:
        cfg_template.save_every_epochs = max(1, int(cfg_template.epochs))
    if args.keep_last_k is not None:
        cfg_template.keep_last_k = int(args.keep_last_k)
    cfg_template.use_tqdm = bool(args.use_tqdm)

    filtered_mapping = build_filtered_index_to_cifar_row_map(
        data_root=cfg_template.data_root,
        batch_names=cfg_template.batch_names,
        class_names=cfg_template.class_names,
    )
    filtered_size = len(filtered_mapping)
    all_filtered = set(range(filtered_size))
    simple_loss_timesteps = (
        parse_int_list(args.simple_loss_timesteps)
        if args.simple_loss_timesteps is not None
        else list(range(int(cfg_template.timesteps)))
    )
    simple_loss_noise_seeds = parse_int_list(args.simple_loss_noise_seeds)

    evaluator = None
    if not args.dry_run:
        evaluator = CifarTargetEvaluator(
            code_file=args.code_file,
            base_checkpoint=args.base_checkpoint,
            prompt=prompt,
            prefer_device=args.prefer_device,
            data_root=cfg_template.data_root,
            target_function=args.target_function,
            sample_root=sample_dir,
            sample_seed=args.attribution_sample_seed,
            sample_index=int(args.attribution_sample_index),
            max_trajectory_steps=args.max_trajectory_steps,
            trajectory_reduction=args.trajectory_reduction,
            simple_loss_timesteps=simple_loss_timesteps,
            simple_loss_noise_seeds=simple_loss_noise_seeds,
            simple_loss_num_mc=int(args.simple_loss_num_mc),
            simple_loss_mc_seed=int(args.simple_loss_mc_seed),
        )

    classes = split_csv(args.classes)
    base_run_name = sanitize_tag(args.run_name, "class_f")
    out_dir = os.path.abspath(os.path.join(args.out_root, base_run_name))
    models_dir = os.path.join(out_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    rows = []
    for class_name in classes:
        kept, _ = class_indices(
            class_name=class_name,
            data_root=cfg_template.data_root,
            batch_names=cfg_template.batch_names,
            configured_class_names=cfg_template.class_names,
        )
        kept_set = set(int(i) for i in kept)
        excluded = sorted(all_filtered - kept_set)
        subset_dir = os.path.join(models_dir, f"class_{sanitize_tag(class_name, 'class')}")
        os.makedirs(subset_dir, exist_ok=True)
        save_json(
            os.path.join(subset_dir, "class_subset_metadata.json"),
            {
                "class_name": class_name,
                "kept_filtered_indices": kept,
                "excluded_filtered_indices": excluded,
                "num_kept": len(kept),
                "num_excluded": len(excluded),
            },
        )
        print("=" * 92)
        print(f"class model: {class_name}")
        print(f"kept     : {len(kept)}")
        print(f"excluded : {len(excluded)}")
        print(f"out      : {subset_dir}")
        print("=" * 92)

        row = {
            "class_name": class_name,
            "num_kept": len(kept),
            "num_excluded": len(excluded),
            "subset_dir": os.path.abspath(subset_dir),
            "target_function": args.target_function,
            "trajectory_reduction": args.trajectory_reduction,
        }
        if not args.dry_run:
            class_cfg = type(cfg_template)(**asdict(cfg_template))
            class_cfg.checkpoint_dir = subset_dir
            class_cfg.resume_from = None
            class_cfg.exclude_indices = selected_indices_to_exclude_indices(
                excluded,
                filtered_mapping,
            )
            class_cfg.exclude_ranges = None
            run_train_with_optional_logging(
                class_cfg,
                log_path=os.path.join(subset_dir, "train.log"),
                quiet=bool(args.quiet_train),
                prefix=f"[class {class_name}] ",
                progress_bar=bool(args.progress_bar),
            )
            ckpt = latest_checkpoint(subset_dir)
            f_value, f_details = evaluator.evaluate(ckpt)
            row.update(
                {
                    "f": float(f_value),
                    "checkpoint": os.path.abspath(ckpt),
                    "target_details": f_details,
                }
            )
            print(f"[class {class_name}] f={f_value:.6g} | ckpt={ckpt}")
        rows.append(row)

    summary = {
        "run_name": base_run_name,
        "out_dir": out_dir,
        "prompt": prompt,
        "attribution_sample_dir": sample_dir,
        "target_function": args.target_function,
        "trajectory_reduction": args.trajectory_reduction,
        "filtered_dataset_size": filtered_size,
        "classes": rows,
        "dry_run": bool(args.dry_run),
    }
    save_json(os.path.join(out_dir, "class_f_summary.json"), summary)
    write_summary_csv(os.path.join(out_dir, "class_f_summary.csv"), rows)
    print("=" * 92)
    print("Class f comparison complete")
    for row in rows:
        if "f" in row:
            print(f"{row['class_name']}: f={row['f']:.6g} | kept={row['num_kept']}")
        else:
            print(f"{row['class_name']}: dry-run | kept={row['num_kept']}")
    print(f"summary json: {os.path.join(out_dir, 'class_f_summary.json')}")
    print(f"summary csv : {os.path.join(out_dir, 'class_f_summary.csv')}")
    print("=" * 92)


if __name__ == "__main__":
    main()
