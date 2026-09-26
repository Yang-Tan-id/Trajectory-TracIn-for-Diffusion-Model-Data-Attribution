#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import re
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
REFINE_ROOT = SHAPES_ROOT.parent
LEGACY_ROOT = REFINE_ROOT / "legacy_jax"
if str(LEGACY_ROOT) not in sys.path:
    sys.path.insert(0, str(LEGACY_ROOT))


METHODS = ("traj_next", "traj_previous", "das_mc4_lambda1")


def prompt_tag(prompt: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", prompt.replace(",", "__"))
    return re.sub(r"_+", "_", text).strip("_")[:160] or "empty"


def load_queries(path: Path) -> dict[int, dict]:
    payload = json.loads(path.read_text())
    records = payload["queries"] if isinstance(payload, dict) else payload
    return {int(record.get("query_id", i)): record for i, record in enumerate(records)}


def score_directory(
    *, result_root: Path, train_seed: int, record: dict, method: str
) -> Path:
    prompt = str(record["prompt"])
    sample_seed = int(record["initial_seed"])
    root = (
        result_root
        / "attribution_score"
        / "prompted_solo"
        / f"train_seed_{train_seed}"
        / f"query_{prompt_tag(prompt)}"
        / f"initial_seed_{sample_seed}"
    )
    if method == "traj_next":
        return root / "traj_tracin_adamw_full_aligned10x10" / "score"
    if method == "traj_previous":
        return (
            root
            / "traj_tracin_adamw_full_aligned10x10_previous_target_linear_previous_lr_q0_20"
            / "score"
        )
    if method == "das_mc4_lambda1":
        return root / "das_factorized_mc4_indist100q_original100x1" / "score" / "lambda_1"
    raise ValueError(method)


def select_top1000(score_dir: Path, method: str, topk: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(np.load(score_dir / "scores.npy"), dtype=np.float64).reshape(-1)
    indices = np.asarray(np.load(score_dir / "score_indices.npy"), dtype=np.int64).reshape(-1)
    if raw.shape != indices.shape:
        raise ValueError(f"score/index mismatch in {score_dir}: {raw.shape} vs {indices.shape}")
    # Requested removal convention: reverse both Traj TracIn scores, but leave
    # DAS in its stored/raw orientation, then take the largest ranking values.
    ranking = -raw if method.startswith("traj_") else raw
    ranking = np.where(np.isfinite(ranking), ranking, -np.inf)
    order = np.argsort(-ranking, kind="stable")[:topk]
    if len(order) != topk or np.any(~np.isfinite(ranking[order])):
        raise ValueError(f"{score_dir} does not contain {topk} finite scores")
    return indices[order], raw[order], ranking[order]


def load_training_config(base_checkpoint: Path, output_dir: Path, removed: np.ndarray):
    from DM__training_CIFAR5_MULTI_pixel import TrainConfig

    with base_checkpoint.open("rb") as handle:
        payload = pickle.load(handle)
    saved = dict(payload.get("config", {}))
    valid = set(TrainConfig.__dataclass_fields__)
    cfg = TrainConfig(**{key: value for key, value in saved.items() if key in valid})
    cfg.resume_from = None
    cfg.exclude_ranges = None
    cfg.exclude_indices = {0: tuple(int(x) for x in removed.tolist())}
    cfg.checkpoint_dir = str(output_dir)
    cfg.seed = 42
    cfg.epochs = 200
    cfg.save_every_epochs = 200
    cfg.keep_last_k = 1
    cfg.prefer_device = "gpu"
    cfg.use_data_parallel = False
    return cfg


def latest_checkpoint(directory: Path) -> Path:
    paths = sorted(directory.glob("seed_*_epoch_*.ckpt"))
    if not paths:
        raise FileNotFoundError(f"No trained checkpoint under {directory}")
    return paths[-1]


def sample_root(result_root: Path, record: dict, train_seed: int, epochs: int) -> Path:
    return (
        result_root
        / "sample_ddim_eta0_1000"
        / "cifar"
        / f"prompt_{prompt_tag(str(record['prompt']))}"
        / f"model_prompted_solo__ckpt_seed_{train_seed}_epoch_{epochs:04d}"
    )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--query-id", type=int, required=True)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--stage", choices=("train", "eval", "all"), default="all")
    args = parser.parse_args()

    records = load_queries(args.query_file)
    if args.query_id not in records:
        raise KeyError(f"query id {args.query_id} is absent from {args.query_file}")
    record = records[args.query_id]
    result_root = SHAPES_ROOT / "result" / args.experiment
    score_dir = score_directory(
        result_root=result_root,
        train_seed=args.train_seed,
        record=record,
        method=args.method,
    )
    removed, raw_scores, ranking_scores = select_top1000(score_dir, args.method, args.topk)
    run_dir = (
        result_root
        / "top1000_removal"
        / args.method
        / f"query_{args.query_id:03d}_seed_{int(record['initial_seed']):03d}"
    )
    model_dir = run_dir / "model"
    metadata_path = run_dir / "removal_manifest.json"
    metadata = {
        "method": args.method,
        "query_id": args.query_id,
        "prompt": str(record["prompt"]),
        "initial_seed": int(record["initial_seed"]),
        "score_directory": str(score_dir.resolve()),
        "score_component": "raw",
        "ranking_sign": -1 if args.method.startswith("traj_") else 1,
        "topk": args.topk,
        "removed_dataset_indices": [int(x) for x in removed],
        "removed_raw_scores": [float(x) for x in raw_scores],
        "removed_ranking_scores": [float(x) for x in ranking_scores],
    }
    write_json(metadata_path, metadata)

    base_checkpoint = (
        result_root
        / "model"
        / "prompted_jax"
        / f"seed_{args.train_seed}_epoch_{args.epochs:04d}.ckpt"
    )
    if args.stage in ("train", "all"):
        final = model_dir / f"seed_{args.train_seed}_epoch_{args.epochs:04d}.ckpt"
        if final.is_file():
            print(f"[skip] trained model exists: {final}", flush=True)
        else:
            from DM__training_CIFAR5_MULTI_pixel import train

            cfg = load_training_config(base_checkpoint, model_dir, removed)
            metadata["train_config"] = asdict(cfg)
            write_json(metadata_path, metadata)
            print(f"[train] {args.method} Q{args.query_id} remove top {args.topk}", flush=True)
            train(cfg)
            del cfg
            gc.collect()

    if args.stage in ("eval", "all"):
        target_checkpoint = latest_checkpoint(model_dir)
        from LDS.DM_cifar_lds import CifarTargetEvaluator

        evaluator = CifarTargetEvaluator(
            code_file=str(LEGACY_ROOT / "DM__training_3DSHAPES_pixel.py"),
            base_checkpoint=str(base_checkpoint),
            prompt=str(record["prompt"]),
            prefer_device="gpu",
            data_root=str(REFINE_ROOT / "dataset" / "3dshapes" / "20000"),
            target_function="traj_counterfactual",
            sample_root=str(sample_root(result_root, record, args.train_seed, args.epochs)),
            sample_seed=int(record["initial_seed"]),
            sample_index=0,
            max_trajectory_steps=None,
            trajectory_reduction="mean",
            trajectory_projection=None,
            simple_loss_timesteps=[0],
            simple_loss_noise_seeds=None,
            simple_loss_num_mc=1,
            simple_loss_mc_seed=0,
            trajectory_sampler="ddim_eta0",
        )
        values = evaluator.evaluate_many(
            str(target_checkpoint), ("endpoint_counterfactual", "traj_counterfactual")
        )
        output = {
            **metadata,
            "base_checkpoint": str(base_checkpoint.resolve()),
            "removal_checkpoint": str(target_checkpoint.resolve()),
            "endpoint_counterfactual": float(values["endpoint_contarfactual"][0]),
            "trajectory_counterfactual": float(values["traj_contarfactual"][0]),
            "target_details": {key: detail for key, (_, detail) in values.items()},
        }
        write_json(run_dir / "counterfactual_metrics.json", output)
        print(
            f"[done] {args.method} Q{args.query_id} endpoint={output['endpoint_counterfactual']:.9g} "
            f"trajectory={output['trajectory_counterfactual']:.9g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
