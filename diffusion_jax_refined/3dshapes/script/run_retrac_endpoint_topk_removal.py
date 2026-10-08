#!/usr/bin/env python3
"""Retrain after top-k removal for the 100x1 ReTrac/endpoint comparison."""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
REFINE_ROOT = SHAPES_ROOT.parent
LEGACY_ROOT = REFINE_ROOT / "legacy_jax"
for path in (SHAPES_ROOT, REFINE_ROOT, LEGACY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataset_config import _prompt_tag


METHODS = (
    "retrac_adamw_both_l2_neg",
    "endpoint_pollute_adamw_timestamp_train_l2",
)
TOPKS = (400, 1000)
FULL_TRAIN_SIZE = 20_000
ATTRIBUTED_SIZE = 5_000


def load_queries(path: Path) -> dict[int, dict]:
    payload = json.loads(path.read_text())
    records = payload["queries"] if isinstance(payload, dict) else payload
    return {int(record.get("query_id", i)): record for i, record in enumerate(records)}


def score_spec(
    *, result_root: Path, train_seed: int, record: dict, method: str
) -> tuple[Path, int, str]:
    root = (
        result_root
        / "attribution_score"
        / "prompted_solo"
        / f"train_seed_{train_seed}"
        / f"query_{_prompt_tag(str(record['prompt']))}"
        / f"initial_seed_{int(record['initial_seed'])}"
    )
    if method == "retrac_adamw_both_l2_neg":
        return (
            root
            / "traj_tracin_paper_retrac_adamw_full_exact4_endpoint100x1_q0_99"
            / "score_query_train_l2_normalized",
            -1,
            "negative of stored AdamW-full paper-ReTrac query/train-L2 score",
        )
    if method == "endpoint_pollute_adamw_timestamp_train_l2":
        return (
            root
            / (
                "traj_tracin_recreate_adamw_full_polluted_endpoint_"
                "delta_l2normalized_timestamp_sum_squared_aligned100x1_q0_99"
            )
            / "score_train_l2_normalized",
            1,
            "stored AdamW-full endpoint-pollute timestamp-square train-L2 score",
        )
    raise ValueError(method)


def select_topk(
    score_dir: Path, ranking_sign: int, topk: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score_path = score_dir / "scores.npy"
    index_path = score_dir / "score_indices.npy"
    if not score_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(f"Missing score artifact under {score_dir}")
    raw = np.asarray(np.load(score_path), dtype=np.float64).reshape(-1)
    indices = np.asarray(np.load(index_path), dtype=np.int64).reshape(-1)
    if raw.shape != indices.shape:
        raise ValueError(f"score/index mismatch in {score_dir}: {raw.shape} vs {indices.shape}")
    if len(raw) != ATTRIBUTED_SIZE:
        raise ValueError(
            f"expected {ATTRIBUTED_SIZE} attributed points in {score_dir}, found {len(raw)}"
        )
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"duplicate score indices in {score_dir}")
    ranking = ranking_sign * raw
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


def final_checkpoint(model_dir: Path, train_seed: int, epochs: int) -> Path:
    return model_dir / f"seed_{train_seed}_epoch_{epochs:04d}.ckpt"


def sample_root(result_root: Path, record: dict, train_seed: int, epochs: int) -> Path:
    return (
        result_root
        / "sample_ddim_eta0_1000"
        / "cifar"
        / f"prompt_{_prompt_tag(str(record['prompt']))}"
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
    parser.add_argument("--topk", type=int, choices=TOPKS, required=True)
    parser.add_argument("--stage", choices=("validate", "train", "eval", "all"), default="all")
    args = parser.parse_args()

    records = load_queries(args.query_file)
    if args.query_id not in records:
        raise KeyError(f"query id {args.query_id} is absent from {args.query_file}")
    record = records[args.query_id]
    result_root = SHAPES_ROOT / "result" / args.experiment
    score_dir, ranking_sign, score_definition = score_spec(
        result_root=result_root,
        train_seed=args.train_seed,
        record=record,
        method=args.method,
    )
    removed, raw_scores, ranking_scores = select_topk(
        score_dir, ranking_sign, args.topk
    )
    run_dir = (
        result_root
        / "retrac_endpoint_topk_removal_60q"
        / args.method
        / f"topk_{args.topk:04d}"
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
        "score_definition": score_definition,
        "ranking_sign": ranking_sign,
        "topk": args.topk,
        "full_training_size": FULL_TRAIN_SIZE,
        "attributed_candidate_size": ATTRIBUTED_SIZE,
        "removal_fraction_of_full_training_set": args.topk / FULL_TRAIN_SIZE,
        "removed_dataset_indices": [int(x) for x in removed],
        "removed_raw_scores": [float(x) for x in raw_scores],
        "removed_ranking_scores": [float(x) for x in ranking_scores],
    }
    if args.stage == "validate":
        print(
            f"[valid] {args.method} Q{args.query_id} topk={args.topk} "
            f"score={score_dir}",
            flush=True,
        )
        return
    write_json(metadata_path, metadata)

    base_checkpoint = (
        result_root
        / "model"
        / "prompted_jax"
        / f"seed_{args.train_seed}_epoch_{args.epochs:04d}.ckpt"
    )
    if not base_checkpoint.is_file():
        raise FileNotFoundError(base_checkpoint)

    target_checkpoint = final_checkpoint(model_dir, args.train_seed, args.epochs)
    if args.stage in ("train", "all"):
        if target_checkpoint.is_file():
            print(f"[skip] final checkpoint exists: {target_checkpoint}", flush=True)
        else:
            from DM__training_CIFAR5_MULTI_pixel import train

            cfg = load_training_config(base_checkpoint, model_dir, removed)
            metadata["train_config"] = asdict(cfg)
            write_json(metadata_path, metadata)
            print(
                f"[train] {args.method} Q{args.query_id} remove top {args.topk} "
                f"({args.topk / FULL_TRAIN_SIZE:.1%} of full set)",
                flush=True,
            )
            train(cfg)
            del cfg
            gc.collect()
            if not target_checkpoint.is_file():
                raise FileNotFoundError(
                    f"training completed without final checkpoint {target_checkpoint}"
                )

    if args.stage in ("eval", "all"):
        metrics_path = run_dir / "counterfactual_metrics.json"
        if metrics_path.is_file():
            print(f"[skip] metrics exist: {metrics_path}", flush=True)
            return
        if not target_checkpoint.is_file():
            raise FileNotFoundError(target_checkpoint)
        from LDS.DM_cifar_lds import CifarTargetEvaluator

        evaluator = CifarTargetEvaluator(
            code_file=str(LEGACY_ROOT / "DM__training_3DSHAPES_pixel.py"),
            base_checkpoint=str(base_checkpoint),
            prompt=str(record["prompt"]),
            prefer_device="gpu",
            data_root=str(REFINE_ROOT / "dataset" / "3dshapes" / "20000"),
            target_function="traj_counterfactual",
            sample_root=str(
                sample_root(result_root, record, args.train_seed, args.epochs)
            ),
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
            "endpoint_difference": float(values["endpoint_contarfactual"][0]),
            "trajectory_difference": float(values["traj_contarfactual"][0]),
            "target_details": {key: detail for key, (_, detail) in values.items()},
        }
        write_json(metrics_path, output)
        print(
            f"[done] {args.method} Q{args.query_id} topk={args.topk} "
            f"endpoint={output['endpoint_difference']:.9g} "
            f"trajectory={output['trajectory_difference']:.9g} "
            f"checkpoint={target_checkpoint}",
            flush=True,
        )


if __name__ == "__main__":
    main()
