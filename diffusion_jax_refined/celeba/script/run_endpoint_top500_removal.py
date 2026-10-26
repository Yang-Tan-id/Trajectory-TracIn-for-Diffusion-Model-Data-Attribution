#!/usr/bin/env python3
"""Retrain CelebA after removing score-ranked attribution candidates.

This worker consumes existing 50-checkpoint score banks for:

* endpoint-pollute AdamW timestamp-sum-squared, raw, query L2, or query+train L2;
* endpoint-TracIn AdamW linear, raw, query L2, or query+train L2;
* paper-ReTrac AdamW query+train L2.

Query-L2 and query+train-L2 score banks covering the complete 10,000-example
training set are separate method choices, so they cannot be confused with the
historical 5,000-candidate experiments.

The ranking orientation is part of each method contract and is recorded in every
removal manifest.  It is chosen from the sign=-1 LDS convention used by the
cached CelebA evaluation: positive LDS removes the largest stored scores, while
negative LDS removes the smallest stored scores.
"""

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


CELEBA_ROOT = Path(__file__).resolve().parents[1]
REFINED_ROOT = CELEBA_ROOT.parent
LEGACY_ROOT = REFINED_ROOT / "legacy_jax"
for path in (CELEBA_ROOT, REFINED_ROOT, LEGACY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataset_config_attr40 import CLASS_NAMES, DATA_ROOT, _prompt_path_tag


METHODS = (
    "random_full10k_seeded",
    "endpoint_pollute_adamw_timestamp_raw_pos",
    "endpoint_pollute_adamw_timestamp_query_l2_pos",
    "endpoint_pollute_adamw_timestamp_query_l2_full10k_pos",
    "endpoint_pollute_adamw_timestamp_both_l2_pos",
    "endpoint_pollute_adamw_timestamp_both_l2_full10k_pos",
    "endpoint_tracin_adamw_linear_raw_neg",
    "endpoint_tracin_adamw_linear_raw_full10k_neg",
    "endpoint_tracin_adamw_linear_query_l2_neg",
    "endpoint_tracin_adamw_linear_query_l2_full10k_neg",
    "endpoint_tracin_adamw_linear_both_l2_neg",
    "endpoint_tracin_adamw_linear_both_l2_full10k_neg",
    "retrac_adamw_query_train_l2_neg",
    "retrac_adamw_query_train_l2_full10k_neg",
)
DEFAULT_TOPK = 500
FULL_TRAIN_SIZE = 10_000
ATTRIBUTED_SIZE = 5_000
DEFAULT_RANDOM_SEED = 2_026


def load_queries(path: Path) -> dict[int, dict]:
    payload = json.loads(path.read_text())
    records = payload["queries"] if isinstance(payload, dict) else payload
    output: dict[int, dict] = {}
    for position, record in enumerate(records):
        query_id = int(record.get("query_id", record.get("id", position)))
        output[query_id] = record
    return output


def query_seed(record: dict) -> int:
    if "initial_seed" in record:
        return int(record["initial_seed"])
    if "seed" in record:
        return int(record["seed"])
    raise KeyError("query record has neither initial_seed nor seed")


def score_spec(
    *, result_root: Path, train_seed: int, record: dict, method: str
) -> tuple[Path, int, str, float]:
    root = (
        result_root
        / "attribution_score"
        / "prompted_solo"
        / f"train_seed_{train_seed}"
        / f"query_{_prompt_path_tag(str(record['prompt']))}"
        / f"initial_seed_{query_seed(record)}"
    )
    pollute_namespace = (
        "celeba_endpoint_pollute_adamw_full_train1x50_query1x50_"
        "timestamp_sum_squared_shape3d_semantics"
    )
    tracin_namespace = (
        "celeba_endpoint_tracin_adamw_full_"
        "train_mean50_query_mean50_linear_shape3d_semantics"
    )
    retrac_namespace = (
        "celeba_paper_retrac_adamw_full_train_replay4_batch1_"
        "query_mean50_of_full_l2_linear_shape3d_semantics"
    )
    if method == "endpoint_pollute_adamw_timestamp_raw_pos":
        return (
            root / f"traj_tracin_{pollute_namespace}" / "score",
            +1,
            "stored endpoint-pollute timestamp-sum-squared raw score descending",
            +1.0,
        )
    if method == "endpoint_pollute_adamw_timestamp_query_l2_pos":
        return (
            root
            / f"traj_tracin_{pollute_namespace}"
            / "score_query_normalized",
            +1,
            (
                "stored endpoint-pollute timestamp-sum-squared query-L2 "
                "score descending"
            ),
            +1.0,
        )
    if method == "endpoint_pollute_adamw_timestamp_query_l2_full10k_pos":
        return (
            root
            / f"traj_tracin_{pollute_namespace}_full10000"
            / "score_query_normalized",
            +1,
            (
                "stored full-10000 endpoint-pollute timestamp-sum-squared "
                "query-L2 score descending"
            ),
            +1.0,
        )
    if method == "endpoint_pollute_adamw_timestamp_both_l2_pos":
        return (
            root
            / f"traj_tracin_{pollute_namespace}"
            / "score_query_train_l2_normalized",
            +1,
            (
                "stored endpoint-pollute timestamp-sum-squared query+train-L2 "
                "score descending"
            ),
            +1.0,
        )
    if method == "endpoint_pollute_adamw_timestamp_both_l2_full10k_pos":
        return (
            root
            / f"traj_tracin_{pollute_namespace}_full10000"
            / "score_query_train_l2_normalized",
            +1,
            (
                "stored full-10000 endpoint-pollute timestamp-sum-squared "
                "query+train-L2 score descending"
            ),
            +1.0,
        )
    if method == "endpoint_tracin_adamw_linear_raw_neg":
        return (
            root / f"traj_tracin_{tracin_namespace}" / "score",
            -1,
            "stored endpoint-TracIn linear raw score ascending",
            -1.0,
        )
    if method == "endpoint_tracin_adamw_linear_raw_full10k_neg":
        return (
            root / f"traj_tracin_{tracin_namespace}_full10000" / "score",
            -1,
            "stored full-10000 endpoint-TracIn linear raw score ascending",
            -1.0,
        )
    if method == "endpoint_tracin_adamw_linear_query_l2_neg":
        return (
            root
            / f"traj_tracin_{tracin_namespace}"
            / "score_query_normalized",
            -1,
            "stored endpoint-TracIn linear query-L2 score ascending",
            -1.0,
        )
    if method == "endpoint_tracin_adamw_linear_query_l2_full10k_neg":
        return (
            root
            / f"traj_tracin_{tracin_namespace}_full10000"
            / "score_query_normalized",
            -1,
            "stored full-10000 endpoint-TracIn linear query-L2 score ascending",
            -1.0,
        )
    if method == "endpoint_tracin_adamw_linear_both_l2_neg":
        return (
            root
            / f"traj_tracin_{tracin_namespace}"
            / "score_query_train_l2_normalized",
            -1,
            "stored endpoint-TracIn linear query+train-L2 score ascending",
            -1.0,
        )
    if method == "endpoint_tracin_adamw_linear_both_l2_full10k_neg":
        return (
            root
            / f"traj_tracin_{tracin_namespace}_full10000"
            / "score_query_train_l2_normalized",
            -1,
            "stored full-10000 endpoint-TracIn linear query+train-L2 score ascending",
            -1.0,
        )
    if method == "retrac_adamw_query_train_l2_neg":
        return (
            root
            / f"traj_tracin_{retrac_namespace}"
            / "score_query_train_l2_normalized",
            -1,
            (
                "stored paper-ReTrac AdamW-full replay4 batch1 query+train-L2 "
                "linear score ascending"
            ),
            -1.0,
        )
    if method == "retrac_adamw_query_train_l2_full10k_neg":
        return (
            root
            / f"traj_tracin_{retrac_namespace}_full10000"
            / "score_query_train_l2_normalized",
            -1,
            (
                "stored full-10000 paper-ReTrac AdamW-full replay4 batch1 "
                "query+train-L2 linear score ascending"
            ),
            -1.0,
        )
    raise ValueError(method)


def select_topk(
    score_dir: Path, ranking_sign: int, topk: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    score_path = score_dir / "scores.npy"
    index_path = score_dir / "score_indices.npy"
    if not score_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(f"missing score artifact under {score_dir}")
    stored = np.asarray(np.load(score_path), dtype=np.float64).reshape(-1)
    indices = np.asarray(np.load(index_path), dtype=np.int64).reshape(-1)
    if stored.shape != indices.shape:
        raise ValueError(f"score/index mismatch: {stored.shape} vs {indices.shape}")
    if len(stored) not in (ATTRIBUTED_SIZE, FULL_TRAIN_SIZE):
        raise ValueError(
            f"expected {ATTRIBUTED_SIZE} or {FULL_TRAIN_SIZE} scores, "
            f"found {len(stored)}"
        )
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"duplicate candidate indices under {score_dir}")
    if np.any(indices < 0) or np.any(indices >= FULL_TRAIN_SIZE):
        raise ValueError(f"candidate index outside [0, {FULL_TRAIN_SIZE})")
    ranking = ranking_sign * stored
    ranking = np.where(np.isfinite(ranking), ranking, -np.inf)
    order = np.argsort(-ranking, kind="stable")[:topk]
    if len(order) != topk or np.any(~np.isfinite(ranking[order])):
        raise ValueError(f"fewer than {topk} finite scores under {score_dir}")
    return indices[order], stored[order], ranking[order], len(stored)


def select_random_removal(
    *, topk: int, random_seed: int, query_id: int
) -> np.ndarray:
    """Select a deterministic, query-specific random subset of the full set."""
    if topk <= 0 or topk > FULL_TRAIN_SIZE:
        raise ValueError(f"topk must be in [1, {FULL_TRAIN_SIZE}]")
    if random_seed < 0:
        raise ValueError("random_seed must be non-negative")
    if query_id < 0:
        raise ValueError("query_id must be non-negative")
    seed_sequence = np.random.SeedSequence([int(random_seed), int(query_id)])
    generator = np.random.default_rng(seed_sequence)
    selected = generator.choice(FULL_TRAIN_SIZE, size=topk, replace=False)
    return np.sort(np.asarray(selected, dtype=np.int64))


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def load_training_config(
    base_checkpoint: Path,
    output_dir: Path,
    removed: np.ndarray,
    *,
    train_seed: int,
    epochs: int,
):
    from DM__training_CELEBA_pixel import TrainConfig

    with base_checkpoint.open("rb") as handle:
        payload = pickle.load(handle)
    saved = dict(payload.get("config", {}))
    valid = set(TrainConfig.__dataclass_fields__)
    cfg = TrainConfig(**{key: value for key, value in saved.items() if key in valid})
    cfg.data_root = DATA_ROOT
    cfg.class_names = tuple(CLASS_NAMES)
    cfg.resume_from = None
    cfg.exclude_ranges = None
    cfg.exclude_indices = {0: tuple(int(x) for x in removed.tolist())}
    cfg.checkpoint_dir = str(output_dir)
    cfg.seed = train_seed
    cfg.epochs = epochs
    cfg.save_every_epochs = epochs
    cfg.keep_last_k = 1
    cfg.prefer_device = "gpu"
    cfg.use_data_parallel = False
    cfg.num_devices = 1
    cfg.use_wandb = False
    cfg.num_workers = 0
    cfg.batch_size = int(os.environ.get("JAX_BATCH_SIZE", cfg.batch_size))
    cfg.prefetch_size = int(os.environ.get("JAX_PREFETCH_SIZE", "1"))
    cfg.use_bfloat16 = env_bool("JAX_BFLOAT16", cfg.use_bfloat16)
    return cfg


def final_checkpoint(model_dir: Path, train_seed: int, epochs: int) -> Path:
    return model_dir / f"seed_{train_seed}_epoch_{epochs:04d}.ckpt"


def sample_root(result_root: Path, record: dict, checkpoint: Path) -> Path:
    return (
        result_root
        / "sample_ddim_eta0_50"
        / "cifar"
        / f"prompt_{_prompt_path_tag(str(record['prompt']))}"
        / f"model_prompted_solo__ckpt_{checkpoint.stem}"
    )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--query-id", type=int, required=True)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--experiment", default="attr40_experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=(
            "Base seed for random_full10k_seeded. The actual deterministic "
            "stream is keyed by this value and --query-id."
        ),
    )
    parser.add_argument(
        "--output-group",
        help=(
            "Result directory name below the experiment root. The default keeps "
            "the historical q0_9 layout."
        ),
    )
    parser.add_argument(
        "--stage", choices=("validate", "train", "eval", "all"), default="all"
    )
    args = parser.parse_args()
    if args.topk <= 0 or args.topk > ATTRIBUTED_SIZE:
        parser.error(f"--topk must be in [1, {ATTRIBUTED_SIZE}]")
    output_group = args.output_group or f"endpoint_top{args.topk}_removal_q0_9"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", output_group):
        parser.error("--output-group may contain only letters, digits, '.', '_', '-'")

    records = load_queries(args.query_file)
    if args.query_id not in records:
        raise KeyError(f"query {args.query_id} is absent from {args.query_file}")
    record = records[args.query_id]
    initial_seed = query_seed(record)
    result_root = CELEBA_ROOT / "result" / args.experiment
    if args.method == "random_full10k_seeded":
        if args.random_seed < 0:
            parser.error("--random-seed must be non-negative")
        score_dir = None
        ranking_sign = 0
        score_definition = (
            "deterministic query-specific uniform random removal without "
            "replacement from all 10000 training examples"
        )
        cached_lds_sign = None
        removed = select_random_removal(
            topk=args.topk,
            random_seed=args.random_seed,
            query_id=args.query_id,
        )
        raw_scores = np.asarray([], dtype=np.float64)
        ranking_scores = np.asarray([], dtype=np.float64)
        candidate_size = FULL_TRAIN_SIZE
    else:
        score_dir, ranking_sign, score_definition, cached_lds_sign = score_spec(
            result_root=result_root,
            train_seed=args.train_seed,
            record=record,
            method=args.method,
        )
        removed, raw_scores, ranking_scores, candidate_size = select_topk(
            score_dir, ranking_sign, args.topk
        )
    if "full10k" in args.method and candidate_size != FULL_TRAIN_SIZE:
        raise ValueError(
            f"{args.method} requires {FULL_TRAIN_SIZE} candidates, "
            f"but {score_dir} contains {candidate_size}"
        )
    base_checkpoint = (
        result_root
        / "model"
        / "prompted_jax"
        / f"seed_{args.train_seed}_epoch_{args.epochs:04d}.ckpt"
    )
    reference_root = sample_root(result_root, record, base_checkpoint)
    reference_seed = reference_root / f"seed_{initial_seed:06d}"
    required_inputs = (
        base_checkpoint,
        Path(DATA_ROOT) / "dataset.npz",
        reference_seed / "trajectory_xt.npy",
        reference_seed / "trajectory_t.npy",
    )
    missing = [path for path in required_inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing input(s): " + ", ".join(map(str, missing)))

    run_dir = (
        result_root
        / output_group
        / args.method
        / f"query_{args.query_id:03d}_seed_{initial_seed:03d}"
    )
    model_dir = run_dir / "model"
    manifest_path = run_dir / "removal_manifest.json"
    metadata = {
        "method": args.method,
        "output_group": output_group,
        "query_id": args.query_id,
        "prompt": str(record["prompt"]),
        "initial_seed": initial_seed,
        "score_directory": str(score_dir.resolve()) if score_dir is not None else None,
        "score_definition": score_definition,
        "ranking_sign": ranking_sign,
        "ranking_order": (
            "uniform_random_without_replacement"
            if ranking_sign == 0
            else (
                "stored_score_descending"
                if ranking_sign == 1
                else "stored_score_ascending"
            )
        ),
        "cached_lds_prediction_sign": None
        if args.method == "random_full10k_seeded"
        else -1,
        "cached_lds_direction": cached_lds_sign,
        "random_seed_base": args.random_seed
        if args.method == "random_full10k_seeded"
        else None,
        "random_seed_key": [args.random_seed, args.query_id]
        if args.method == "random_full10k_seeded"
        else None,
        "random_generator": "numpy.random.PCG64"
        if args.method == "random_full10k_seeded"
        else None,
        "topk": args.topk,
        "full_training_size": FULL_TRAIN_SIZE,
        "attributed_candidate_size": candidate_size,
        "removal_fraction_of_full_training_set": args.topk / FULL_TRAIN_SIZE,
        "removal_fraction_of_attributed_candidates": args.topk / candidate_size,
        "removed_dataset_indices": [int(value) for value in removed],
        "removed_raw_scores": [float(value) for value in raw_scores],
        "removed_ranking_scores": [float(value) for value in ranking_scores],
    }

    if args.stage == "validate":
        from DM__training_CELEBA_pixel import TrainConfig

        if "exclude_indices" not in TrainConfig.__dataclass_fields__:
            raise RuntimeError("CelebA TrainConfig lacks exclude_indices support")
        print(
            f"[valid] method={args.method} query={args.query_id} topk={args.topk} "
            f"ranking_sign={ranking_sign:+d} order={metadata['ranking_order']} "
            f"score={score_dir if score_dir is not None else 'NONE'}",
            flush=True,
        )
        return

    write_json(manifest_path, metadata)
    target_checkpoint = final_checkpoint(model_dir, args.train_seed, args.epochs)
    if args.stage in ("train", "all"):
        if target_checkpoint.is_file():
            print(f"[skip train] {target_checkpoint}", flush=True)
        else:
            from DM__training_CELEBA_pixel import train

            cfg = load_training_config(
                base_checkpoint,
                model_dir,
                removed,
                train_seed=args.train_seed,
                epochs=args.epochs,
            )
            metadata["train_config"] = asdict(cfg)
            write_json(manifest_path, metadata)
            print(
                f"[train] method={args.method} Q{args.query_id} remove={args.topk} "
                f"order={metadata['ranking_order']} batch_size={cfg.batch_size}",
                flush=True,
            )
            train(cfg)
            del cfg
            gc.collect()
            if not target_checkpoint.is_file():
                raise FileNotFoundError(
                    f"training finished without final checkpoint {target_checkpoint}"
                )

    if args.stage in ("eval", "all"):
        metrics_path = run_dir / "counterfactual_metrics.json"
        if metrics_path.is_file():
            print(f"[skip eval] {metrics_path}", flush=True)
            return
        if not target_checkpoint.is_file():
            raise FileNotFoundError(target_checkpoint)
        from LDS.DM_cifar_lds import CifarTargetEvaluator

        evaluator = CifarTargetEvaluator(
            code_file=str(LEGACY_ROOT / "DM__training_CELEBA_pixel.py"),
            base_checkpoint=str(base_checkpoint),
            prompt=str(record["prompt"]),
            prefer_device="gpu",
            data_root=DATA_ROOT,
            target_function="traj_counterfactual",
            sample_root=str(reference_root),
            sample_seed=initial_seed,
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
            f"[done] method={args.method} Q{args.query_id} topk={args.topk} "
            f"endpoint={output['endpoint_difference']:.9g} "
            f"trajectory={output['trajectory_difference']:.9g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
