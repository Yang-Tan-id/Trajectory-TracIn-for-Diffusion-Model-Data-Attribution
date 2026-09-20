#!/usr/bin/env python3
"""Strictly timestamp-aligned AdamW event scores against own-trajectory next deltas."""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import pickle
import sys
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent / "legacy_jax")]

from analyze_predicted_noise_probe8_choose4 import cache_group, load_target_data
from analyze_predicted_noise_probe12_sign_flips import rowwise_spearman
from dataset_config import _prompt_tag
from run_expected_residual_jacobian_scores import load_query_bank


EVENT_METHODS = ("e1", "e2", "e3", "e4", "four")
METHODS = EVENT_METHODS + tuple(f"{name}_residual" for name in EVENT_METHODS)
VARIANTS = ("raw", "query_l2", "train_l2", "query_train_l2")
CONTRACTIONS = ("linear", "squared")
TARGETS = (
    "endpoint_contarfactual",
    "traj_contarfactual",
    "simple_loss",
    "noise_trajectory",
)


def event(root: Path, epoch: int, shards: int = 2):
    features = []
    indices = []
    timesteps = []
    for shard in range(shards):
        path = root / f"event_gradient_epoch_{epoch:04d}_shard_{shard:02d}_of_{shards:02d}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            features.append(np.asarray(payload["train_features"], dtype=np.float32))
            indices.append(np.asarray(payload["dataset_indices"], dtype=np.int64))
            timesteps.append(np.asarray(payload["timesteps"], dtype=np.int32))
    index = np.concatenate(indices)
    if len(np.unique(index)) != len(index):
        raise ValueError(f"duplicate dataset indices in {root}")
    return np.concatenate(features), index, np.concatenate(timesteps)


def atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def build_template(args):
    module = importlib.import_module("DM__training_CIFAR5_MULTI_pixel")
    checkpoint_root = ROOT / "result" / args.experiment / "model" / "prompted_jax"
    with (checkpoint_root / f"seed_{args.train_seed}_epoch_0004.ckpt").open("rb") as handle:
        payload = pickle.load(handle)
    cfg = module.TrainConfig(**dict(payload["config"]))
    device = module.choose_devices("gpu")[0]
    dataset = module.CIFAR10Dataset(
        root=cfg.data_root,
        batch_names=cfg.batch_names,
        use_test=cfg.use_test,
        class_names=cfg.class_names,
        normalize="minus_one_to_one",
        channels_last=True,
        exclude_ranges=cfg.exclude_ranges,
        exclude_indices=cfg.exclude_indices,
        cond_mode=cfg.cond_mode,
    )
    cfg = module.TrainConfig(**{**asdict(cfg), "num_classes": len(dataset.label_names)})
    model = module.build_model(cfg)
    template = module.create_train_state(
        cfg,
        model,
        jax.random.PRNGKey(cfg.seed),
        device,
        (len(dataset) // cfg.batch_size) * cfg.epochs,
    )
    return module, checkpoint_root, template


def query_bank(args):
    class QueryArgs:
        pass

    query_args = QueryArgs()
    query_args.experiment = args.experiment
    query_args.train_seed = args.train_seed
    query_args.epochs = args.epochs
    query, metadata = load_query_bank(
        query_args,
        args.query_namespace,
        "trajectory_next_checkpoint_noise_mse",
        range(10),
    )
    lookup = {
        (int(checkpoint), int(timestep)): term
        for term, (checkpoint, timestep) in enumerate(
            zip(metadata["ckpt_indices"], metadata["timesteps"])
        )
    }
    term_ids = np.empty((49, 1000), dtype=np.int64)
    for checkpoint in range(49):
        for timestep in range(1000):
            term = lookup.get((checkpoint, timestep))
            if term is None:
                raise ValueError(
                    f"missing query term checkpoint={checkpoint}, timestep={timestep}"
                )
            term_ids[checkpoint, timestep] = term
    if query.shape != (10, 49000, 4096):
        raise ValueError(f"expected query bank (10,49000,4096), got {query.shape}")
    return query, term_ids


def aligned_event_values(train, timesteps, query_checkpoint):
    train_device = jax.device_put(jnp.asarray(train, dtype=jnp.float32))
    timestep_device = jax.device_put(jnp.asarray(timesteps, dtype=jnp.int32))
    query_device = jax.device_put(jnp.asarray(query_checkpoint, dtype=jnp.float32))
    matched_query = jnp.take(query_device, timestep_device, axis=1)
    dots = jnp.einsum("nd,qnd->qn", train_device, matched_query)
    train_norm = jnp.linalg.norm(train_device, axis=1)[None, :] + 1e-8
    query_norm = jnp.linalg.norm(matched_query, axis=2) + 1e-8
    values = jnp.stack(
        (
            dots,
            dots / query_norm,
            dots / train_norm,
            dots / query_norm / train_norm,
        ),
        axis=0,
    )
    return np.asarray(jax.device_get(values), dtype=np.float64)


def score_shard(args) -> None:
    output = args.out_dir / "shards" / f"shard_{args.shard_index:02d}.npz"
    if output.is_file():
        print(f"[skip] aligned score shard exists: {output}", flush=True)
        return

    from dtrak.algorithm import _countsketch_project_grad_jax

    module, checkpoint_root, template = build_template(args)
    query, term_ids = query_bank(args)
    artifact_root = (
        ROOT
        / "result"
        / args.experiment
        / f"fixed_checkpoint_adamw_four_events_n{args.attribution_points}"
    )
    # contraction, method, variant, query, datapoint
    scores = np.zeros(
        (len(CONTRACTIONS), len(METHODS), len(VARIANTS), 10, args.attribution_points),
        dtype=np.float64,
    )
    score_indices = None
    processed = []

    for checkpoint in range(args.shard_index, 49, args.shard_count):
        start_epoch = 4 * (checkpoint + 1)
        state, _ = module._restore_checkpoint(
            str(checkpoint_root / f"seed_{args.train_seed}_epoch_{start_epoch:04d}.ckpt"),
            template,
        )
        zero = jax.tree_util.tree_map(jnp.zeros_like, state.params)
        history_update, _ = state.tx.update(zero, state.opt_state, state.params)
        history = np.asarray(
            _countsketch_project_grad_jax(
                history_update,
                4096,
                seed_parts=(args.train_seed, "traj_tracin_projection", checkpoint),
            ),
            dtype=np.float32,
        )
        query_checkpoint = query[:, term_ids[checkpoint], :]
        combined = np.zeros((len(VARIANTS), 10, args.attribution_points), np.float64)
        combined_residual = np.zeros_like(combined)
        combined_squared = np.zeros_like(combined)
        combined_residual_squared = np.zeros_like(combined)

        for event_index, epoch in enumerate(range(start_epoch + 1, start_epoch + 5)):
            feature, indices, timesteps = event(
                artifact_root / f"epoch_{start_epoch}_{start_epoch + 4}", epoch
            )
            if score_indices is None:
                score_indices = indices
            elif not np.array_equal(score_indices, indices):
                raise ValueError(
                    f"dataset-index mismatch checkpoint={checkpoint}, event={event_index + 1}"
                )
            if np.any((timesteps < 0) | (timesteps >= 1000)):
                raise ValueError(f"invalid event timesteps checkpoint={checkpoint}")

            aligned = aligned_event_values(feature, timesteps, query_checkpoint)
            residual = aligned_event_values(feature - history, timesteps, query_checkpoint)
            method = event_index
            residual_method = len(EVENT_METHODS) + event_index
            scores[0, method] += aligned
            scores[1, method] += np.square(aligned)
            scores[0, residual_method] += residual
            scores[1, residual_method] += np.square(residual)
            combined += aligned
            combined_residual += residual
            combined_squared += np.square(aligned)
            combined_residual_squared += np.square(residual)

        combined_method = EVENT_METHODS.index("four")
        residual_combined_method = len(EVENT_METHODS) + combined_method
        scores[0, combined_method] += combined
        scores[1, combined_method] += combined_squared
        scores[0, residual_combined_method] += combined_residual
        scores[1, residual_combined_method] += combined_residual_squared
        processed.append(checkpoint)
        print(
            f"[aligned shard {args.shard_index}/{args.shard_count}] "
            f"checkpoint={checkpoint + 1}/49",
            flush=True,
        )

    if score_indices is None:
        raise RuntimeError("shard processed no checkpoints")
    atomic_savez(
        output,
        scores=scores,
        score_indices=score_indices,
        processed_checkpoints=np.asarray(processed, dtype=np.int32),
        methods=np.asarray(METHODS),
        variants=np.asarray(VARIANTS),
        contractions=np.asarray(CONTRACTIONS),
    )
    print(f"[saved] {output}", flush=True)


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def merge(args) -> None:
    scores = None
    score_indices = None
    checkpoints = []
    for shard in range(args.shard_count):
        path = args.out_dir / "shards" / f"shard_{shard:02d}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            shard_scores = np.asarray(payload["scores"], dtype=np.float64)
            indices = np.asarray(payload["score_indices"], dtype=np.int64)
            checkpoints.extend(np.asarray(payload["processed_checkpoints"], dtype=np.int32))
        scores = shard_scores if scores is None else scores + shard_scores
        if score_indices is None:
            score_indices = indices
        elif not np.array_equal(score_indices, indices):
            raise ValueError(f"score-index mismatch in {path}")
    if sorted(checkpoints) != list(range(49)):
        raise ValueError(f"expected checkpoints 0..48, got {sorted(checkpoints)}")
    assert scores is not None and score_indices is not None

    records = json.loads((ROOT / "queries_seed_0_9.json").read_text())["queries"]
    rows = []
    for query_id, record in enumerate(records):
        eval_root = (
            ROOT
            / "result"
            / args.experiment
            / "eval"
            / "prompted_solo"
            / f"query_{_prompt_tag(str(record['prompt']))}"
            / f"initial_seed_{int(record['initial_seed'])}"
        )
        incidence, true_values = load_target_data(cache_group(eval_root), score_indices)
        for contraction_index, contraction in enumerate(CONTRACTIONS):
            for method_index, method in enumerate(METHODS):
                for variant_index, variant in enumerate(VARIANTS):
                    prediction = scores[
                        contraction_index, method_index, variant_index, query_id
                    ] @ incidence.T
                    for target in TARGETS:
                        lds = 100.0 * float(
                            rowwise_spearman(prediction[None, :], true_values[target])[0]
                        )
                        rows.append(
                            {
                                "contraction": contraction,
                                "method": method,
                                "variant": variant,
                                "query": query_id,
                                "target": target,
                                "lds_percent": lds,
                                "prediction_sign": "p1",
                                "checkpoint_weighting": "uniform",
                                "alignment": "exact_event_timestamp",
                                "trajectory": "own",
                            }
                        )
    write_csv(args.out_dir / "per_query.csv", rows)

    for contraction in CONTRACTIONS:
        print(f"\n{contraction.upper()} — EXACT EVENT-TIMESTAMP ALIGNMENT — OWN TRAJECTORY")
        print("METHOD             TARGET                         RAW   QUERY-L2   TRAIN-L2    BOTH-L2")
        print("-" * 94)
        for method in METHODS:
            for target in TARGETS:
                values = []
                for variant in VARIANTS:
                    selected = [
                        float(row["lds_percent"])
                        for row in rows
                        if row["contraction"] == contraction
                        and row["method"] == method
                        and row["variant"] == variant
                        and row["target"] == target
                    ]
                    values.append(float(np.mean(selected)))
                print(
                    f"{method:18s} {target:24s} "
                    + " ".join(f"{value:+9.3f}%" for value in values)
                )
    print(f"[saved] {args.out_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("score-shard", "merge"))
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--attribution-points", type=int, default=5000)
    parser.add_argument(
        "--query-namespace",
        default="loss_direction_original_f_checkpoint_own_trajectory_1000t",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "score-shard":
        score_shard(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
