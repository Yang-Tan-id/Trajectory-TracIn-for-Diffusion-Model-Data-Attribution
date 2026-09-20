#!/usr/bin/env python3
"""Combine cached reference/own 100-timestamp directions at the score-term level."""
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
from run_adamw_four_event_original_f_scores import event
from run_expected_residual_jacobian_scores import load_query_bank


METHODS = ("four", "e1", "four_residual", "e1_residual")
REDUCTIONS = ("linear_sum", "square_sum", "absolute_sum")
VARIANTS = ("raw", "query_l2", "train_l2", "query_train_l2")
TARGETS = (
    "endpoint_contarfactual",
    "traj_contarfactual",
    "simple_loss",
    "noise_trajectory",
)


def atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_bank(args, namespace):
    class QueryArgs:
        pass

    query_args = QueryArgs()
    query_args.experiment = args.experiment
    query_args.train_seed = args.train_seed
    query_args.epochs = args.epochs
    query, metadata = load_query_bank(
        query_args,
        namespace,
        "trajectory_next_checkpoint_noise_mse",
        range(10),
    )
    lookup = {
        (int(checkpoint), int(timestep)): term
        for term, (checkpoint, timestep) in enumerate(
            zip(metadata["ckpt_indices"], metadata["timesteps"])
        )
    }
    term_ids = np.empty((49, 100), dtype=np.int64)
    timestep_values = []
    for checkpoint in range(49):
        checkpoint_timesteps = sorted(
            timestep for (found_checkpoint, timestep) in lookup if found_checkpoint == checkpoint
        )
        if len(checkpoint_timesteps) != 100:
            raise ValueError(
                f"{namespace}: checkpoint {checkpoint} expected 100 timestamps, "
                f"got {len(checkpoint_timesteps)}"
            )
        if checkpoint == 0:
            timestep_values = checkpoint_timesteps
        elif checkpoint_timesteps != timestep_values:
            raise ValueError(f"{namespace}: timestamp grid changes at checkpoint {checkpoint}")
        term_ids[checkpoint] = [lookup[(checkpoint, timestep)] for timestep in timestep_values]
    if query.shape != (10, 4900, 4096):
        raise ValueError(f"{namespace}: expected (10,4900,4096), got {query.shape}")
    return query, term_ids, np.asarray(timestep_values, dtype=np.int32)


def build_template(args):
    module = importlib.import_module("DM__training_CIFAR5_MULTI_pixel")
    checkpoint_root = ROOT / "result" / args.experiment / "model" / "prompted_jax"
    with (checkpoint_root / f"seed_{args.train_seed}_epoch_0004.ckpt").open("rb") as handle:
        payload = pickle.load(handle)
    cfg = module.TrainConfig(**dict(payload["config"]))
    device = module.choose_devices(args.device)[0]
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
    template = module.create_train_state(
        cfg,
        module.build_model(cfg),
        jax.random.PRNGKey(cfg.seed),
        device,
        (len(dataset) // cfg.batch_size) * cfg.epochs,
    )
    return module, checkpoint_root, template


@jax.jit
def checkpoint_scores(train, reference_query, own_query):
    # train: M,N,D; query: Q,T,D
    method_count, datapoints, dimension = train.shape
    queries, timestamps, query_dimension = reference_query.shape
    if dimension != query_dimension:
        raise ValueError("train/query projection dimensions differ")
    train_flat = train.reshape(method_count * datapoints, dimension)
    reference_flat = reference_query.reshape(queries * timestamps, dimension)
    own_flat = own_query.reshape(queries * timestamps, dimension)
    reference_dot = (train_flat @ reference_flat.T).reshape(
        method_count, datapoints, queries, timestamps
    )
    own_dot = (train_flat @ own_flat.T).reshape(
        method_count, datapoints, queries, timestamps
    )
    train_norm = jnp.linalg.norm(train, axis=2)[:, :, None, None] + 1e-8
    reference_norm = jnp.linalg.norm(reference_query, axis=2)[None, None, :, :] + 1e-8
    own_norm = jnp.linalg.norm(own_query, axis=2)[None, None, :, :] + 1e-8

    reference_variants = jnp.stack(
        (
            reference_dot,
            reference_dot / reference_norm,
            reference_dot / train_norm,
            reference_dot / reference_norm / train_norm,
        ),
        axis=0,
    )
    own_variants = jnp.stack(
        (
            own_dot,
            own_dot / own_norm,
            own_dot / train_norm,
            own_dot / own_norm / train_norm,
        ),
        axis=0,
    )
    reductions = jnp.stack(
        (
            reference_variants + own_variants,
            jnp.square(reference_variants) + jnp.square(own_variants),
            jnp.abs(reference_variants) + jnp.abs(own_variants),
        ),
        axis=0,
    )
    # R,V,M,N,Q,T -> R,M,V,Q,N, averaged over the 100 timestamps.
    return jnp.transpose(jnp.mean(reductions, axis=-1), (0, 2, 1, 4, 3))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--attribution-points", type=int, default=5000)
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default="cpu",
        help="Execution device. CPU can run directly without a GPU allocation.",
    )
    parser.add_argument(
        "--reference-namespace",
        default="loss_direction_original_f_reference_trajectory_100t",
    )
    parser.add_argument(
        "--own-namespace",
        default="loss_direction_original_f_checkpoint_own_trajectory_100t",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    reference, reference_terms, reference_timesteps = load_bank(
        args, args.reference_namespace
    )
    own, own_terms, own_timesteps = load_bank(args, args.own_namespace)
    if not np.array_equal(reference_timesteps, own_timesteps):
        raise ValueError("reference and own timestamp grids differ")

    module, checkpoint_root, template = build_template(args)
    from dtrak.algorithm import _countsketch_project_grad_jax

    artifact_root = (
        ROOT
        / "result"
        / args.experiment
        / f"fixed_checkpoint_adamw_four_events_n{args.attribution_points}"
    )
    # R,M,V,Q,N
    scores = np.zeros(
        (
            len(REDUCTIONS),
            len(METHODS),
            len(VARIANTS),
            10,
            args.attribution_points,
        ),
        dtype=np.float64,
    )
    score_indices = None

    for checkpoint in range(49):
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
        events = []
        for epoch in range(start_epoch + 1, start_epoch + 5):
            feature, indices = event(
                artifact_root / f"epoch_{start_epoch}_{start_epoch + 4}", epoch
            )
            if score_indices is None:
                score_indices = indices
            elif not np.array_equal(score_indices, indices):
                raise ValueError(f"dataset-index mismatch at checkpoint {checkpoint}")
            events.append(feature)
        train = np.stack(
            (
                sum(events),
                events[0],
                sum(feature - history for feature in events),
                events[0] - history,
            ),
            axis=0,
        )
        reference_checkpoint = reference[:, reference_terms[checkpoint], :]
        own_checkpoint = own[:, own_terms[checkpoint], :]
        value = checkpoint_scores(
            jax.device_put(jnp.asarray(train, dtype=jnp.float32)),
            jax.device_put(jnp.asarray(reference_checkpoint, dtype=jnp.float32)),
            jax.device_put(jnp.asarray(own_checkpoint, dtype=jnp.float32)),
        )
        scores += np.asarray(jax.device_get(value), dtype=np.float64)
        print(f"[combine] checkpoint={checkpoint + 1}/49", flush=True)

    if score_indices is None:
        raise RuntimeError("no train score indices loaded")
    atomic_savez(
        args.out_dir / "combined_scores.npz",
        scores=scores,
        score_indices=score_indices,
        reductions=np.asarray(REDUCTIONS),
        methods=np.asarray(METHODS),
        variants=np.asarray(VARIANTS),
        reference_namespace=np.asarray(args.reference_namespace),
        own_namespace=np.asarray(args.own_namespace),
        timesteps=reference_timesteps,
    )

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
        for reduction_index, reduction in enumerate(REDUCTIONS):
            for method_index, method in enumerate(METHODS):
                for variant_index, variant in enumerate(VARIANTS):
                    prediction = scores[
                        reduction_index, method_index, variant_index, query_id
                    ] @ incidence.T
                    for target in TARGETS:
                        lds = 100.0 * float(
                            rowwise_spearman(prediction[None, :], true_values[target])[0]
                        )
                        rows.append(
                            {
                                "reduction": reduction,
                                "method": method,
                                "variant": variant,
                                "query": query_id,
                                "target": target,
                                "lds_percent": lds,
                                "prediction_sign": "p1",
                                "checkpoint_weighting": "uniform",
                                "query_directions": "reference_plus_own_100t",
                            }
                        )
    write_csv(args.out_dir / "per_query.csv", rows)

    for reduction in REDUCTIONS:
        print(f"\n{reduction.upper()} — REFERENCE + OWN 100 TIMESTAMPS — FIXED P1")
        print("METHOD             TARGET                         RAW   QUERY-L2   TRAIN-L2    BOTH-L2")
        print("-" * 94)
        for method in METHODS:
            for target in TARGETS:
                means = []
                for variant in VARIANTS:
                    selected = [
                        float(row["lds_percent"])
                        for row in rows
                        if row["reduction"] == reduction
                        and row["method"] == method
                        and row["variant"] == variant
                        and row["target"] == target
                    ]
                    means.append(float(np.mean(selected)))
                print(
                    f"{method:18s} {target:24s} "
                    + " ".join(f"{value:+9.3f}%" for value in means)
                )
    print(f"[saved] {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
