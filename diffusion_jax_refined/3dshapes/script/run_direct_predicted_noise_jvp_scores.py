#!/usr/bin/env python3
"""Stream exact parameter-direction JVP scores through reference/own trajectories."""
from __future__ import annotations

import argparse
import csv
import functools
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
LEGACY = ROOT.parent / "legacy_jax"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LEGACY))

from analyze_linear_sign_endpoint_trajectory_geometry import geometry_artifact
from dataset_config import _prompt_tag
from traj_tracin.algorithm import CIFAR10TaskAdapter


METHODS = ("four", "e1", "four_residual", "e1_residual")
REDUCTIONS = ("square", "root")
VARIANTS = ("raw", "query_l2", "train_l2", "query_train_l2")
SEMANTICS = "direct_output_jvp_adamw_direction_constant_checkpoint_weight"


def tree_add(left, right):
    return jax.tree_util.tree_map(lambda x, y: x + y, left, right)


def tree_sub(left, right):
    return jax.tree_util.tree_map(lambda x, y: x - y, left, right)


def tree_norm(tree):
    return jnp.sqrt(sum(jnp.vdot(x, x).real for x in jax.tree_util.tree_leaves(tree)))


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def key(row: dict[str, str], name: str):
    return jnp.asarray(json.loads(row[name]), dtype=jnp.uint32)


def event_lookup(path: Path, epochs: tuple[int, ...], wanted: set[int]):
    if not path.is_file():
        raise FileNotFoundError(path)
    result = {epoch: {} for epoch in epochs}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            epoch = int(row["epoch"])
            if epoch not in result:
                continue
            indices = json.loads(row["dataset_indices"])
            for position, index in enumerate(indices):
                if index in wanted:
                    result[epoch][index] = (row, position)
    for epoch in epochs:
        missing = wanted.difference(result[epoch])
        if missing:
            raise RuntimeError(f"{path}: epoch {epoch} missing {len(missing)} attribution points")
    return result


def load_trajectory_bank(args, records):
    required = (
        "checkpoint_own_trajectory_states",
        "checkpoint_own_trajectory_reference_states",
        "checkpoint_own_trajectory_state_timesteps",
    )
    banks = []
    timesteps = None
    for query in range(10):
        record = records[query]
        prompt = str(record["prompt"]).replace(",", "_")
        seed = int(record["initial_seed"])
        sample_dir = (
            ROOT / "result" / args.experiment / "sample_ddim_eta0_1000" / "cifar"
            / f"prompt_{prompt}"
            / f"model_prompted_solo__ckpt_seed_{args.train_seed}_epoch_{args.epochs:04d}"
            / f"seed_{seed:06d}"
        )
        if args.trajectory == "reference":
            trajectory_path = sample_dir / "trajectory_xt.npy"
            timestep_path = sample_dir / "trajectory_t.npy"
            if not trajectory_path.is_file() or not timestep_path.is_file():
                raise FileNotFoundError(
                    f"Q{query}: missing saved reference trajectory in {sample_dir}"
                )
            trajectory = np.load(trajectory_path)
            saved_timesteps = np.asarray(np.load(timestep_path), np.int32)
            if trajectory.ndim != 5 or trajectory.shape[0] != len(saved_timesteps):
                raise ValueError(
                    f"Q{query}: invalid saved trajectory shapes "
                    f"xt={trajectory.shape}, t={saved_timesteps.shape}"
                )
            positions = np.linspace(
                0, len(saved_timesteps) - 1, 10, dtype=np.int32
            )
            reference = np.asarray(trajectory[positions, 0], np.float32)
            current_t = saved_timesteps[positions]
            own = None
        else:
            try:
                path = geometry_artifact(args, query, required)
            except FileNotFoundError:
                # Geometry collection was introduced after several own-trajectory
                # probe runs. Search every completed query artifact before requiring
                # regeneration, rather than assuming one historical namespace.
                candidates = sorted(
                    sample_dir.parent.glob(
                        f"seed_{seed:06d}_query_gradient_*/traj_tracin/query_gradient_artifact.npz"
                    )
                )
                path = None
                for candidate in candidates:
                    with np.load(candidate, allow_pickle=False) as payload:
                        if all(name in payload.files for name in required):
                            path = candidate
                            break
                if path is None:
                    raise FileNotFoundError(
                        f"Q{query}: no completed artifact contains own-trajectory states"
                    )
            with np.load(path, allow_pickle=False) as payload:
                own = np.asarray(payload["checkpoint_own_trajectory_states"], np.float32)
                reference = np.asarray(
                    payload["checkpoint_own_trajectory_reference_states"], np.float32
                )
                current_t = np.asarray(
                    payload["checkpoint_own_trajectory_state_timesteps"], np.int32
                )
        if timesteps is None:
            timesteps = current_t
        elif not np.array_equal(timesteps, current_t):
            raise ValueError(f"Q{query}: trajectory timesteps differ")
        if args.trajectory == "reference":
            banks.append(np.broadcast_to(reference[None], (49,) + reference.shape).copy())
        else:
            assert own is not None
            if own.shape[0] < 49:
                raise ValueError(f"Q{query}: only {own.shape[0]} own trajectories")
            banks.append(own[:49])
    return np.stack(banks), np.asarray(timesteps, np.int32)


def atomic_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def setup(args):
    module = importlib.import_module("DM__training_CIFAR5_MULTI_pixel")
    ckpt_dir = ROOT / "result" / args.experiment / "model" / "prompted_jax"
    first = ckpt_dir / f"seed_{args.train_seed}_epoch_0004.ckpt"
    cfg = module.TrainConfig(**dict(load_pickle(first)["config"]))
    device = module.choose_devices(cfg.prefer_device)[0]
    ds = module.CIFAR10Dataset(
        root=cfg.data_root, batch_names=cfg.batch_names, use_test=cfg.use_test,
        class_names=cfg.class_names, normalize="minus_one_to_one", channels_last=True,
        exclude_ranges=cfg.exclude_ranges, exclude_indices=cfg.exclude_indices,
        cond_mode=cfg.cond_mode,
    )
    cfg = module.TrainConfig(**{**asdict(cfg), "num_classes": len(ds.label_names)})
    model = module.build_model(cfg)
    template = module.create_train_state(
        cfg, model, jax.random.PRNGKey(cfg.seed), device,
        (len(ds) // cfg.batch_size) * cfg.epochs,
    )
    schedule = module.make_diffusion_schedule(cfg.timesteps, cfg.beta_start, cfg.beta_end)
    return module, cfg, device, ds, model, template, schedule, ckpt_dir


def score_shard(args):
    module, cfg, device, ds, model, template, schedule, ckpt_dir = setup(args)
    adapter = CIFAR10TaskAdapter(module)
    records = json.loads((ROOT / "queries_seed_0_9.json").read_text())["queries"]
    trajectory_bank, trajectory_timesteps = load_trajectory_bank(args, records)
    query_conditions = np.concatenate(
        [np.asarray(adapter.make_query_cond(ds, record["prompt"], cfg)) for record in records],
        axis=0,
    )

    all_indices = np.asarray(
        np.random.default_rng(cfg.seed).choice(len(ds), size=args.attribution_points, replace=False),
        np.int64,
    )
    lo = len(all_indices) * args.shard_id // args.num_shards
    hi = len(all_indices) * (args.shard_id + 1) // args.num_shards
    owned = all_indices[lo:hi]
    wanted = set(map(int, owned))
    shape = (len(METHODS), len(REDUCTIONS), len(VARIANTS), 10, len(owned))
    scores = np.zeros(shape, np.float64)
    start_checkpoint = 0
    partial_dir = args.out_dir / "partials"
    shard_tag = f"shard_{args.shard_id:02d}_of_{args.num_shards:02d}"
    completed = sorted(partial_dir.glob(f"checkpoint_*_{shard_tag}.npz"))
    if completed:
        with np.load(completed[-1], allow_pickle=False) as payload:
            if not np.array_equal(np.asarray(payload["score_indices"]), owned):
                raise ValueError(f"resume indices differ in {completed[-1]}")
            scores = np.asarray(payload["scores"], np.float64)
            start_checkpoint = int(payload["completed_checkpoint"]) + 1
        print(f"[resume] shard={args.shard_id} checkpoint={start_checkpoint + 1}/49", flush=True)

    output_size = int(np.prod(trajectory_bank.shape[-3:]))
    rng = np.random.default_rng(args.output_projection_seed)
    buckets = jax.device_put(rng.integers(0, args.output_projection_dim, output_size, dtype=np.int32), device)
    signs = jax.device_put(rng.choice(np.asarray([-1.0, 1.0], np.float32), output_size), device)

    for checkpoint in range(start_checkpoint, 49):
        epoch = 4 * (checkpoint + 1)
        state, restored = module._restore_checkpoint(
            str(ckpt_dir / f"seed_{args.train_seed}_epoch_{epoch:04d}.ckpt"), template
        )
        if restored != epoch:
            raise RuntimeError(f"checkpoint epoch mismatch {restored} != {epoch}")
        source = (
            ROOT / "result" / args.experiment
            / f"fixed_checkpoint_adamw_four_events_n{args.attribution_points}"
            / f"epoch_{epoch}_{epoch + 4}"
        )
        event_file = source / "batch_events_shard_00_of_02.csv"
        epochs = tuple(range(epoch + 1, epoch + 5))
        events = event_lookup(event_file, epochs, wanted)
        zero_grads = jax.tree_util.tree_map(jnp.zeros_like, state.params)
        history_update, _ = state.tx.update(zero_grads, state.opt_state, state.params)

        def selected_gradient(params, x, y, noise, train_t, dropout_rng, position):
            xt = module.q_sample(schedule, x, train_t, noise)
            target = x if cfg.predict_x0 else noise
            def loss_fn(pp):
                pred = state.apply_fn(
                    {"params": pp}, xt, train_t, y if cfg.class_cond else None,
                    train=True, rngs={"dropout": dropout_rng},
                )
                axes = tuple(range(1, pred.ndim))
                return jnp.mean((pred - target) ** 2, axis=axes)[position]
            return jax.grad(loss_fn)(params)

        selected_gradient_jit = jax.jit(selected_gradient)

        def projected_outputs(params, states, cond):
            cond_batch = jnp.broadcast_to(cond[None], (states.shape[0], cond.shape[0]))
            eps = state.apply_fn(
                {"params": params}, states, trajectory_timesteps,
                cond_batch if cfg.class_cond else None, train=False,
            )
            flat = eps.reshape((eps.shape[0], -1)).astype(jnp.float32)
            projected = jnp.zeros((eps.shape[0], args.output_projection_dim), jnp.float32)
            return projected.at[:, buckets].add(flat * signs[None, :])

        @jax.jit
        def direction_scores(params, direction, states, cond):
            projected, tangent = jax.jvp(
                lambda pp: projected_outputs(pp, states, cond),
                (params,), (direction,),
            )
            query_norm_sq = jnp.maximum(jnp.sum(projected * projected), 1e-12)
            train_norm_sq = jnp.maximum(tree_norm(direction) ** 2, 1e-12)
            tangent_sq = jnp.sum(tangent * tangent) / tangent.shape[0]
            tangent_root = jnp.sum(jnp.sqrt(jnp.maximum(jnp.sum(tangent * tangent, axis=1), 0.0))) / tangent.shape[0]
            query_norm = jnp.sqrt(query_norm_sq)
            train_norm = jnp.sqrt(train_norm_sq)
            return jnp.asarray([
                tangent_sq,
                tangent_sq / query_norm_sq,
                tangent_sq / train_norm_sq,
                tangent_sq / query_norm_sq / train_norm_sq,
                tangent_root,
                tangent_root / query_norm,
                tangent_root / train_norm,
                tangent_root / query_norm / train_norm,
            ])

        for row_index, dataset_index in enumerate(owned.tolist()):
            updates = []
            residuals = []
            for event_epoch in epochs:
                row, position = events[event_epoch][int(dataset_index)]
                selected = np.asarray(json.loads(row["dataset_indices"]), np.int64)
                train_t = jax.device_put(np.asarray(json.loads(row["timesteps"]), np.int32), device)
                x = jax.device_put(module.maybe_to_dtype(jnp.asarray(ds.images[selected]), cfg.use_bfloat16), device)
                y = jax.device_put(jnp.asarray(ds.labels[selected]), device)
                noise = jax.random.normal(key(row, "noise_key"), x.shape, dtype=x.dtype)
                grad = selected_gradient_jit(
                    state.params, x, y, noise, train_t, key(row, "dropout_key"),
                    jnp.asarray(position, jnp.int32),
                )
                update, _ = state.tx.update(grad, state.opt_state, state.params)
                updates.append(update)
                residuals.append(tree_sub(update, history_update))
            directions = (
                functools.reduce(tree_add, updates[1:], updates[0]),
                updates[0],
                functools.reduce(tree_add, residuals[1:], residuals[0]),
                residuals[0],
            )
            for method_index, direction in enumerate(directions):
                for query in range(10):
                    values = np.asarray(direction_scores(
                        state.params, direction,
                        jax.device_put(trajectory_bank[query, checkpoint], device),
                        jax.device_put(query_conditions[query], device),
                    ), np.float64)
                    scores[method_index, 0, :, query, row_index] += values[:4]
                    scores[method_index, 1, :, query, row_index] += values[4:]
            if (row_index + 1) % args.progress_every == 0 or row_index + 1 == len(owned):
                print(
                    f"[direct-jvp] trajectory={args.trajectory} shard={args.shard_id} "
                    f"checkpoint={checkpoint + 1}/49 datapoints={row_index + 1}/{len(owned)}",
                    flush=True,
                )
        atomic_npz(
            args.out_dir / "partials" / f"checkpoint_{checkpoint:04d}_{shard_tag}.npz",
            scores=scores, score_indices=owned,
            completed_checkpoint=np.asarray(checkpoint, np.int32),
        )
    atomic_npz(args.out_dir / f"{shard_tag}.npz", scores=scores, score_indices=owned)


def merge(args):
    records = json.loads((ROOT / "queries_seed_0_9.json").read_text())["queries"]
    pieces = []
    for shard in range(args.num_shards):
        path = args.out_dir / f"shard_{shard:02d}_of_{args.num_shards:02d}.npz"
        with np.load(path, allow_pickle=False) as payload:
            pieces.append((np.asarray(payload["score_indices"]), np.asarray(payload["scores"])))
    indices = np.concatenate([item[0] for item in pieces])
    scores = np.concatenate([item[1] for item in pieces], axis=-1)
    order = np.argsort(indices)
    indices, scores = indices[order], scores[..., order]
    schemes = []
    for mi, method in enumerate(METHODS):
        for ri, reduction in enumerate(REDUCTIONS):
            namespace = (
                f"traj_tracin_direct_jvp_{reduction}_proj{args.output_projection_dim}_"
                f"adamw4_{method}_{args.trajectory}_constant_lr"
            )
            schemes.append(namespace.removeprefix("traj_tracin_"))
            for vi, variant in enumerate(VARIANTS):
                component = {
                    "raw": "score", "query_l2": "score_query_normalized",
                    "train_l2": "score_train_l2_normalized",
                    "query_train_l2": "score_query_train_l2_normalized",
                }[variant]
                for query, record in enumerate(records):
                    out = (
                        ROOT / "result" / args.experiment / "attribution_score/prompted_solo"
                        / f"train_seed_{args.train_seed}"
                        / f"query_{_prompt_tag(str(record['prompt']))}"
                        / f"initial_seed_{int(record['initial_seed'])}" / namespace / component
                    )
                    out.mkdir(parents=True, exist_ok=True)
                    np.save(out / "scores.npy", scores[mi, ri, vi, query])
                    np.save(out / "score_indices.npy", indices)
                    (out / "score_artifact_manifest.json").write_text(json.dumps({
                        "semantics": SEMANTICS, "trajectory": args.trajectory,
                        "method": method, "reduction": reduction,
                        "checkpoint_weighting": "constant",
                        "output_projection": "fixed_countsketch",
                        "output_projection_dim": args.output_projection_dim,
                    }, indent=2, sort_keys=True))
    print(",".join(schemes), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("score-shard", "merge"))
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--trajectory", choices=("reference", "own"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--attribution-points", type=int, default=5000)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--output-projection-dim", type=int, default=4096)
    parser.add_argument("--output-projection-seed", type=int, default=20260919)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--namespace", default="loss_direction_original_f_checkpoint_own_trajectory_endpoints_all10")
    args = parser.parse_args()
    args.epochs = 200
    if args.command == "score-shard":
        score_shard(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
