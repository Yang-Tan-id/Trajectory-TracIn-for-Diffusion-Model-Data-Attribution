from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
REFINE_ROOT = SHAPES_ROOT.parent
LEGACY_ROOT = REFINE_ROOT / "legacy_jax"
for path in (SHAPES_ROOT, LEGACY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp

from dataset_config import _prompt_tag, attribution_config
from dtrak.algorithm import build_countsketch_projector_jax
from traj_tracin.algorithm import (
    TrajAttributionConfig,
    apply_checkpoint_config,
    array_to_device,
    checkpoint_timestamp_shared_noise_key,
    get_adapter,
    list_checkpoints_sorted,
    make_diffusion_schedule,
    save_npz_compressed_atomic,
    schedule_to_device,
    select_state_params,
    tracin_checkpoint_lr_weight,
    tree_to_device,
)


RAW_OBJECTIVE = "trajectory_polluted_endpoint_next_delta_projection"
NORMALIZED_OBJECTIVE = RAW_OBJECTIVE + "_normalized"


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value.strip()]


def artifact_path(
    sample_root: Path,
    checkpoint: Path,
    prompt: str,
    seed: int,
    namespace: str,
) -> Path:
    return (
        sample_root
        / "cifar"
        / f"prompt_{_prompt_tag(prompt)}"
        / f"model_prompted_solo__ckpt_{checkpoint.stem}"
        / f"seed_{seed:06d}_query_gradient_{namespace}"
        / "traj_tracin"
        / "query_gradient_artifact.npz"
    )


def load_queries(
    query_file: Path,
    query_ids: list[int],
    sample_root: Path,
    checkpoint: Path,
    adapter: Any,
    dataset: Any,
    cfg: TrajAttributionConfig,
) -> list[dict[str, Any]]:
    records = json.loads(query_file.read_text())["queries"]
    queries: list[dict[str, Any]] = []
    for query_id in query_ids:
        record = records[query_id]
        prompt = str(record["prompt"])
        seed = int(record["initial_seed"])
        run_root = (
            sample_root
            / "cifar"
            / f"prompt_{_prompt_tag(prompt)}"
            / f"model_prompted_solo__ckpt_{checkpoint.stem}"
        )
        seed_dir = run_root / f"seed_{seed:06d}"
        endpoint_path = seed_dir / "final_state.npy"
        if not endpoint_path.is_file():
            raise FileNotFoundError(endpoint_path)
        endpoint_all = np.load(endpoint_path)
        if endpoint_all.ndim != 4 or endpoint_all.shape[0] < 1:
            raise ValueError(f"invalid endpoint array {endpoint_path}: {endpoint_all.shape}")
        cond = np.asarray(adapter.make_query_cond(dataset, prompt, cfg))
        queries.append(
            {
                "id": query_id,
                "prompt": prompt,
                "seed": seed,
                "endpoint": np.asarray(endpoint_all[0:1], dtype=np.float32),
                "cond": cond,
            }
        )
    return queries


def make_projected_raw_and_norm_batch_fn(adapter, model, projector):
    def scalar_with_norm(params, target_params, xt, t_scalar, cond):
        t = jnp.full((xt.shape[0],), t_scalar, dtype=jnp.int32)
        eps = adapter.eps_apply(model, params, xt, t, cond)
        target = jax.lax.stop_gradient(
            adapter.eps_apply(model, target_params, xt, t, cond)
        )
        delta = jax.lax.stop_gradient(target - eps)
        norm = jnp.sqrt(jnp.sum(jnp.square(delta), dtype=jnp.float32))
        return jnp.mean(eps * delta), norm

    value_grad = jax.value_and_grad(scalar_with_norm, has_aux=True)

    def one(params, target_params, xt, t_scalar, cond):
        (_value, norm), grad = value_grad(
            params, target_params, xt, t_scalar, cond
        )
        return projector(grad).astype(jnp.float32), norm.astype(jnp.float32)

    by_timestamp = jax.vmap(one, in_axes=(None, None, 0, 0, None))
    by_query = jax.vmap(by_timestamp, in_axes=(None, None, 0, None, 0))
    return jax.jit(by_query)


def common_part_payload(
    features: np.ndarray,
    norms: np.ndarray,
    ckpt_i: int,
    ckpt_path: str,
    timesteps: np.ndarray,
    positions: np.ndarray,
    weight: float,
    objective: str,
    normalized: bool,
) -> dict[str, np.ndarray]:
    count = len(timesteps)
    return {
        "query_features": np.asarray(features, dtype=np.float32),
        "polluted_endpoint_delta_l2_norms": np.asarray(norms, dtype=np.float32),
        "ckpt_indices": np.full(count, ckpt_i, dtype=np.int32),
        "timesteps": np.asarray(timesteps, dtype=np.int32),
        "snapshot_positions": np.asarray(positions, dtype=np.int32),
        "term_weights": np.full(count, weight / count, dtype=np.float32),
        "ckpt_paths": np.asarray([ckpt_path] * count),
        "proj_dim": np.asarray(features.shape[-1], dtype=np.int32),
        "query_objective": np.asarray(objective),
        "query_target_checkpoint": np.asarray("next_checkpoint"),
        "polluted_endpoint_x_t": np.asarray(True),
        "polluted_endpoint_delta_normalized": np.asarray(normalized),
        "normalized_delta_derived_from_raw": np.asarray(normalized),
        "polluted_endpoint_noise_seed_rule": np.asarray(
            "checkpoint_timestamp_shared_noise_key(train_seed,checkpoint_index,timestep)"
        ),
        "polluted_endpoint_delta_definition": np.asarray(
            "stopgrad(eps(params[c+1],x_t,t)-eps(params[c],x_t,t))"
        ),
        "projection_seed_rule": np.asarray(
            "(train_seed,'traj_tracin_projection',checkpoint_index)"
        ),
        "checkpoint_major_query_batch": np.asarray(True),
    }


def merge_artifact(artifact: Path, expected_parts: int) -> None:
    if artifact.is_file():
        print(f"[skip] merged artifact exists: {artifact}", flush=True)
        return
    part_dir = Path(str(artifact) + ".parts")
    parts = [part_dir / f"ckpt_{i:04d}.npz" for i in range(expected_parts)]
    missing = [path for path in parts if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{artifact}: missing {len(missing)} checkpoint parts; first={missing[0]}"
        )
    payloads = []
    for path in parts:
        with np.load(path, allow_pickle=False) as data:
            payloads.append({key: np.asarray(data[key]) for key in data.files})
    concatenate = {
        "query_features",
        "polluted_endpoint_delta_l2_norms",
        "ckpt_indices",
        "timesteps",
        "snapshot_positions",
        "term_weights",
        "ckpt_paths",
    }
    merged = {
        key: (
            np.concatenate([payload[key] for payload in payloads], axis=0)
            if key in concatenate
            else payloads[0][key]
        )
        for key in payloads[0]
    }
    save_npz_compressed_atomic(str(artifact), **merged)
    print(f"[merge] {artifact} shape={merged['query_features'].shape}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--query-ids", required=True)
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--raw-namespace", required=True)
    parser.add_argument("--normalized-namespace", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--timestamp-count", type=int, default=10)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()

    query_ids = parse_ints(args.query_ids)
    os.environ.update(
        EXPERIMENT_TAG=args.experiment,
        TRAIN_SEED=str(args.train_seed),
        JAX_EPOCHS=str(args.epochs),
        TRAJ_PARAMETER_SOURCE="raw",
        TRAJ_QUERY_OBJECTIVE=RAW_OBJECTIVE,
        TRAJ_NUM_SNAPSHOTS=str(args.timestamp_count),
        TRAJ_TRACIN_PROJ_DIM="4096",
    )
    cfg_values = attribution_config("traj_tracin")
    result_root = SHAPES_ROOT / "result" / args.experiment
    checkpoint_root = result_root / "model" / "prompted_jax"
    cfg_values.update(
        checkpoint_dir=str(checkpoint_root),
        reference_ckpt=str(
            checkpoint_root
            / f"seed_{args.train_seed}_epoch_{args.epochs:04d}.ckpt"
        ),
        seed=args.train_seed,
        epochs=args.epochs,
        parameter_source="raw",
        query_objective=RAW_OBJECTIVE,
        num_traj_snapshots=args.timestamp_count,
        proj_dim=4096,
        out_dir=str(result_root / "tmp"),
    )
    cfg = TrajAttributionConfig(**cfg_values)
    checkpoints = list_checkpoints_sorted(cfg.checkpoint_dir)
    if len(checkpoints) != 50:
        raise ValueError(f"expected 50 checkpoints, found {len(checkpoints)}")
    reference_checkpoint = Path(cfg.reference_ckpt or checkpoints[-1])
    sample_root = SHAPES_ROOT / "result" / args.experiment / "sample_ddim_eta0_1000"

    records = json.loads(args.query_file.read_text())["queries"]
    artifact_pairs = []
    for query_id in query_ids:
        record = records[query_id]
        prompt, seed = str(record["prompt"]), int(record["initial_seed"])
        artifact_pairs.append(
            (
                artifact_path(sample_root, reference_checkpoint, prompt, seed, args.raw_namespace),
                artifact_path(sample_root, reference_checkpoint, prompt, seed, args.normalized_namespace),
            )
        )
    if args.merge_only:
        for raw, normalized in artifact_pairs:
            merge_artifact(raw, 49)
            merge_artifact(normalized, 49)
        return

    apply_checkpoint_config(cfg, str(reference_checkpoint))
    adapter = get_adapter(cfg)
    device = adapter.choose_device(cfg.prefer_device)
    dataset = adapter.iter_dataset(cfg)
    model = adapter.build_model(cfg)
    state_template = adapter.build_state_template(cfg, model, device)
    schedule = schedule_to_device(
        make_diffusion_schedule(cfg.timesteps, cfg.beta_start, cfg.beta_end), device
    )
    queries = load_queries(
        args.query_file, query_ids, sample_root, reference_checkpoint,
        adapter, dataset, cfg,
    )
    positions = np.linspace(0, 999, args.timestamp_count, dtype=np.int32)
    timesteps = (999 - positions).astype(np.int32)
    alpha_bars = np.asarray(jax.device_get(schedule.alphas_cumprod), dtype=np.float32)

    for ckpt_i in range(49):
        if ckpt_i % args.shard_count != args.shard_index:
            continue
        required = []
        for raw, normalized in artifact_pairs:
            if not raw.is_file():
                required.append(Path(str(raw) + ".parts") / f"ckpt_{ckpt_i:04d}.npz")
            if not normalized.is_file():
                required.append(Path(str(normalized) + ".parts") / f"ckpt_{ckpt_i:04d}.npz")
        if not required or all(path.is_file() for path in required):
            print(f"[skip] checkpoint {ckpt_i + 1}/49 parts complete", flush=True)
            continue

        current_state, _ = adapter.restore_state(checkpoints[ckpt_i], state_template)
        target_state, _ = adapter.restore_state(checkpoints[ckpt_i + 1], state_template)
        params = tree_to_device(select_state_params(current_state, "raw"), device)
        target_params = tree_to_device(select_state_params(target_state, "raw"), device)
        projector = build_countsketch_projector_jax(
            params,
            4096,
            seed_parts=(args.train_seed, "traj_tracin_projection", ckpt_i),
            device=device,
        )
        batch_fn = make_projected_raw_and_norm_batch_fn(adapter, model, projector)
        noise = []
        endpoint_shape = queries[0]["endpoint"].shape
        for timestep in timesteps:
            key = checkpoint_timestamp_shared_noise_key(
                args.train_seed, ckpt_i, int(timestep)
            )
            noise.append(
                np.asarray(
                    jax.device_get(jax.random.normal(key, endpoint_shape, dtype=jnp.float32)),
                    dtype=np.float32,
                )
            )
        noise = np.stack(noise, axis=0)
        weight = tracin_checkpoint_lr_weight(
            cfg, checkpoints[ckpt_i], ckpt_i, len(checkpoints), len(dataset)
        )
        print(f"[checkpoint {ckpt_i + 1}/49] restored; query batches start", flush=True)

        for start in range(0, len(queries), args.batch_size):
            real = queries[start : start + args.batch_size]
            padded = list(real)
            while len(padded) < args.batch_size:
                padded.append(padded[-1])
            endpoints = np.stack([item["endpoint"] for item in padded], axis=0)
            conds = np.stack([item["cond"] for item in padded], axis=0)
            xt = []
            for endpoint in endpoints:
                xt.append(
                    np.stack(
                        [
                            np.sqrt(float(alpha_bars[t])) * endpoint
                            + np.sqrt(max(0.0, 1.0 - float(alpha_bars[t]))) * noise_i
                            for t, noise_i in zip(timesteps, noise)
                        ],
                        axis=0,
                    )
                )
            raw_features, norms = batch_fn(
                params,
                target_params,
                array_to_device(jnp.asarray(np.stack(xt), dtype=jnp.float32), device),
                array_to_device(jnp.asarray(timesteps, dtype=jnp.int32), device),
                array_to_device(jnp.asarray(conds), device),
            )
            raw_features = np.asarray(jax.device_get(raw_features), dtype=np.float32)
            norms = np.asarray(jax.device_get(norms), dtype=np.float32)
            for local_i, item in enumerate(real):
                raw_artifact, normalized_artifact = artifact_pairs[start + local_i]
                raw_part = Path(str(raw_artifact) + ".parts") / f"ckpt_{ckpt_i:04d}.npz"
                normalized_part = Path(str(normalized_artifact) + ".parts") / f"ckpt_{ckpt_i:04d}.npz"
                if not raw_artifact.is_file() and not raw_part.is_file():
                    save_npz_compressed_atomic(
                        str(raw_part),
                        **common_part_payload(
                            raw_features[local_i], norms[local_i], ckpt_i,
                            checkpoints[ckpt_i], timesteps, positions, weight,
                            RAW_OBJECTIVE, False,
                        ),
                    )
                if not normalized_artifact.is_file() and not normalized_part.is_file():
                    normalized = raw_features[local_i] / np.maximum(
                        norms[local_i, :, None], 1e-12
                    )
                    save_npz_compressed_atomic(
                        str(normalized_part),
                        **common_part_payload(
                            normalized, norms[local_i], ckpt_i,
                            checkpoints[ckpt_i], timesteps, positions, weight,
                            NORMALIZED_OBJECTIVE, True,
                        ),
                    )
            print(
                f"[checkpoint {ckpt_i + 1}/49] queries "
                f"{start + 1}-{start + len(real)}/{len(queries)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
