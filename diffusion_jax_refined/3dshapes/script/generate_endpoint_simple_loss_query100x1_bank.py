#!/usr/bin/env python3
"""Cache checkpoint-level endpoint-TracIn query100x1 features.

Each query/checkpoint feature is the CountSketch projection of the gradient of
the mean simple denoising loss over the historical 100-timestep MC1 noise bank.
The train side is deliberately absent from this script; the resulting small
query bank is consumed by the fused train/score stream.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
REFINE_ROOT = SHAPES_ROOT.parent
LEGACY_ROOT = REFINE_ROOT / "legacy_jax"
for path in (SHAPES_ROOT, REFINE_ROOT, LEGACY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp

from dataset_config import _prompt_tag, attribution_config
from direction20_query_common import load_queries
from dtrak.algorithm import build_countsketch_projector_jax
from score_retrac_endpoint100x1_checkpoint_major import (
    endpoint_noise_key,
    make_endpoint_query_fn,
)
from traj_tracin.algorithm import (
    TrajAttributionConfig,
    apply_checkpoint_config,
    array_to_device,
    get_adapter,
    list_checkpoints_sorted,
    make_diffusion_schedule,
    save_npz_compressed_atomic,
    schedule_to_device,
    select_snapshot_positions,
    select_state_params,
    tree_to_device,
)


OBJECTIVE = "endpoint_simple_loss_mean100t_mc1"


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value]


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


def merge_artifact(artifact: Path, expected_parts: int) -> None:
    if artifact.is_file():
        print(f"[skip] merged artifact exists: {artifact}", flush=True)
        return
    part_dir = Path(str(artifact) + ".parts")
    parts = [part_dir / f"ckpt_{index:04d}.npz" for index in range(expected_parts)]
    missing = [path for path in parts if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{artifact}: missing {len(missing)} checkpoint parts; first={missing[0]}"
        )
    payloads = []
    for part in parts:
        with np.load(part, allow_pickle=False) as data:
            payloads.append({key: np.asarray(data[key]) for key in data.files})
    concatenate = {
        "query_features",
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
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--artifact-list", type=Path)
    args = parser.parse_args()

    query_ids = parse_ints(args.query_ids)
    os.environ.update(
        EXPERIMENT_TAG=args.experiment,
        TRAIN_SEED=str(args.train_seed),
        JAX_EPOCHS=str(args.epochs),
        TRAJ_PARAMETER_SOURCE="raw",
        TRAJ_TRACIN_PROJ_DIM="4096",
    )
    values = attribution_config("traj_tracin")
    result_root = SHAPES_ROOT / "result" / args.experiment
    checkpoint_root = result_root / "model" / "prompted_jax"
    values.update(
        checkpoint_dir=str(checkpoint_root),
        reference_ckpt=str(
            checkpoint_root / f"seed_{args.train_seed}_epoch_{args.epochs:04d}.ckpt"
        ),
        seed=args.train_seed,
        epochs=args.epochs,
        parameter_source="raw",
        num_traj_snapshots=100,
        proj_dim=4096,
        out_dir=str(result_root / "tmp"),
    )
    cfg = TrajAttributionConfig(**values)
    checkpoints = list_checkpoints_sorted(cfg.checkpoint_dir)
    if len(checkpoints) != 50:
        raise ValueError(f"expected 50 checkpoints, found {len(checkpoints)}")
    reference_checkpoint = Path(cfg.reference_ckpt or checkpoints[-1])
    sample_root = result_root / "sample_ddim_eta0_1000"
    records = json.loads(args.query_file.read_text())["queries"]
    artifacts = []
    for query_id in query_ids:
        record = records[query_id]
        artifacts.append(
            artifact_path(
                sample_root,
                reference_checkpoint,
                str(record["prompt"]),
                int(record["initial_seed"]),
                args.namespace,
            )
        )
    if args.merge_only:
        for artifact in artifacts:
            merge_artifact(artifact, 49)
        if args.artifact_list is not None:
            args.artifact_list.parent.mkdir(parents=True, exist_ok=True)
            args.artifact_list.write_text(
                "".join(f"{artifact}\n" for artifact in artifacts)
            )
            print(f"[manifest] {args.artifact_list}", flush=True)
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
        args.query_file,
        query_ids,
        sample_root,
        reference_checkpoint,
        adapter,
        dataset,
        cfg,
    )
    ddim_timesteps = np.linspace(
        int(cfg.timesteps) - 1, 0, int(cfg.ddim_steps), dtype=np.int32
    )
    positions = select_snapshot_positions(int(cfg.ddim_steps), 100, None)
    timesteps = np.asarray(
        [int(ddim_timesteps[int(position)]) for position in positions], dtype=np.int32
    )

    for checkpoint in range(49):
        if checkpoint % args.shard_count != args.shard_index:
            continue
        parts = [
            Path(str(artifact) + ".parts") / f"ckpt_{checkpoint:04d}.npz"
            for artifact in artifacts
        ]
        if all(artifact.is_file() or part.is_file() for artifact, part in zip(artifacts, parts)):
            print(f"[skip] checkpoint {checkpoint + 1}/49 parts complete", flush=True)
            continue

        state, _ = adapter.restore_state(checkpoints[checkpoint], state_template)
        params = tree_to_device(select_state_params(state, "raw"), device)
        projector = build_countsketch_projector_jax(
            params,
            4096,
            seed_parts=(args.train_seed, "traj_tracin_projection", checkpoint),
            device=device,
        )
        query_fn = make_endpoint_query_fn(adapter, model, schedule, projector)
        print(f"[checkpoint {checkpoint + 1}/49] restored", flush=True)

        for start in range(0, len(queries), args.batch_size):
            real = queries[start : start + args.batch_size]
            padded = list(real)
            while len(padded) < args.batch_size:
                padded.append(padded[-1])
            endpoints = np.stack([query["endpoint"] for query in padded])
            conds = np.stack([query["cond"] for query in padded])
            noises = np.stack(
                [
                    np.asarray(
                        jax.device_get(
                            jax.random.normal(
                                endpoint_noise_key(
                                    args.train_seed,
                                    int(query["id"]),
                                    checkpoint,
                                    100_001,
                                ),
                                (100,) + query["endpoint"].shape,
                                dtype=jnp.float32,
                            )
                        ),
                        dtype=np.float32,
                    )
                    for query in padded
                ]
            )
            features = query_fn(
                params,
                array_to_device(jnp.asarray(endpoints), device),
                array_to_device(jnp.asarray(conds), device),
                array_to_device(jnp.asarray(noises), device),
                array_to_device(jnp.asarray(timesteps), device),
            )
            features.block_until_ready()
            features_host = np.asarray(jax.device_get(features), dtype=np.float32)
            for offset, _query in enumerate(real):
                artifact = artifacts[start + offset]
                part = parts[start + offset]
                if artifact.is_file() or part.is_file():
                    continue
                save_npz_compressed_atomic(
                    str(part),
                    query_features=features_host[offset : offset + 1],
                    ckpt_indices=np.asarray([checkpoint], dtype=np.int32),
                    timesteps=np.asarray([-1], dtype=np.int32),
                    snapshot_positions=np.asarray([-1], dtype=np.int32),
                    term_weights=np.asarray([1.0], dtype=np.float32),
                    ckpt_paths=np.asarray([str(checkpoints[checkpoint])]),
                    proj_dim=np.asarray(4096, dtype=np.int32),
                    query_objective=np.asarray(OBJECTIVE),
                    query_timestamp_aggregation=np.asarray("mean_loss_then_gradient"),
                    query_timestamp_count=np.asarray(100, dtype=np.int32),
                    query_mc_per_timestamp=np.asarray(1, dtype=np.int32),
                    query_noise_seed_rule=np.asarray(
                        "endpoint_noise_key(train_seed,query_id,checkpoint,100001)"
                    ),
                    projection_seed_rule=np.asarray(
                        "(train_seed,'traj_tracin_projection',checkpoint_index)"
                    ),
                    checkpoint_major_query_batch=np.asarray(True),
                )
            print(
                f"[checkpoint {checkpoint + 1}/49] queries "
                f"{start + 1}-{start + len(real)}/{len(queries)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
