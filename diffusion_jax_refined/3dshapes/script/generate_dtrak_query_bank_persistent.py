#!/usr/bin/env python3
"""Generate many D-TRAK query artifacts in one persistent JAX process.

The legacy single-query entry point rebuilds the dataset, restores the same
checkpoint, recreates CountSketch, and recompiles JAX for every query.  This
worker performs those invariant operations once per GPU shard and then streams
all assigned queries through three reusable compiled objective functions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
REFINE_ROOT = SHAPES_ROOT.parent
LEGACY_ROOT = REFINE_ROOT / "legacy_jax"
for root in (SHAPES_ROOT, LEGACY_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from dataset_config import _prompt_tag, attribution_config  # noqa: E402
from dtrak.algorithm import (  # noqa: E402
    EndpointDTrakJAXConfig,
    apply_checkpoint_config,
    array_to_device,
    build_countsketch_projector_jax,
    get_adapter,
    load_attribution_endpoint,
    make_diffusion_schedule,
    make_jax_key,
    make_query_phi_fn,
    repeat_condition_to_batch,
    schedule_to_device,
    tree_to_device,
)


OBJECTIVES = ("simple_loss", "square", "average")


def parse_ids(text: str) -> list[int]:
    return [int(token) for token in text.replace(",", " ").split()]


def artifact_path(
    sample_root: Path,
    prompt: str,
    seed: int,
    train_seed: int,
    epochs: int,
    objective: str,
) -> Path:
    model_dir = (
        sample_root
        / "cifar"
        / f"prompt_{_prompt_tag(prompt)}"
        / f"model_prompted_solo__ckpt_seed_{train_seed}_epoch_{epochs:04d}"
    )
    return (
        model_dir
        / f"seed_{seed:06d}_query_gradient_dtrak_{objective}_100x1"
        / "dtrak"
        / "query_gradient_artifact.npz"
    )


def valid_artifact(path: Path, objective: str, proj_dim: int, samples: int) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            return (
                payload["query_features"].shape == (1, proj_dim)
                and str(payload["output_function"].item()) == objective
                and int(payload["expectation_samples"].item()) == samples
                and bool(payload["explicit_timestep_grid"].item())
            )
    except (OSError, ValueError, KeyError, EOFError):
        return False


def atomic_save(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--query-ids", required=True)
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--objectives", default=",".join(OBJECTIVES))
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    objectives = tuple(args.objectives.replace(",", " ").split())
    unknown = sorted(set(objectives) - set(OBJECTIVES))
    if unknown:
        raise ValueError(f"unsupported objectives: {unknown}")

    records = json.loads(args.query_file.read_text())["queries"]
    query_ids = parse_ids(args.query_ids)[args.shard_index :: args.shard_count]
    if not query_ids:
        print(f"[done] shard {args.shard_index}/{args.shard_count}: no queries")
        return

    cfg = EndpointDTrakJAXConfig(**attribution_config("dtrak"))
    cfg.reference_ckpt = str(
        SHAPES_ROOT
        / "result"
        / os.environ.get("EXPERIMENT_TAG", "experiment1")
        / "model"
        / "prompted_jax"
        / f"seed_{args.train_seed}_epoch_{args.epochs:04d}.ckpt"
    )
    if not Path(cfg.reference_ckpt).is_file():
        raise FileNotFoundError(f"reference checkpoint not found: {cfg.reference_ckpt}")

    apply_checkpoint_config(cfg, cfg.reference_ckpt)
    adapter = get_adapter(cfg)
    device = adapter.choose_device(cfg.prefer_device)
    print(
        f"[setup] persistent shard={args.shard_index}/{args.shard_count} "
        f"queries={len(query_ids)} objectives={','.join(objectives)} device={device}",
        flush=True,
    )

    dataset = adapter.iter_dataset(cfg)
    model = adapter.build_model(cfg)
    state_template = adapter.build_state_template(cfg, model, device)
    schedule = schedule_to_device(
        make_diffusion_schedule(cfg.timesteps, cfg.beta_start, cfg.beta_end), device
    )
    state, _ = adapter.restore_state(cfg.reference_ckpt, state_template)
    params = tree_to_device(state.ema_params, device)
    projector = build_countsketch_projector_jax(
        params,
        int(cfg.proj_dim),
        seed_parts=(cfg.seed, "dtrak_projection", 0, 0),
        device=device,
    )
    rng = array_to_device(make_jax_key(cfg.seed, "q", 0, 0), device)
    t_max = min(int(schedule.betas.shape[0]) - 1, int(cfg.t_max_end_frac * cfg.timesteps))

    query_functions = {
        objective: make_query_phi_fn(
            adapter,
            model,
            schedule,
            projector,
            t_min=int(cfg.t_min_end),
            t_max=t_max,
            num_expectation_samples=int(cfg.query_expectation_samples),
            output_function=objective,
            explicit_timestep_grid=bool(cfg.explicit_timestep_grid),
        )
        for objective in objectives
    }

    started = time.perf_counter()
    completed = 0
    skipped = 0
    total = len(query_ids) * len(objectives)
    for position, query_id in enumerate(query_ids, start=1):
        record = records[query_id]
        prompt = str(record["prompt"])
        seed = int(record.get("initial_seed", record.get("seed")))
        paths = {
            objective: artifact_path(
                args.sample_root,
                prompt,
                seed,
                args.train_seed,
                args.epochs,
                objective,
            )
            for objective in objectives
        }
        pending = [
            objective
            for objective in objectives
            if not valid_artifact(
                paths[objective],
                objective,
                int(cfg.proj_dim),
                int(cfg.query_expectation_samples),
            )
        ]
        skipped += len(objectives) - len(pending)
        if not pending:
            print(f"[skip] Q{query_id} ({position}/{len(query_ids)}) all objectives", flush=True)
            continue

        model_dir = paths[pending[0]].parents[2]
        cfg.attribution_sample_dir = str(model_dir)
        cfg.attribution_sample_seed = seed
        x0_ref, _ = load_attribution_endpoint(cfg)
        cond = adapter.make_query_cond(dataset, prompt, cfg)
        cond = repeat_condition_to_batch(cond, int(x0_ref.shape[0]))
        x0_ref = array_to_device(x0_ref, device)
        cond = array_to_device(cond, device)

        for objective in pending:
            loss, feature = query_functions[objective](params, x0_ref, cond, rng)
            feature.block_until_ready()
            atomic_save(
                paths[objective],
                query_features=np.asarray(feature, dtype=np.float32)[None, :],
                ckpt_indices=np.asarray([0], dtype=np.int32),
                sample_indices=np.asarray([0], dtype=np.int32),
                damping=np.asarray(float(cfg.damping), dtype=np.float32),
                proj_dim=np.asarray(int(cfg.proj_dim), dtype=np.int32),
                output_function=np.asarray(objective),
                explicit_timestep_grid=np.asarray(bool(cfg.explicit_timestep_grid)),
                expectation_samples=np.asarray(
                    int(cfg.query_expectation_samples), dtype=np.int32
                ),
            )
            completed += 1
            print(
                f"[saved] Q{query_id} objective={objective} loss={float(loss):.6f} "
                f"progress={completed + skipped}/{total}",
                flush=True,
            )

    print(
        f"[done] persistent query shard={args.shard_index}/{args.shard_count} "
        f"computed={completed} skipped={skipped} elapsed={time.perf_counter() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
