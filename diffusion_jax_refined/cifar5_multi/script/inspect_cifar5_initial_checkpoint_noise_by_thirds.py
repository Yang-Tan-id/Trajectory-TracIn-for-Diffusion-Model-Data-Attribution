#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_cifar5_multi_experiment import Job, gpu_env, parse_gpus, run_parallel_jobs, slot_for, worker_gpus
from run_cifar5_multi_random_prompted_queries import build_query_specs, query_tag
from run_cifar5_multi_traj_temporal_thirds import SEGMENTS, segment_position_map


def run_checked(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print(f"[noise-thirds] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def result_path(root: Path, args: argparse.Namespace, spec: dict[str, int | str]) -> Path:
    return (
        root / "result" / args.experiment / "diagnostics"
        / args.output_name / f"query_{query_tag(str(spec['query']))}"
        / f"initial_seed_{int(spec['initial_seed'])}.json"
    )


def run_worker(root: Path, args: argparse.Namespace, spec_index: int) -> None:
    spec = build_query_specs(args)[spec_index]
    query = str(spec["query"])
    seed = int(spec["initial_seed"])
    tag = query_tag(query)
    output = result_path(root, args, spec)
    if output.is_file():
        print(f"[noise-thirds] skip complete: {output}", flush=True)
        return

    temp_root = (
        root / "result" / args.experiment / "sample" / ".noise_thirds"
        / f"query_{tag}_seed_{seed}"
    )
    model_root = (
        temp_root / "cifar" / f"prompt_{tag}"
        / f"model_prompted_solo__ckpt_seed_{args.train_seed}_epoch_{args.epochs:04d}"
    )
    sample_dir = model_root / f"seed_{seed:06d}"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHON_BIN": args.python_bin,
            "EXPERIMENT_TAG": args.experiment,
            "CIFAR5_MULTI_SIZE": str(args.size),
            "TRAIN_SEED": str(args.train_seed),
            "JAX_EPOCHS": str(args.epochs),
            "QUERY": query,
            "INITIAL_SEED": str(seed),
            "SAMPLE_SEEDS": str(seed),
            "SAMPLE_MODEL_MODE": "prompted_solo",
            "ATTRIBUTION_SAMPLE_MODEL_MODE": "prompted_solo",
            "SAMPLE_ROOT": str(temp_root),
            "SAMPLE_TRAJECTORY_STEPS": "1000",
            "JAX_BFLOAT16": "1",
            "TF_GPU_ALLOCATOR": "cuda_malloc_async",
        }
    )

    try:
        if not (sample_dir / "trajectory_xt.npy").is_file():
            run_checked(
                [args.python_bin, str(root / "sampling" / "run_sampling.py")],
                cwd=root.parent,
                env=env,
            )

        os.environ.update(env)
        legacy_root = root.parent / "legacy_jax"
        for path in (root.parent, legacy_root):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

        import jax
        import jax.numpy as jnp
        from common.config_loader import load_config
        from traj_tracin.algorithm import (
            TrajAttributionConfig,
            apply_checkpoint_config,
            array_to_device,
            get_adapter,
            list_checkpoints_sorted,
            load_attribution_trajectory,
            select_state_params,
            tree_to_device,
        )

        config_module = load_config(root / "dataset_config.py")
        values = dict(config_module.attribution_config("traj_tracin"))
        values.update(
            {
                "query": query,
                "attribution_sample_dir": str(model_root),
                "attribution_sample_seed": seed,
                "parameter_source": "raw",
            }
        )
        cfg = TrajAttributionConfig(**values)
        checkpoints = list_checkpoints_sorted(cfg.checkpoint_dir)
        if len(checkpoints) < 2:
            raise RuntimeError("at least two checkpoints are required")
        apply_checkpoint_config(cfg, checkpoints[0])
        adapter = get_adapter(cfg)
        device = adapter.choose_device(cfg.prefer_device)
        dataset = adapter.iter_dataset(cfg)
        model = adapter.build_model(cfg)
        state_template = adapter.build_state_template(cfg, model, device)

        params = {}
        for label, checkpoint, source in (
            ("initial", checkpoints[0], "raw"),
            ("next", checkpoints[1], "raw"),
            ("final", checkpoints[-1], "raw"),
            # Sampling uses the final checkpoint's EMA parameters, so this is
            # the model that actually generated x_t^ref.
            ("reference_trajectory", checkpoints[-1], "ema"),
        ):
            state, _ = adapter.restore_state(checkpoint, state_template)
            params[label] = tree_to_device(select_state_params(state, source), device)

        xt_all, timestep_all, position_all, _ = load_attribution_trajectory(cfg)
        by_position = {int(position): i for i, position in enumerate(position_all)}
        positions = sorted(segment_position_map())
        missing = [position for position in positions if position not in by_position]
        if missing:
            raise ValueError(f"temporary full trajectory is missing positions: {missing}")
        xt = array_to_device(
            jnp.stack([xt_all[by_position[position]] for position in positions]), device
        )
        timesteps = np.asarray(
            [int(timestep_all[by_position[position]]) for position in positions],
            dtype=np.int32,
        )
        t_device = array_to_device(jnp.asarray(timesteps), device)
        cond = array_to_device(adapter.make_query_cond(dataset, query, cfg), device)

        @jax.jit
        def predict(parameters, states, ts):
            return jax.vmap(
                lambda state, timestep: adapter.eps_apply(
                    model,
                    parameters,
                    state,
                    jnp.full((state.shape[0],), timestep, dtype=jnp.int32),
                    cond,
                )
            )(states, ts)

        predictions = {}
        for label in ("initial", "next", "final", "reference_trajectory"):
            value = predict(params[label], xt, t_device)
            value.block_until_ready()
            predictions[label] = np.asarray(jax.device_get(value), dtype=np.float32)

        rows = []
        position_segments = segment_position_map()
        for i, (position, timestep) in enumerate(zip(positions, timesteps)):
            initial = predictions["initial"][i]
            row = {
                "position": int(position),
                "timestep": int(timestep),
                "segment": position_segments[int(position)],
                "initial_rms": float(np.sqrt(np.mean(initial * initial))),
            }
            for reference in ("next", "final", "reference_trajectory"):
                target = predictions[reference][i]
                difference = initial - target
                target_rms = float(np.sqrt(np.mean(target * target)))
                difference_rms = float(np.sqrt(np.mean(difference * difference)))
                row[f"{reference}_rms"] = target_rms
                row[f"delta_{reference}_rms"] = difference_rms
                row[f"delta_{reference}_mse"] = float(np.mean(difference * difference))
                row[f"delta_{reference}_relative_rms"] = difference_rms / max(target_rms, 1e-12)
            rows.append(row)

        summary = {}
        for segment in SEGMENTS:
            selected = [row for row in rows if row["segment"] == segment]
            summary[segment] = {
                key: float(np.mean([row[key] for row in selected]))
                for key in (
                    "initial_rms",
                    "next_rms",
                    "delta_next_rms",
                    "delta_next_mse",
                    "delta_next_relative_rms",
                    "final_rms",
                    "delta_final_rms",
                    "delta_final_mse",
                    "delta_final_relative_rms",
                    "reference_trajectory_rms",
                    "delta_reference_trajectory_rms",
                    "delta_reference_trajectory_mse",
                    "delta_reference_trajectory_relative_rms",
                )
            }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "query": query,
                    "seed": seed,
                    "initial_checkpoint": str(checkpoints[0]),
                    "next_checkpoint": str(checkpoints[1]),
                    "final_checkpoint": str(checkpoints[-1]),
                    "reference_trajectory_parameter_source": "final_checkpoint_ema",
                    "rows": rows,
                    "summary": summary,
                },
                indent=2,
            )
        )
        print(f"[noise-thirds] wrote {output}", flush=True)
    finally:
        if not args.keep_temporary_trajectories and temp_root.exists():
            shutil.rmtree(temp_root)
            print(f"[noise-thirds] deleted temporary trajectory: {temp_root}", flush=True)


def print_summary(root: Path, args: argparse.Namespace) -> None:
    grouped = defaultdict(list)
    by_timestep = defaultdict(list)
    files = []
    for spec in build_query_specs(args):
        path = result_path(root, args, spec)
        if not path.is_file():
            continue
        files.append(path)
        payload = json.loads(path.read_text())
        for row in payload["rows"]:
            by_timestep[(int(row["position"]), int(row["timestep"]), row["segment"])].append(row)
        for segment, values in payload["summary"].items():
            for key, value in values.items():
                grouped[(segment, key)].append(float(value))
    print(f"\nresults={len(files)}/{len(build_query_specs(args))}")
    print("\nPer timestep, ckpt0 raw vs reference-trajectory model (final checkpoint EMA):")
    print(
        f"{'segment':15s} {'position':>8s} {'t':>5s} {'eps0_rms':>10s} "
        f"{'ref_rms':>10s} {'delta_rms':>10s} {'delta_mse':>10s} {'relative':>10s} {'n':>4s}"
    )
    print("-" * 102)
    for (position, timestep, segment), rows in sorted(by_timestep.items()):
        keys = (
            "initial_rms",
            "reference_trajectory_rms",
            "delta_reference_trajectory_rms",
            "delta_reference_trajectory_mse",
            "delta_reference_trajectory_relative_rms",
        )
        means = [float(np.mean([row[key] for row in rows])) for key in keys]
        print(
            f"{segment:15s} {position:8d} {timestep:5d} "
            f"{means[0]:10.5f} {means[1]:10.5f} {means[2]:10.5f} "
            f"{means[3]:10.6f} {means[4]:10.5f} {len(rows):4d}"
        )

    print("\nMean by temporal third:")
    print(
        f"{'segment':15s} {'comparison':26s} {'eps0_rms':>10s} "
        f"{'ref_rms':>10s} {'delta_rms':>10s} {'delta_mse':>10s} {'relative':>10s} {'n':>4s}"
    )
    print("-" * 106)
    for segment in SEGMENTS:
        for reference in ("next", "final", "reference_trajectory"):
            keys = (
                "initial_rms",
                f"{reference}_rms",
                f"delta_{reference}_rms",
                f"delta_{reference}_mse",
                f"delta_{reference}_relative_rms",
            )
            arrays = [grouped[(segment, key)] for key in keys]
            means = [sum(values) / len(values) if values else float("nan") for values in arrays]
            print(
                f"{segment:15s} {'ckpt0-'+reference:26s} "
                f"{means[0]:10.5f} {means[1]:10.5f} {means[2]:10.5f} "
                f"{means[3]:10.6f} {means[4]:10.5f} {len(arrays[0]):4d}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare initial-checkpoint raw predicted noise with next raw, final raw, "
            "and the final EMA model that generated the reference trajectory."
        )
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--experiment", default="cifar5_multi_exp1")
    parser.add_argument("--size", type=int, default=10000)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num-queries", type=int, default=20)
    parser.add_argument("--random-query-seed", type=int, default=0)
    parser.add_argument("--initial-seed-start", type=int, default=1000)
    parser.add_argument("--initial-seeds", default="")
    parser.add_argument("--extra-prompted-queries", default="")
    parser.add_argument("--extra-initial-seed", type=int, default=0)
    parser.add_argument(
        "--output-name",
        default="initial_checkpoint_noise_temporal_thirds_reference_ema",
    )
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--gpu-per-node", type=int, default=4)
    parser.add_argument("--cpus-per-worker", type=int, default=8)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--slot-backend", choices=("local", "ibrun", "srun"), default="local")
    parser.add_argument("--use-task-affinity", action="store_true")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--keep-temporary-trajectories", action="store_true")
    parser.add_argument("--worker-spec-index", type=int, default=-1)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    repo_root = root.parent.parent
    specs = build_query_specs(args)
    if not args.execute:
        print_summary(root, args)
        print("[dry-run] add --execute to calculate missing diagnostics")
        return
    if args.worker_spec_index >= 0:
        run_worker(root, args, args.worker_spec_index)
        return

    gpus = parse_gpus(args)
    worker_gpu_ids = worker_gpus(args, gpus)
    jobs = []
    for spec_index, spec in enumerate(specs):
        if result_path(root, args, spec).is_file():
            continue
        slot = slot_for(spec_index, len(worker_gpu_ids))
        jobs.append(
            Job(
                name=f"noise_thirds_{query_tag(str(spec['query']))}_seed_{spec['initial_seed']}",
                cmd=[sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker-spec-index", str(spec_index)],
                cwd=repo_root,
                env=gpu_env(os.environ.copy(), worker_gpu_ids[slot]),
                log_path=(
                    root / "result" / args.experiment / "logs" / "noise_temporal_thirds"
                    / f"query_{query_tag(str(spec['query']))}_seed_{spec['initial_seed']}_gpu_{worker_gpu_ids[slot]}.log"
                ),
                slot=slot,
            )
        )
    run_parallel_jobs(
        jobs,
        args=args,
        execute=True,
        max_parallel=max(1, min(args.max_parallel, len(worker_gpu_ids))),
    )
    print_summary(root, args)


if __name__ == "__main__":
    main()
