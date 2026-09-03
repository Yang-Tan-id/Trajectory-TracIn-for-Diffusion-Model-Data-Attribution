#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


DEFAULT_NAMESPACE = "raw_nextckpt_shared_gradient_once_100ts"
DEFAULT_RANGES = "1-2500,2501-5000,5001-7500,7501-10000"


def env_flag(value: bool) -> str:
    return "1" if value else "0"


def run(cmd: list[str], *, cwd: Path, env: dict[str, str], execute: bool) -> None:
    prefix = "RUN" if execute else "DRY"
    print(f"[{prefix}] {' '.join(cmd)}", flush=True)
    if execute:
        subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run CIFAR5 TrajTracIn shared_gradient_once: raw next-checkpoint query objective, "
            "100 evenly spaced train timesteps, one MC sample per timestep, and one train "
            "loss-gradient backward pass per checkpoint."
        )
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--experiment", default=os.environ.get("EXPERIMENT_TAG", "cifar5_multi_exp1"))
    parser.add_argument("--size", type=int, default=int(os.environ.get("CIFAR5_MULTI_SIZE", "10000")))
    parser.add_argument("--train-seed", type=int, default=int(os.environ.get("TRAIN_SEED", "42")))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("JAX_EPOCHS", "200")))
    parser.add_argument("--namespace", default=os.environ.get("ATTRIBUTION_ARTIFACT_NAMESPACE", DEFAULT_NAMESPACE))
    parser.add_argument("--num-queries", type=int, default=int(os.environ.get("NUM_RANDOM_PROMPTED_QUERIES", "20")))
    parser.add_argument("--random-query-seed", type=int, default=int(os.environ.get("RANDOM_QUERY_SEED", "0")))
    parser.add_argument("--initial-seed-start", type=int, default=int(os.environ.get("INITIAL_SEED_START", "1000")))
    parser.add_argument("--initial-seeds", default=os.environ.get("INITIAL_SEEDS", ""))
    parser.add_argument("--lds-m", type=int, default=int(os.environ.get("LDS_M", "64")))
    parser.add_argument("--lds-percentage", type=float, default=float(os.environ.get("LDS_DATASET_PERCENTAGE", "25")))
    parser.add_argument("--lds-subset-seeds", default=os.environ.get("LDS_SUBSET_SEEDS", "0,1,2"))
    parser.add_argument("--gpus", default=os.environ.get("GPU_IDS"))
    parser.add_argument("--slots", type=int, default=int(os.environ.get("GPU_SLOTS", "0")) or None)
    parser.add_argument("--gpu-per-node", type=int, default=int(os.environ.get("GPU_PER_NODE", "4")))
    parser.add_argument("--cpus-per-worker", type=int, default=int(os.environ.get("CPUS_PER_WORKER", "8")))
    parser.add_argument("--slot-backend", choices=("local", "ibrun", "srun"), default=os.environ.get("TACC_SLOT_BACKEND", "local"))
    parser.add_argument("--max-parallel", type=int, default=int(os.environ.get("MAX_PARALLEL", "0")) or None)
    parser.add_argument(
        "--score-index-ranges",
        "--index-ranges",
        dest="index_ranges",
        default=os.environ.get("ATTRIBUTION_INDEX_RANGES", os.environ.get("SCORE_INDEX_RANGES", DEFAULT_RANGES)),
    )
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", "python3"))
    parser.add_argument("--skip-train-gradient", action="store_true")
    parser.add_argument("--only-train-gradient", action="store_true")
    parser.add_argument("--skip-sampling", action="store_true")
    parser.add_argument("--skip-query-gradient", action="store_true")
    parser.add_argument("--skip-lds-eval", action="store_true")
    parser.add_argument("--only-lds-eval", action="store_true")
    parser.add_argument("--tracin-score-query-normalize", action="store_true")
    parser.add_argument("--tracin-score-query-normalize-eps", type=float, default=float(os.environ.get("TRACIN_SCORE_QUERY_NORMALIZE_EPS", "1e-8")))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("JAX_BFLOAT16", "1")
    env.setdefault("JAX_PREFETCH_SIZE", "1")
    env.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    env.setdefault("TRAJ_TRACIN_PROJ_DIM", "4096")
    env.setdefault("PROJECTED_CACHE_DIM", "4096")
    env.setdefault("PROJECTED_DIMS", "4096")
    env["ATTRIBUTION_ARTIFACT_NAMESPACE"] = args.namespace
    env["TRAJ_ATTRIBUTION_ARTIFACT_NAMESPACE"] = args.namespace
    env["TRAJ_QUERY_OBJECTIVE"] = "trajectory_next_checkpoint_noise_mse"
    env["TRAJ_PARAMETER_SOURCE"] = "raw"
    env["TRACIN_PARAMETER_SOURCE"] = "raw"
    env["TRACIN_USE_SHARED_TRAIN_GRADIENT"] = "1"
    env["TRAJ_TRACIN_TRAIN_AGGREGATE_TIMESTAMPS"] = "1"
    env["TRAJ_TRACIN_TRAIN_AGGREGATE_NUM_TIMESTEPS"] = "100"
    env["TRAJ_TRACIN_TRAIN_TIMESTAMP_CHUNK_SIZE"] = "100"
    env["TRAJ_NUM_SNAPSHOTS"] = "100"
    env["TRAJ_TRAIN_MC_SAMPLES"] = "1"
    env["TRACIN_SCORE_QUERY_NORMALIZE"] = env_flag(args.tracin_score_query_normalize)
    env["TRACIN_SCORE_QUERY_NORMALIZE_EPS"] = str(args.tracin_score_query_normalize_eps)

    gpu_args: list[str] = []
    if args.gpus:
        gpu_args.extend(["--gpus", args.gpus])
    if args.slots is not None:
        gpu_args.extend(["--slots", str(args.slots)])
    gpu_args.extend(
        [
            "--gpu-per-node",
            str(args.gpu_per_node),
            "--cpus-per-worker",
            str(args.cpus_per_worker),
            "--slot-backend",
            args.slot_backend,
        ]
    )
    if args.max_parallel is not None:
        gpu_args.extend(["--max-parallel", str(args.max_parallel)])

    print("CIFAR5 TrajTracIn shared_gradient_once")
    print(f"repo={root.parent.parent}")
    print(f"namespace={args.namespace}")
    print("mode=raw + next checkpoint")
    print("train gradient=100 evenly spaced timesteps, 1 MC/timestep, one backward per checkpoint")
    print("expected train terms=49 for next-checkpoint objective")
    print(f"index ranges={args.index_ranges}")

    common = [
        "--execute",
        "--experiment",
        args.experiment,
        "--size",
        str(args.size),
        "--train-seed",
        str(args.train_seed),
        "--epochs",
        str(args.epochs),
    ]
    if not args.execute:
        common = common[1:]

    if not args.skip_train_gradient and not args.only_lds_eval:
        train_cmd = [
            args.python_bin,
            str(root / "script" / "run_cifar5_multi_attribution_distributed.py"),
            *common,
            "--artifact-namespace",
            args.namespace,
            *gpu_args,
            "--skip-das",
            "--skip-query-gradient",
            "--skip-lds-eval",
            "--no-unprompted",
            "--only-train-gradient",
            "--score-index-ranges",
            args.index_ranges,
        ]
        run(train_cmd, cwd=root, env=env, execute=args.execute)

    if args.only_train_gradient:
        print("[done] shared_gradient_once train-gradient-only")
        return

    random_cmd = [
        args.python_bin,
        str(root / "script" / "run_cifar5_multi_random_prompted_queries.py"),
        *common,
        "--num-queries",
        str(args.num_queries),
        "--random-query-seed",
        str(args.random_query_seed),
        "--initial-seed-start",
        str(args.initial_seed_start),
        "--lds-m",
        str(args.lds_m),
        "--lds-percentage",
        str(args.lds_percentage),
        "--lds-subset-seeds",
        args.lds_subset_seeds,
        "--artifact-namespace",
        args.namespace,
        "--traj-artifact-namespace",
        args.namespace,
        "--namespace-query-gradient",
        *gpu_args,
        "--skip-das",
        "--score-index-ranges",
        args.index_ranges,
    ]
    if args.initial_seeds:
        random_cmd.extend(["--initial-seeds", args.initial_seeds])
    if args.skip_sampling:
        random_cmd.append("--skip-sampling")
    if args.skip_query_gradient:
        random_cmd.append("--skip-query-gradient")
    if args.skip_lds_eval:
        random_cmd.append("--skip-lds-eval")
    if args.only_lds_eval:
        random_cmd.append("--only-lds-eval")
    if args.tracin_score_query_normalize:
        random_cmd.extend(
            [
                "--tracin-score-query-normalize",
                "--tracin-score-query-normalize-eps",
                str(args.tracin_score_query_normalize_eps),
            ]
        )
    run(random_cmd, cwd=root, env=env, execute=args.execute)


if __name__ == "__main__":
    main()
