#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


TRAIN_ARTIFACT = "train_datapoint_gradient_artifact.npz"
DEFAULT_RANGES = "1-2500,2501-5000,5001-7500,7501-10000"


def parse_ranges(text: str) -> list[tuple[int, int]]:
    ranges = []
    for item in text.replace(";", ",").split(","):
        start, end = item.strip().replace(":", "-").split("-", 1)
        ranges.append((int(start), int(end)))
    return ranges


def train_shard(root: Path, experiment: str, seed: int, namespace: str, start: int, end: int) -> Path:
    return (
        root
        / "result"
        / experiment
        / "model"
        / "prompted_solo"
        / f"seed_{seed}_train_gradient_{namespace}"
        / "traj_tracin"
        / "datapoint_shards"
        / f"range_{start}_{end}"
        / TRAIN_ARTIFACT
    )


def link_train_shards(args: argparse.Namespace) -> None:
    for start, end in parse_ranges(args.score_index_ranges):
        source = train_shard(
            args.root, args.experiment, args.train_seed, args.train_namespace, start, end
        )
        target = train_shard(
            args.root, args.experiment, args.train_seed, args.namespace, start, end
        )
        print(f"[train-link] {target} -> {source}", flush=True)
        if not args.execute:
            continue
        if not source.is_file():
            raise FileNotFoundError(f"missing reusable train shard: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            if target.resolve() == source.resolve():
                continue
            target.unlink()
        elif target.exists():
            raise FileExistsError(
                f"refusing to replace non-symlink train artifact in the new namespace: {target}"
            )
        target.symlink_to(source.resolve())


def cleanup_trajectory_caches(root: Path, experiment: str, namespace: str, *, execute: bool) -> int:
    sample_root = root / "result" / experiment / "sample" / "cifar"
    pattern = f"seed_*_query_gradient_{namespace}/traj_tracin/query_gradient_artifact.npz.trajectory_cache"
    caches = sorted(sample_root.glob(f"prompt_*/model_*/{pattern}"))
    for cache in caches:
        print(f"[trajectory-cache] delete {cache}", flush=True)
        if execute:
            shutil.rmtree(cache)
    print(f"[trajectory-cache] matched={len(caches)}", flush=True)
    return len(caches)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run CIFAR5 TrajTracIn with a next-checkpoint trajectory implied-noise query target "
            "while reusing existing 10x10 train gradients."
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
    parser.add_argument("--namespace", default="raw_nextckpt_school_traj_next_trajectory_implied_noise_10x10")
    parser.add_argument("--train-namespace", default="raw_nextckpt_school_traj_aligned_10x10")
    parser.add_argument(
        "--query-objective",
        choices=(
            "trajectory_next_checkpoint_implied_noise_mse",
            "trajectory_next_checkpoint_trajectory_noise_mse",
        ),
        default="trajectory_next_checkpoint_implied_noise_mse",
    )
    parser.add_argument(
        "--num-traj-snapshots",
        type=int,
        default=10,
        help="Number of uniformly spaced trajectory timesteps in each query artifact.",
    )
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--gpu-per-node", type=int, default=4)
    parser.add_argument("--cpus-per-worker", type=int, default=8)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--score-index-ranges", default=DEFAULT_RANGES)
    parser.add_argument("--lds-m", type=int, default=64)
    parser.add_argument("--lds-percentage", type=float, default=25.0)
    parser.add_argument("--lds-subset-seeds", default="0,1,2")
    parser.add_argument("--skip-query-gradient", action="store_true")
    parser.add_argument("--only-query-gradient", action="store_true")
    parser.add_argument("--skip-lds-eval", action="store_true")
    parser.add_argument("--keep-trajectory-cache", action="store_true")
    parser.add_argument("--cleanup-trajectory-cache-only", action="store_true")
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", "python3"))
    args = parser.parse_args()

    if args.only_query_gradient and args.skip_query_gradient:
        parser.error("--only-query-gradient cannot be combined with --skip-query-gradient")
    if args.num_traj_snapshots < 1 or args.num_traj_snapshots > 1000:
        parser.error("--num-traj-snapshots must be between 1 and 1000")
    if args.namespace == "raw_nextckpt_school_traj_next_trajectory_implied_noise_10x10" and (
        args.num_traj_snapshots != 10
        or args.query_objective != "trajectory_next_checkpoint_implied_noise_mse"
    ):
        parser.error("set a distinct --namespace for a different objective or snapshot count")

    args.root = Path(__file__).resolve().parents[1]
    if args.cleanup_trajectory_cache_only:
        cleanup_trajectory_caches(
            args.root, args.experiment, args.namespace, execute=args.execute
        )
        return

    link_train_shards(args)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    env.setdefault("TRAJ_TRACIN_PROJ_DIM", "4096")
    env.setdefault("PROJECTED_CACHE_DIM", "4096")
    env.setdefault("PROJECTED_DIMS", "4096")
    env["TRAJ_QUERY_OBJECTIVE"] = args.query_objective
    env["TRAJ_PARAMETER_SOURCE"] = "raw"
    env["TRACIN_PARAMETER_SOURCE"] = "raw"
    env["TRAJ_NUM_SNAPSHOTS"] = str(args.num_traj_snapshots)
    env["TRAJ_TRAIN_MC_SAMPLES"] = "10"
    env["TRAJ_QUERY_USE_CONFIG_SNAPSHOTS"] = "1"
    env["TRACIN_ALIGN_TERMS_BY_CKPT_TIMESTEP"] = "1"
    env["TRAJ_TRACIN_DELETE_CHECKPOINT_TRAJECTORY_CACHE"] = (
        "0" if args.keep_trajectory_cache else "1"
    )

    cmd = [
        args.python_bin,
        str(args.root / "script" / "run_cifar5_multi_random_prompted_queries.py"),
        "--experiment",
        args.experiment,
        "--size",
        str(args.size),
        "--train-seed",
        str(args.train_seed),
        "--epochs",
        str(args.epochs),
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
        "--gpus",
        args.gpus,
        "--slots",
        str(args.slots),
        "--gpu-per-node",
        str(args.gpu_per_node),
        "--cpus-per-worker",
        str(args.cpus_per_worker),
        "--max-parallel",
        str(args.max_parallel),
        "--score-index-ranges",
        args.score_index_ranges,
        "--skip-das",
        "--skip-sampling",
    ]
    if args.initial_seeds:
        cmd.extend(["--initial-seeds", args.initial_seeds])
    if args.skip_query_gradient:
        cmd.append("--skip-query-gradient")
    if args.only_query_gradient:
        cmd.append("--only-query-gradient")
    if args.skip_lds_eval:
        cmd.append("--skip-lds-eval")
    if args.execute:
        cmd.insert(2, "--execute")

    print("CIFAR5 next-checkpoint trajectory implied-noise TrajTracIn", flush=True)
    print(f"namespace={args.namespace}", flush=True)
    print(f"query_objective={args.query_objective}", flush=True)
    print(f"reused_train_namespace={args.train_namespace}", flush=True)
    print(f"query_trajectory_snapshots={args.num_traj_snapshots}", flush=True)
    print(f"delete_cache_after_success={int(not args.keep_trajectory_cache)}", flush=True)
    print("command:", " ".join(cmd), flush=True)
    if args.execute:
        subprocess.run(cmd, cwd=args.root, env=env, check=True)


if __name__ == "__main__":
    main()
