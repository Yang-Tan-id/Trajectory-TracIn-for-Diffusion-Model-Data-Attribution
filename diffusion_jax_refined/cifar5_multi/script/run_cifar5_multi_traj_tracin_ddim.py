#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
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


def train_shard(
    root: Path,
    experiment: str,
    train_seed: int,
    namespace: str,
    start: int,
    end: int,
) -> Path:
    return (
        root
        / "result"
        / experiment
        / "model"
        / "prompted_solo"
        / f"seed_{train_seed}_train_gradient_{namespace}"
        / "traj_tracin"
        / "datapoint_shards"
        / f"range_{start}_{end}"
        / TRAIN_ARTIFACT
    )


def link_reusable_train_shards(args: argparse.Namespace) -> None:
    for start, end in parse_ranges(args.score_index_ranges):
        source = train_shard(
            args.root,
            args.experiment,
            args.train_seed,
            args.train_namespace,
            start,
            end,
        )
        target = train_shard(
            args.root,
            args.experiment,
            args.train_seed,
            args.namespace,
            start,
            end,
        )
        print(f"[train-reuse] {target} -> {source}", flush=True)
        if not args.execute:
            continue
        if not source.is_file():
            raise FileNotFoundError(f"missing reusable train shard: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() and target.resolve() == source.resolve():
            continue
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"refusing to replace existing train artifact: {target}")
        target.symlink_to(source.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run CIFAR5 next-raw TrajTracIn on deterministic DDIM eta=0 "
            "trajectories while reusing existing train gradients."
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
    parser.add_argument(
        "--namespace",
        default="raw_nextckpt_school_traj_ddim_eta0_10x10",
    )
    parser.add_argument(
        "--train-namespace",
        default="raw_nextckpt_school_traj_aligned_10x10",
    )
    parser.add_argument("--sample-root-name", default="sample_ddim_eta0")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--gpu-per-node", type=int, default=4)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--score-index-ranges", default=DEFAULT_RANGES)
    parser.add_argument("--lds-subset-seeds", default="0,1,2")
    parser.add_argument("--skip-sampling", action="store_true")
    parser.add_argument(
        "--include-das",
        action="store_true",
        help="Generate/evaluate DAS alongside TrajTracIn on the same DDIM trajectories.",
    )
    parser.add_argument("--skip-query-gradient", action="store_true")
    parser.add_argument("--skip-lds-eval", action="store_true")
    parser.add_argument("--only-query-gradient", action="store_true")
    parser.add_argument("--only-lds-eval", action="store_true")
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", "python3"))
    args = parser.parse_args()

    args.root = Path(__file__).resolve().parents[1]
    if not args.only_query_gradient:
        link_reusable_train_shards(args)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    env["DIFFUSION_TRAJECTORY_SAMPLER"] = "ddim_eta0"
    env["LDS_TRAJECTORY_SAMPLER"] = "ddim_eta0"
    env["TRAJ_QUERY_OBJECTIVE"] = "trajectory_next_checkpoint_noise_mse"
    env["TRAJ_PARAMETER_SOURCE"] = "raw"
    env["TRACIN_PARAMETER_SOURCE"] = "raw"
    env["TRAJ_NUM_SNAPSHOTS"] = "10"
    env["TRAJ_TRAIN_MC_SAMPLES"] = "10"
    env["TRACIN_ALIGN_TERMS_BY_CKPT_TIMESTEP"] = "1"

    cmd = [
        args.python_bin,
        str(args.root / "script" / "run_cifar5_multi_random_prompted_queries.py"),
        "--experiment", args.experiment,
        "--size", str(args.size),
        "--train-seed", str(args.train_seed),
        "--epochs", str(args.epochs),
        "--num-queries", str(args.num_queries),
        "--random-query-seed", str(args.random_query_seed),
        "--initial-seed-start", str(args.initial_seed_start),
        "--sample-root-name", args.sample_root_name,
        "--artifact-namespace", args.namespace,
        "--traj-artifact-namespace", args.namespace,
        "--namespace-query-gradient",
        "--gpus", args.gpus,
        "--slots", str(args.slots),
        "--gpu-per-node", str(args.gpu_per_node),
        "--max-parallel", str(args.max_parallel),
        "--score-index-ranges", args.score_index_ranges,
        "--lds-subset-seeds", args.lds_subset_seeds,
    ]
    if not args.include_das:
        cmd.append("--skip-das")
    if args.initial_seeds:
        cmd.extend(["--initial-seeds", args.initial_seeds])
    for flag in (
        "skip_sampling",
        "skip_query_gradient",
        "skip_lds_eval",
        "only_query_gradient",
        "only_lds_eval",
    ):
        if getattr(args, flag):
            cmd.append("--" + flag.replace("_", "-"))
    if args.execute:
        cmd.insert(2, "--execute")

    print("CIFAR5 deterministic DDIM eta=0 TrajTracIn", flush=True)
    print(f"namespace={args.namespace}", flush=True)
    print(f"sample_root={args.sample_root_name}", flush=True)
    print(f"reused_train_namespace={args.train_namespace}", flush=True)
    print("command:", " ".join(cmd), flush=True)
    if args.execute:
        subprocess.run(cmd, cwd=args.root, env=env, check=True)


if __name__ == "__main__":
    main()
