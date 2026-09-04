#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from run_cifar5_multi_random_prompted_queries import build_query_specs, query_tag


DEFAULT_QUERY_NS = "raw_nextckpt_school_traj_10x10"
DEFAULT_ADDON_NS = "raw_nextckpt_school_traj_addon_mid10"
DEFAULT_RANGES = "1-2500,2501-5000,5001-7500,7501-10000"
DEFAULT_ADDON_100GRID_INDICES = (5, 16, 27, 38, 49, 60, 71, 82, 93, 98)


def run(cmd: list[str], *, cwd: Path, env: dict[str, str], execute: bool) -> None:
    prefix = "RUN" if execute else "DRY"
    print(f"[{prefix}] {' '.join(cmd)}", flush=True)
    if execute:
        subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def snapshot_positions_from_100grid(indices: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(idx * 999 // 99) for idx in indices)


def copy_query_artifacts(root: Path, *, experiment: str, train_seed: int, epochs: int, old_ns: str, new_ns: str, specs: list[dict[str, int | str]], execute: bool) -> None:
    for spec in specs:
        query = str(spec["query"])
        seed = int(spec["initial_seed"])
        base = (
            root
            / "result"
            / experiment
            / "sample"
            / "cifar"
            / f"prompt_{query_tag(query)}"
            / f"model_prompted_solo__ckpt_seed_{train_seed}_epoch_{epochs:04d}"
        )
        src = base / f"seed_{seed:06d}_query_gradient_{old_ns}" / "traj_tracin"
        dst = base / f"seed_{seed:06d}_query_gradient_{new_ns}" / "traj_tracin"
        if not execute:
            print(f"[DRY][copy-query] {src} -> {dst}", flush=True)
            continue
        if (dst / "query_gradient_artifact.npz").is_file():
            print(f"[skip] query already copied: {dst}", flush=True)
            continue
        if not (src / "query_gradient_artifact.npz").is_file():
            raise FileNotFoundError(f"missing source query artifact: {src / 'query_gradient_artifact.npz'}")
        print(f"[copy-query] {src} -> {dst}", flush=True)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a raw next-checkpoint TrajTracIn add-on train-gradient artifact at 10 extra "
            "midpoint timesteps, then score/LDS the add-on contribution only."
        )
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--experiment", default=os.environ.get("EXPERIMENT_TAG", "cifar5_multi_exp1"))
    parser.add_argument("--size", type=int, default=int(os.environ.get("CIFAR5_MULTI_SIZE", "10000")))
    parser.add_argument("--train-seed", type=int, default=int(os.environ.get("TRAIN_SEED", "42")))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("JAX_EPOCHS", "200")))
    parser.add_argument("--query-namespace", default=os.environ.get("TRAJ_QUERY_NAMESPACE", DEFAULT_QUERY_NS))
    parser.add_argument("--addon-namespace", default=os.environ.get("ADDON_TRAJ_TRAIN_NAMESPACE", DEFAULT_ADDON_NS))
    parser.add_argument("--num-queries", type=int, default=int(os.environ.get("NUM_RANDOM_PROMPTED_QUERIES", "20")))
    parser.add_argument("--random-query-seed", type=int, default=int(os.environ.get("RANDOM_QUERY_SEED", "0")))
    parser.add_argument("--initial-seed-start", type=int, default=int(os.environ.get("INITIAL_SEED_START", "1000")))
    parser.add_argument("--initial-seeds", default=os.environ.get("INITIAL_SEEDS", ""))
    parser.add_argument("--extra-prompted-queries", default=os.environ.get("EXTRA_PROMPTED_QUERIES", ""))
    parser.add_argument("--extra-initial-seed", type=int, default=int(os.environ.get("EXTRA_INITIAL_SEED", "0")))
    parser.add_argument("--lds-m", type=int, default=int(os.environ.get("LDS_M", "64")))
    parser.add_argument("--lds-percentage", type=float, default=float(os.environ.get("LDS_DATASET_PERCENTAGE", "25")))
    parser.add_argument("--lds-subset-seeds", default=os.environ.get("LDS_SUBSET_SEEDS", "0,1,2"))
    parser.add_argument("--gpus", default=os.environ.get("GPU_IDS"))
    parser.add_argument("--slots", type=int, default=int(os.environ.get("GPU_SLOTS", "0")) or None)
    parser.add_argument("--gpu-per-node", type=int, default=int(os.environ.get("GPU_PER_NODE", "4")))
    parser.add_argument("--cpus-per-worker", type=int, default=int(os.environ.get("CPUS_PER_WORKER", "8")))
    parser.add_argument("--slot-backend", choices=("local", "ibrun", "srun"), default=os.environ.get("TACC_SLOT_BACKEND", "local"))
    parser.add_argument("--max-parallel", type=int, default=int(os.environ.get("MAX_PARALLEL", "0")) or None)
    parser.add_argument("--score-index-ranges", default=os.environ.get("ATTRIBUTION_INDEX_RANGES", os.environ.get("SCORE_INDEX_RANGES", DEFAULT_RANGES)))
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", "python3"))
    parser.add_argument("--skip-addon-train", action="store_true")
    parser.add_argument("--skip-query-copy", action="store_true")
    parser.add_argument("--skip-lds-eval", action="store_true")
    parser.add_argument("--only-train-gradient", action="store_true")
    parser.add_argument("--addon-100grid-indices", default=",".join(str(x) for x in DEFAULT_ADDON_100GRID_INDICES))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    repo_root = root.parent.parent
    addon_indices = tuple(int(x) for x in args.addon_100grid_indices.replace(",", " ").split() if x.strip())
    addon_positions = snapshot_positions_from_100grid(addon_indices)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("JAX_BFLOAT16", "1")
    env.setdefault("JAX_PREFETCH_SIZE", "1")
    env.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    env.setdefault("TRAJ_TRACIN_PROJ_DIM", "4096")
    env.setdefault("PROJECTED_CACHE_DIM", "4096")
    env.setdefault("PROJECTED_DIMS", "4096")
    env["TRAJ_QUERY_OBJECTIVE"] = "trajectory_next_checkpoint_noise_mse"
    env["TRAJ_PARAMETER_SOURCE"] = "raw"
    env["TRACIN_PARAMETER_SOURCE"] = "raw"
    env["TRACIN_USE_SHARED_TRAIN_GRADIENT"] = "0"
    env["TRACIN_ALIGN_TERMS_BY_CKPT_TIMESTEP"] = "1"
    env["TRAJ_NUM_SNAPSHOTS"] = str(len(addon_positions))
    env["TRAJ_SNAPSHOT_POSITIONS"] = ",".join(str(x) for x in addon_positions)
    env.setdefault("TRAJ_TRAIN_MC_SAMPLES", "10")
    env.setdefault("TRAJ_SCORE_BATCH_SIZE", "8")

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

    common = [
        "--experiment",
        args.experiment,
        "--size",
        str(args.size),
        "--train-seed",
        str(args.train_seed),
        "--epochs",
        str(args.epochs),
    ]
    if args.execute:
        common.insert(0, "--execute")

    print("CIFAR5 raw next-checkpoint TrajTracIn add-on 20-term runner")
    print(f"repo={repo_root}")
    print(f"query_namespace={args.query_namespace}")
    print(f"addon_namespace={args.addon_namespace}")
    print(f"addon_100grid_indices={addon_indices}")
    print(f"addon_snapshot_positions={addon_positions}")
    print(f"score_index_ranges={args.score_index_ranges}")

    if not args.skip_addon_train:
        train_cmd = [
            args.python_bin,
            str(root / "script" / "run_cifar5_multi_attribution_distributed.py"),
            *common,
            "--artifact-namespace",
            args.addon_namespace,
            *gpu_args,
            "--skip-das",
            "--skip-query-gradient",
            "--skip-lds-eval",
            "--no-unprompted",
            "--only-train-gradient",
            "--score-index-ranges",
            args.score_index_ranges,
        ]
        run(train_cmd, cwd=root, env=env, execute=args.execute)

    specs = build_query_specs(args)
    if not args.skip_query_copy:
        copy_query_artifacts(
            root,
            experiment=args.experiment,
            train_seed=args.train_seed,
            epochs=args.epochs,
            old_ns=args.query_namespace,
            new_ns=args.addon_namespace,
            specs=specs,
            execute=args.execute,
        )

    if args.only_train_gradient:
        print("[done] add-on train only")
        return

    score_env = env.copy()
    score_cmd = [
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
        args.addon_namespace,
        "--traj-artifact-namespace",
        args.addon_namespace,
        "--namespace-query-gradient",
        *gpu_args,
        "--skip-das",
        "--skip-sampling",
        "--skip-query-gradient",
        "--score-index-ranges",
        args.score_index_ranges,
    ]
    if args.skip_lds_eval:
        score_cmd.append("--skip-lds-eval")
    run(score_cmd, cwd=root, env=score_env, execute=args.execute)


if __name__ == "__main__":
    main()
