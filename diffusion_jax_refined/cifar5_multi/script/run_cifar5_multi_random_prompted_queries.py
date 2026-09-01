#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import subprocess
from argparse import Namespace
from pathlib import Path

from run_cifar5_multi_experiment import (
    LABELS,
    Job,
    TARGET_FUNCTIONS,
    artifact_namespace,
    attribution_score_dirs,
    base_env,
    gpu_env,
    lds_eval_out_dir,
    lds_model_dirs,
    log_root,
    parse_gpus,
    query_tag,
    run,
    run_parallel_jobs,
    slot_for,
    worker_gpus,
)

from run_cifar5_multi_attribution_distributed import (
    das_global_gram_path,
    query_gradient_artifact_path,
    score_complete,
    shard_artifact_path,
    split_1based_ranges,
    train_artifact_complete,
)


TRAIN_ARTIFACT = "train_datapoint_gradient_artifact.npz"
DEFAULT_DAS_SWEEP = "0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000,2000,5000,10000"


def prompted_combos() -> list[str]:
    return [",".join(combo) for combo in itertools.combinations(LABELS, 3)]


def parse_query_list(text: str) -> list[str]:
    return [query.strip() for query in text.replace("|", ";").split(";") if query.strip()]


def build_query_specs(args: argparse.Namespace) -> list[dict[str, int | str]]:
    seeds = [int(x.strip()) for x in str(args.initial_seeds or "").replace(",", " ").split() if x.strip()]
    if seeds and len(seeds) != args.num_queries:
        raise ValueError("--initial-seeds length must match --num-queries")
    if not seeds:
        seeds = [args.initial_seed_start + i for i in range(args.num_queries)]

    rng = random.Random(args.random_query_seed)
    combos = prompted_combos()
    queries = [rng.choice(combos) for _ in range(args.num_queries)]
    specs = [
        {"query_id": i, "query": query, "initial_seed": seeds[i], "query_tag": query_tag(query)}
        for i, query in enumerate(queries)
    ]
    for query in parse_query_list(args.extra_prompted_queries):
        specs.append(
            {
                "query_id": len(specs),
                "query": query,
                "initial_seed": int(args.extra_initial_seed),
                "query_tag": query_tag(query),
            }
        )
    return specs


def args_for_seed(args: argparse.Namespace, seed: int, *, namespace: str = "") -> argparse.Namespace:
    child = Namespace(**vars(args))
    child.sample_seeds = str(seed)
    child.artifact_namespace = namespace
    child.root = args.root
    return child


def sample_seed_dir(root: Path, args: argparse.Namespace, query: str, seed: int) -> Path:
    return (
        root
        / "result"
        / args.experiment
        / "sample"
        / "cifar"
        / f"prompt_{query_tag(query)}"
        / f"model_prompted_solo__ckpt_seed_{args.train_seed}_epoch_{args.epochs:04d}"
        / f"seed_{seed:06d}"
    )


def sample_complete(path: Path) -> bool:
    return (path / "trajectory_xt.npy").is_file() and (path / "final_state.npy").is_file()


def artifact_complete(path: Path) -> bool:
    return path.is_file()


def query_artifact_complete(path: Path, *, algorithm: str, expected_terms: int) -> bool:
    if not path.is_file():
        return False
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as data:
            if algorithm == "traj_tracin":
                if "query_features" not in data:
                    return False
                return int(np.asarray(data["query_features"]).shape[0]) == int(expected_terms)
            if algorithm == "das":
                if "query_features" not in data:
                    return False
                return int(np.asarray(data["query_features"]).shape[0]) == int(expected_terms)
    except Exception:
        return False
    return True


def run_distributed_for_query(
    args: argparse.Namespace,
    *,
    query: str,
    seed: int,
    namespace: str,
    extra: list[str],
    env: dict[str, str],
) -> None:
    cmd = [
        args.python_bin,
        str(args.root / "script" / "run_cifar5_multi_attribution_distributed.py"),
        "--execute",
        "--experiment",
        args.experiment,
        "--size",
        str(args.size),
        "--train-seed",
        str(args.train_seed),
        "--epochs",
        str(args.epochs),
        "--sample-seeds",
        str(seed),
        "--prompted-queries",
        query,
        "--no-unprompted",
        "--lds-m",
        str(args.lds_m),
        "--lds-percentage",
        str(args.lds_percentage),
        "--lds-subset-seeds",
        args.lds_subset_seeds,
        "--gpus",
        args.gpus_text,
        "--slots",
        str(args.slots),
        "--gpu-per-node",
        str(args.gpu_per_node),
        "--cpus-per-worker",
        str(args.cpus_per_worker),
        "--slot-backend",
        args.slot_backend,
        "--max-parallel",
        str(args.max_parallel),
        *extra,
    ]
    if namespace:
        cmd.extend(["--artifact-namespace", namespace])
    if namespace and args.namespace_query_gradient:
        cmd.append("--namespace-query-gradient")
    if namespace and args.aggregate_traj_train_from_namespace:
        cmd.extend(["--aggregate-traj-train-from-namespace", args.aggregate_traj_train_from_namespace])
    if args.tracin_score_query_normalize:
        env = env | {
            "TRACIN_SCORE_QUERY_NORMALIZE": "1",
            "TRACIN_SCORE_QUERY_NORMALIZE_EPS": str(args.tracin_score_query_normalize_eps),
        }
    run(cmd, env, cwd=args.root, execute=args.execute)


def ensure_train_artifacts(args: argparse.Namespace) -> None:
    if args.skip_das:
        print("[skip] DAS train artifact check")
    else:
        das_args = args_for_seed(args, args.initial_seed_start)
        for mode in ("prompted_solo",):
            gram = das_global_gram_path(args.root, das_args, mode)
            if gram.is_file():
                print(f"[ok] DAS shared train gram for {mode}: {gram}")
            elif args.execute:
                raise FileNotFoundError(f"Missing DAS shared train gram: {gram}")
            else:
                print(f"[warn] missing DAS shared train gram: {gram}")

    if args.skip_traj_tracin:
        print("[skip] TrajTracIn train artifact check")
    else:
        traj_args = args_for_seed(args, args.initial_seed_start, namespace=args.traj_artifact_namespace)
        ranges = split_1based_ranges(args.size, args.slots)
        missing = [
            shard_artifact_path(args.root, traj_args, "prompted_solo", "traj_tracin", start, end)
            for start, end in ranges
            if not train_artifact_complete(
                shard_artifact_path(args.root, traj_args, "prompted_solo", "traj_tracin", start, end),
                expected_points=end - start + 1,
            )
        ]
        if missing and not args.aggregate_traj_train_from_namespace and args.execute:
            raise FileNotFoundError(
                "Missing checkpoint-shared TrajTracIn shard(s). "
                f"Pass --aggregate-traj-train-from-namespace or create them first. First missing: {missing[0]}"
            )
        if missing and args.aggregate_traj_train_from_namespace:
            print(
                f"[info] {len(missing)} TrajTracIn checkpoint-shared shard(s) will be aggregated "
                f"from namespace {args.aggregate_traj_train_from_namespace}"
            )
        elif missing:
            print(f"[warn] missing {len(missing)} TrajTracIn checkpoint-shared shard(s); dry-run continues")
        else:
            print(f"[ok] TrajTracIn checkpoint-shared train shards in namespace {args.traj_artifact_namespace}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DAS and checkpoint-shared TrajTracIn attribution for random CIFAR5_MULTI prompted queries."
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
    parser.add_argument("--extra-prompted-queries", default=os.environ.get("EXTRA_PROMPTED_QUERIES", ""))
    parser.add_argument("--extra-initial-seed", type=int, default=int(os.environ.get("EXTRA_INITIAL_SEED", "0")))
    parser.add_argument("--lds-m", type=int, default=64)
    parser.add_argument("--lds-percentage", type=float, default=25)
    parser.add_argument("--lds-subset-seeds", default="0,1,2")
    parser.add_argument("--lds-epochs", type=int, default=200)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--slots", type=int, default=None)
    parser.add_argument("--gpu-per-node", type=int, default=4)
    parser.add_argument("--cpus-per-worker", type=int, default=8)
    parser.add_argument("--max-parallel", type=int, default=int(os.environ.get("MAX_PARALLEL", "0")) or None)
    parser.add_argument("--slot-backend", choices=("local", "ibrun", "srun"), default=os.environ.get("TACC_SLOT_BACKEND", "local"))
    parser.add_argument("--use-task-affinity", action="store_true")
    parser.add_argument("--skip-sampling", action="store_true")
    parser.add_argument("--skip-query-gradient", action="store_true")
    parser.add_argument("--skip-das", action="store_true")
    parser.add_argument("--skip-traj-tracin", action="store_true")
    parser.add_argument("--skip-lds-eval", action="store_true")
    parser.add_argument("--only-lds-eval", action="store_true")
    parser.add_argument("--artifact-namespace", default=os.environ.get("ATTRIBUTION_ARTIFACT_NAMESPACE", ""))
    parser.add_argument("--namespace-query-gradient", action="store_true")
    parser.add_argument("--traj-artifact-namespace", default=os.environ.get("TRAJ_ATTRIBUTION_ARTIFACT_NAMESPACE", "h100_traj_ckptshared_10x10"))
    parser.add_argument("--aggregate-traj-train-from-namespace", default=os.environ.get("AGGREGATE_TRAJ_TRAIN_FROM_NAMESPACE", ""))
    parser.add_argument(
        "--tracin-score-query-normalize",
        action="store_true",
        default=os.environ.get("TRACIN_SCORE_QUERY_NORMALIZE", "0") not in ("0", "false", "False", "no", "No"),
    )
    parser.add_argument(
        "--tracin-score-query-normalize-eps",
        type=float,
        default=float(os.environ.get("TRACIN_SCORE_QUERY_NORMALIZE_EPS", "1e-8")),
    )
    parser.add_argument("--das-damping-sweep-values", default=os.environ.get("DAS_DAMPING_SWEEP_VALUES", DEFAULT_DAS_SWEEP))
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", "python3"))
    args = parser.parse_args()

    args.root = Path(__file__).resolve().parents[1]
    gpus = parse_gpus(args)
    args.gpus_text = ",".join(gpus)
    args.slots = int(args.slots) if args.slots is not None else len(gpus)
    worker_gpu_ids = worker_gpus(args, gpus)
    args.max_parallel = max(1, min(args.max_parallel or len(worker_gpu_ids), len(worker_gpu_ids)))

    env0 = base_env(args)
    env0.setdefault("PYTHONUNBUFFERED", "1")
    env0.setdefault("JAX_BFLOAT16", "1")
    env0.setdefault("JAX_PREFETCH_SIZE", "1")
    env0.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    env0.setdefault("DAS_PROJ_DIM", "4096")
    env0.setdefault("DAS_DAMPING_SWEEP", "1")
    env0["DAS_DAMPING_SWEEP_VALUES"] = args.das_damping_sweep_values
    env0.setdefault("TRAJ_TRACIN_PROJ_DIM", "4096")
    env0.setdefault("PROJECTED_CACHE_DIM", "4096")
    env0.setdefault("PROJECTED_DIMS", "4096")
    env0.setdefault("TRACIN_USE_SHARED_TRAIN_GRADIENT", "1")
    env0.setdefault("TRAJ_NUM_SNAPSHOTS", "10")
    env0.setdefault("TRAJ_TRAIN_MC_SAMPLES", "10")
    env0["TRACIN_SCORE_QUERY_NORMALIZE"] = "1" if args.tracin_score_query_normalize else "0"
    env0["TRACIN_SCORE_QUERY_NORMALIZE_EPS"] = str(args.tracin_score_query_normalize_eps)

    print(
        "traj_score_query_normalize="
        f"{int(args.tracin_score_query_normalize)} eps={args.tracin_score_query_normalize_eps:g}"
    )

    specs = build_query_specs(args)
    manifest_dir = args.root / "result" / args.experiment / "query_sets"
    manifest_path = manifest_dir / f"random_prompted_{args.num_queries}_seed_{args.random_query_seed}_start_{args.initial_seed_start}.json"
    if args.execute:
        manifest_dir.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as handle:
            json.dump({"queries": specs}, handle, indent=2)
    print(f"random prompted query manifest: {manifest_path}")
    for spec in specs:
        print(f"[query {spec['query_id']:02d}] seed={spec['initial_seed']} prompt={spec['query']}")

    ensure_train_artifacts(args)

    if args.only_lds_eval:
        args.skip_sampling = True
        args.skip_query_gradient = True

    if not args.skip_sampling and not args.only_lds_eval:
        sample_jobs: list[Job] = []
        for i, spec in enumerate(specs):
            query = str(spec["query"])
            seed = int(spec["initial_seed"])
            sample_dir = sample_seed_dir(args.root, args, query, seed)
            if sample_complete(sample_dir):
                print(f"[skip] sample exists: {sample_dir}")
                continue
            env = env0 | {
                "QUERY": query,
                "SAMPLE_SEEDS": str(seed),
                "INITIAL_SEED": str(seed),
                "SAMPLE_MODEL_MODE": "prompted_solo",
                "ATTRIBUTION_SAMPLE_MODEL_MODE": "prompted_solo",
            }
            slot = slot_for(i, len(worker_gpu_ids))
            sample_jobs.append(
                Job(
                    name=f"sample_q{spec['query_id']:02d}_{query_tag(query)}_seed_{seed}",
                    cmd=["bash", "scripts/00_sample.sh"],
                    cwd=args.root,
                    env=gpu_env(env, worker_gpu_ids[slot]),
                    log_path=log_root(args) / "random_prompted_20" / "sample" / f"q{spec['query_id']:02d}_{query_tag(query)}_seed_{seed}.log",
                    slot=slot,
                )
            )
        run_parallel_jobs(sample_jobs, args=args, execute=args.execute, max_parallel=args.max_parallel)

    if not args.skip_query_gradient and not args.only_lds_eval:
        q_jobs: list[Job] = []
        for i, spec in enumerate(specs):
            query = str(spec["query"])
            seed = int(spec["initial_seed"])
            for algorithm in (("das", "traj_tracin")):
                if algorithm == "das" and args.skip_das:
                    continue
                if algorithm == "traj_tracin" and args.skip_traj_tracin:
                    continue
                ns = ""
                if args.namespace_query_gradient:
                    if algorithm == "das":
                        ns = artifact_namespace(args)
                    elif algorithm == "traj_tracin":
                        ns = args.traj_artifact_namespace
                q_args = args_for_seed(args, seed, namespace=ns)
                q_args.namespace_query_gradient = args.namespace_query_gradient
                path = query_gradient_artifact_path(args.root, q_args, "prompted_solo", query, algorithm)
                expected_terms = 100 if algorithm == "das" else 5000
                if query_artifact_complete(path, algorithm=algorithm, expected_terms=expected_terms):
                    print(f"[skip] {algorithm} query gradient exists: {path}")
                    continue
                env = env0 | {
                    "QUERY": query,
                    "INITIAL_SEED": str(seed),
                    "SAMPLE_SEEDS": str(seed),
                    "SAMPLE_MODEL_MODE": "prompted_solo",
                    "ATTRIBUTION_SAMPLE_MODEL_MODE": "prompted_solo",
                    "ATTRIBUTION_SCORE_MODEL_MODE": "prompted_solo",
                    "QUERY_GRADIENT_ARTIFACT_PATH": str(path),
                }
                slot = slot_for(len(q_jobs), len(worker_gpu_ids))
                q_jobs.append(
                    Job(
                        name=f"{algorithm}_query_q{spec['query_id']:02d}_{query_tag(query)}_seed_{seed}",
                        cmd=[args.python_bin, "02_query_gradient.py"],
                        cwd=args.root / "data_attribution" / algorithm,
                        env=gpu_env(env, worker_gpu_ids[slot]),
                        log_path=log_root(args) / "random_prompted_20" / "query_gradient" / algorithm / f"q{spec['query_id']:02d}_{query_tag(query)}_seed_{seed}.log",
                        slot=slot,
                    )
                )
        run_parallel_jobs(q_jobs, args=args, execute=args.execute, max_parallel=args.max_parallel)

    for spec in specs:
        query = str(spec["query"])
        seed = int(spec["initial_seed"])
        if not args.skip_das:
            das_namespace = artifact_namespace(args) if args.namespace_query_gradient else ""
            das_extra = ["--skip-traj-tracin", "--skip-query-gradient"] + (
                ["--skip-lds-eval"] if args.skip_lds_eval else []
            )
            if args.namespace_query_gradient:
                das_extra.append("--namespace-query-gradient")
            run_distributed_for_query(
                args,
                query=query,
                seed=seed,
                namespace=das_namespace,
                extra=das_extra,
                env=env0,
            )
        if not args.skip_traj_tracin:
            run_distributed_for_query(
                args,
                query=query,
                seed=seed,
                namespace=args.traj_artifact_namespace,
                extra=["--skip-das", "--skip-query-gradient"] + (["--skip-lds-eval"] if args.skip_lds_eval else []),
                env=env0,
            )

    if args.execute:
        print("[done] random prompted query attribution complete")
    else:
        print("[dry-run] add --execute to run commands")


if __name__ == "__main__":
    main()
