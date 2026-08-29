#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from run_cifar5_multi_experiment import (
    Job,
    TARGET_FUNCTIONS,
    attribution_score_dirs,
    base_env,
    choose_prompted_queries,
    gpu_env,
    lds_eval_out_dir,
    lds_model_dirs,
    log_root,
    parse_csv,
    parse_gpus,
    query_env,
    query_tag,
    run,
    run_parallel_jobs,
    slot_for,
    worker_gpus,
)


TRAIN_ARTIFACT = "train_datapoint_gradient_artifact.npz"


def train_artifact_path(root: Path, args: argparse.Namespace, mode: str, algorithm: str) -> Path:
    return (
        root
        / "result"
        / args.experiment
        / "model"
        / mode
        / f"seed_{args.train_seed}_train_gradient"
        / algorithm
        / TRAIN_ARTIFACT
    )


def train_artifact_complete(path: Path, *, expected_points: int) -> bool:
    if not path.is_file():
        return False
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as data:
            if "train_features" not in data or "score_indices" not in data:
                return False
            return int(np.asarray(data["score_indices"]).reshape(-1).shape[0]) == int(expected_points)
    except Exception:
        return False


def score_complete(root: Path, args: argparse.Namespace, *, mode: str, query: str, algorithm: str) -> bool:
    score_dirs = attribution_score_dirs(root, args, mode=mode, query=query, algorithm=algorithm)
    return all((score_dir / "scores.npy").is_file() for _tag, score_dir in score_dirs)


def eval_complete(out_dir: Path) -> bool:
    return (out_dir / "lds_summary.json").is_file()


def split_1based_ranges(size: int, shards: int) -> list[tuple[int, int]]:
    shards = max(1, min(int(shards), int(size)))
    base = size // shards
    rem = size % shards
    out: list[tuple[int, int]] = []
    start = 1
    for i in range(shards):
        count = base + (1 if i < rem else 0)
        end = start + count - 1
        out.append((start, end))
        start = end + 1
    return out


def shard_artifact_path(root: Path, args: argparse.Namespace, mode: str, algorithm: str, start: int, end: int) -> Path:
    final_path = train_artifact_path(root, args, mode, algorithm)
    return final_path.parent / "datapoint_shards" / f"range_{start}_{end}" / TRAIN_ARTIFACT


def das_global_gram_path(root: Path, args: argparse.Namespace, mode: str) -> Path:
    final_path = train_artifact_path(root, args, mode, "das")
    return final_path.parent / "global_gram_artifact.npz"


def score_shard_dir(score_dir: Path, start: int, end: int) -> Path:
    return score_dir / "datapoint_shards" / f"range_{start}_{end}"


def shell_join(commands: list[list[str]]) -> list[str]:
    return ["bash", "-lc", " && ".join(" ".join(cmd) for cmd in commands)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume CIFAR5_MULTI DAS + sharded shared-train TrajTracIn attribution.")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--experiment", default="cifar5_multi_exp1")
    parser.add_argument("--size", type=int, default=10000)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--sample-seeds", default="0")
    parser.add_argument("--query-seed", type=int, default=0)
    parser.add_argument("--lds-m", type=int, default=64)
    parser.add_argument("--lds-percentage", type=float, default=25)
    parser.add_argument("--lds-subset-seeds", default="0,1,2")
    parser.add_argument("--lds-epochs", type=int, default=200)
    parser.add_argument("--skip-das", action="store_true")
    parser.add_argument("--skip-traj-tracin", action="store_true")
    parser.add_argument("--skip-lds-eval", action="store_true")
    parser.add_argument("--only-lds-eval", action="store_true")
    parser.add_argument(
        "--eval-algorithms",
        default=None,
        help="Comma-separated algorithms for LDS eval. Defaults to the algorithms not skipped.",
    )
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--slots", type=int, default=None)
    parser.add_argument("--gpu-per-node", type=int, default=4)
    parser.add_argument("--cpus-per-worker", type=int, default=8)
    parser.add_argument("--slot-backend", choices=("local", "ibrun", "srun"), default=os.environ.get("TACC_SLOT_BACKEND", "local"))
    parser.add_argument("--use-task-affinity", action="store_true")
    args = parser.parse_args()

    args.root = Path(__file__).resolve().parents[1]
    env0 = base_env(args)
    env0.setdefault("TRACIN_USE_SHARED_TRAIN_GRADIENT", "1")
    env0.setdefault("TRAJ_TRACIN_PROJ_DIM", "4096")
    env0.setdefault("PROJECTED_CACHE_DIM", "4096")
    env0.setdefault("PROJECTED_DIMS", "4096")
    env0.setdefault("DAS_PROJ_DIM", "4096")
    env0.setdefault("DAS_DAMPING_SWEEP", "1")
    env0.setdefault("JAX_BFLOAT16", "1")
    env0.setdefault("JAX_PREFETCH_SIZE", "1")

    queries = choose_prompted_queries(args.query_seed, 2)
    all_queries = queries + ["unprompted"]
    gpus = parse_gpus(args)
    worker_gpu_ids = worker_gpus(args, gpus)
    python_bin = os.environ.get("PYTHON_BIN", "python3")
    print(f"prompted queries: {queries}")
    print(f"worker slots={len(worker_gpu_ids)} | worker_gpus={worker_gpu_ids} | backend={args.slot_backend}")

    if args.only_lds_eval:
        args.skip_das = True
        args.skip_traj_tracin = True
        args.skip_lds_eval = False

    ranges = split_1based_ranges(args.size, len(worker_gpu_ids))

    if not args.skip_das:
        das_query_jobs: list[Job] = []
        for i, query in enumerate(all_queries):
            mode, env = query_env(args, env0, query)
            if score_complete(args.root, args, mode=mode, query=query, algorithm="das"):
                print(f"[skip] DAS scores already complete for {query_tag(query)}")
                continue
            slot = slot_for(i, len(worker_gpu_ids))
            das_query_jobs.append(
                Job(
                    name=f"das_query_{query_tag(query)}",
                    cmd=[python_bin, "02_query_gradient.py"],
                    cwd=args.root / "data_attribution" / "das",
                    env=gpu_env(env, worker_gpu_ids[slot]),
                    log_path=log_root(args) / "attribution_resume" / "das" / "query" / f"{query_tag(query)}.log",
                    slot=slot,
                )
            )
        run_parallel_jobs(das_query_jobs, args=args, execute=args.execute, max_parallel=len(worker_gpu_ids))

        for mode in ("prompted_solo", "unprompted_solo"):
            final_artifact = train_artifact_path(args.root, args, mode, "das")
            global_gram = das_global_gram_path(args.root, args, mode)
            if final_artifact.is_file() or global_gram.is_file():
                print(f"[skip] complete/shared DAS train state for {mode}: {global_gram if global_gram.is_file() else final_artifact}")
                continue

            shard_jobs: list[Job] = []
            for shard_id, (start, end) in enumerate(ranges):
                shard_path = shard_artifact_path(args.root, args, mode, "das", start, end)
                if train_artifact_complete(shard_path, expected_points=end - start + 1):
                    print(f"[skip] DAS train shard {mode} {start}-{end}: {shard_path}")
                    continue
                env = env0 | {
                    "DATAPOINT_MODEL_MODE": mode,
                    "SAMPLE_MODEL_MODE": mode,
                    "ATTRIBUTION_SCORE_MODEL_MODE": mode,
                    "SCORE_INDEX_RANGES": f"{start}-{end}",
                    "TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH": str(shard_path),
                }
                if mode.startswith("unprompted"):
                    env["UNPROMPTED"] = "1"
                slot = slot_for(shard_id, len(worker_gpu_ids))
                shard_jobs.append(
                    Job(
                        name=f"das_train_{mode}_range_{start}_{end}",
                        cmd=[python_bin, "01_train_datapoint_gradient.py"],
                        cwd=args.root / "data_attribution" / "das",
                        env=gpu_env(env, worker_gpu_ids[slot]),
                        log_path=log_root(args)
                        / "attribution_resume"
                        / "das"
                        / "train_shards"
                        / mode
                        / f"range_{start}_{end}_slot_{slot}_gpu_{worker_gpu_ids[slot]}.log",
                        slot=slot,
                    )
                )
            run_parallel_jobs(shard_jobs, args=args, execute=args.execute, max_parallel=len(worker_gpu_ids))

            shard_paths = [shard_artifact_path(args.root, args, mode, "das", start, end) for start, end in ranges]
            if args.execute:
                missing = [str(path) for path in shard_paths if not path.is_file()]
                if missing:
                    raise FileNotFoundError(f"Missing DAS train shard(s) for {mode}: {missing[:3]}")
            run(
                [python_bin, str(args.root.parent / "common" / "merge_das_train_shards.py"), "--output", str(global_gram), *map(str, shard_paths)],
                env0,
                cwd=args.root,
                execute=args.execute,
            )

        for query in all_queries:
            mode, env = query_env(args, env0, query)
            if score_complete(args.root, args, mode=mode, query=query, algorithm="das"):
                print(f"[skip] DAS scores already complete for {query_tag(query)}")
                continue
            final_artifact = train_artifact_path(args.root, args, mode, "das")
            if train_artifact_complete(final_artifact, expected_points=args.size):
                slot = slot_for(0, len(worker_gpu_ids))
                run_parallel_jobs(
                    [
                        Job(
                            name=f"das_score_{query_tag(query)}",
                            cmd=[python_bin, "03_score.py"],
                            cwd=args.root / "data_attribution" / "das",
                            env=gpu_env(env, worker_gpu_ids[slot]),
                            log_path=log_root(args) / "attribution_resume" / "das" / "score" / f"{query_tag(query)}.log",
                            slot=slot,
                        )
                    ],
                    args=args,
                    execute=args.execute,
                    max_parallel=len(worker_gpu_ids),
                )
                continue
            global_gram = das_global_gram_path(args.root, args, mode)
            score_dirs = attribution_score_dirs(args.root, args, mode=mode, query=query, algorithm="das")
            das_base_score_dir = score_dirs[0][1].parent
            shard_score_jobs: list[Job] = []
            for shard_id, (start, end) in enumerate(ranges):
                shard_path = shard_artifact_path(args.root, args, mode, "das", start, end)
                target_score_base = score_shard_dir(das_base_score_dir, start, end)
                if all((target_score_base / score_tag / "scores.npy").is_file() for score_tag, _score_dir in score_dirs):
                    print(f"[skip] DAS score shard {query_tag(query)} {start}-{end}: {target_score_base}")
                    continue
                shard_env = env | {
                    "TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH": str(shard_path),
                    "DAS_GLOBAL_GRAM_ARTIFACT_PATH": str(global_gram),
                    "SCORE_OUTPUT_DIR": str(target_score_base),
                }
                slot = slot_for(shard_id, len(worker_gpu_ids))
                shard_score_jobs.append(
                    Job(
                        name=f"das_score_{query_tag(query)}_range_{start}_{end}",
                        cmd=[python_bin, "03_score.py"],
                        cwd=args.root / "data_attribution" / "das",
                        env=gpu_env(shard_env, worker_gpu_ids[slot]),
                        log_path=log_root(args)
                        / "attribution_resume"
                        / "das"
                        / "score_shards"
                        / query_tag(query)
                        / f"range_{start}_{end}_slot_{slot}_gpu_{worker_gpu_ids[slot]}.log",
                        slot=slot,
                    )
                )
            run_parallel_jobs(shard_score_jobs, args=args, execute=args.execute, max_parallel=len(worker_gpu_ids))

            for score_tag, final_score_dir in score_dirs:
                shard_dirs = [score_shard_dir(das_base_score_dir, start, end) / score_tag for start, end in ranges]
                if args.execute:
                    missing_scores = [str(path / "scores.npy") for path in shard_dirs if not (path / "scores.npy").is_file()]
                    if missing_scores:
                        raise FileNotFoundError(f"Missing DAS score shard(s) for {query_tag(query)} {score_tag}: {missing_scores[:3]}")
                run(
                    [python_bin, str(args.root.parent / "common" / "merge_score_shards.py"), "--output-dir", str(final_score_dir), *map(str, shard_dirs)],
                    env0,
                    cwd=args.root,
                    execute=args.execute,
                )

    if not args.skip_traj_tracin:
        query_jobs: list[Job] = []
        for i, query in enumerate(all_queries):
            _mode, env = query_env(args, env0, query)
            slot = slot_for(i, len(worker_gpu_ids))
            query_jobs.append(
                Job(
                    name=f"traj_query_{query_tag(query)}",
                    cmd=[python_bin, "02_query_gradient.py"],
                    cwd=args.root / "data_attribution" / "traj_tracin",
                    env=gpu_env(env, worker_gpu_ids[slot]),
                    log_path=log_root(args) / "attribution_resume" / "traj_tracin" / "query" / f"{query_tag(query)}.log",
                    slot=slot,
                )
            )
        run_parallel_jobs(query_jobs, args=args, execute=args.execute, max_parallel=len(worker_gpu_ids))

        for mode in ("prompted_solo", "unprompted_solo"):
            final_artifact = train_artifact_path(args.root, args, mode, "traj_tracin")
            if train_artifact_complete(final_artifact, expected_points=args.size):
                print(f"[skip] complete shared TrajTracIn train artifact for {mode}: {final_artifact}")
                continue

            shard_jobs: list[Job] = []
            for shard_id, (start, end) in enumerate(ranges):
                shard_path = shard_artifact_path(args.root, args, mode, "traj_tracin", start, end)
                if train_artifact_complete(shard_path, expected_points=end - start + 1):
                    print(f"[skip] TrajTracIn train shard {mode} {start}-{end}: {shard_path}")
                    continue
                env = env0 | {
                    "DATAPOINT_MODEL_MODE": mode,
                    "SAMPLE_MODEL_MODE": mode,
                    "ATTRIBUTION_SCORE_MODEL_MODE": mode,
                    "SCORE_INDEX_RANGES": f"{start}-{end}",
                    "TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH": str(shard_path),
                    "TRACIN_USE_SHARED_TRAIN_GRADIENT": "1",
                }
                if mode.startswith("unprompted"):
                    env["UNPROMPTED"] = "1"
                slot = slot_for(shard_id, len(worker_gpu_ids))
                shard_jobs.append(
                    Job(
                        name=f"traj_train_{mode}_range_{start}_{end}",
                        cmd=[python_bin, "01_train_datapoint_gradient.py"],
                        cwd=args.root / "data_attribution" / "traj_tracin",
                        env=gpu_env(env, worker_gpu_ids[slot]),
                        log_path=log_root(args)
                        / "attribution_resume"
                        / "traj_tracin"
                        / "train_shards"
                        / mode
                        / f"range_{start}_{end}_slot_{slot}_gpu_{worker_gpu_ids[slot]}.log",
                        slot=slot,
                    )
                )
            run_parallel_jobs(shard_jobs, args=args, execute=args.execute, max_parallel=len(worker_gpu_ids))

            shard_paths = [shard_artifact_path(args.root, args, mode, "traj_tracin", start, end) for start, end in ranges]
            if args.execute:
                missing = [str(path) for path in shard_paths if not path.is_file()]
                if missing:
                    raise FileNotFoundError(f"Missing TrajTracIn train shard(s) for {mode}: {missing[:3]}")

        for query in all_queries:
            mode, env = query_env(args, env0, query)
            if score_complete(args.root, args, mode=mode, query=query, algorithm="traj_tracin"):
                print(f"[skip] TrajTracIn scores already complete for {query_tag(query)}")
                continue

            final_artifact = train_artifact_path(args.root, args, mode, "traj_tracin")
            score_dirs = attribution_score_dirs(args.root, args, mode=mode, query=query, algorithm="traj_tracin")
            if train_artifact_complete(final_artifact, expected_points=args.size):
                score_jobs = []
                slot = slot_for(len(score_jobs), len(worker_gpu_ids))
                score_jobs.append(
                    Job(
                        name=f"traj_score_{query_tag(query)}",
                        cmd=[python_bin, "03_score.py"],
                        cwd=args.root / "data_attribution" / "traj_tracin",
                        env=gpu_env(env, worker_gpu_ids[slot]),
                        log_path=log_root(args) / "attribution_resume" / "traj_tracin" / "score" / f"{query_tag(query)}.log",
                        slot=slot,
                    )
                )
                run_parallel_jobs(score_jobs, args=args, execute=args.execute, max_parallel=len(worker_gpu_ids))
                continue

            shard_score_jobs: list[Job] = []
            for shard_id, (start, end) in enumerate(ranges):
                shard_path = shard_artifact_path(args.root, args, mode, "traj_tracin", start, end)
                target_score_dir = score_shard_dir(score_dirs[0][1], start, end)
                if (target_score_dir / "scores.npy").is_file():
                    print(f"[skip] TrajTracIn score shard {query_tag(query)} {start}-{end}: {target_score_dir}")
                    continue
                shard_env = env | {
                    "TRAIN_DATAPOINT_GRADIENT_ARTIFACT_PATH": str(shard_path),
                    "SCORE_OUTPUT_DIR": str(target_score_dir),
                }
                slot = slot_for(shard_id, len(worker_gpu_ids))
                shard_score_jobs.append(
                    Job(
                        name=f"traj_score_{query_tag(query)}_range_{start}_{end}",
                        cmd=[python_bin, "03_score.py"],
                        cwd=args.root / "data_attribution" / "traj_tracin",
                        env=gpu_env(shard_env, worker_gpu_ids[slot]),
                        log_path=log_root(args)
                        / "attribution_resume"
                        / "traj_tracin"
                        / "score_shards"
                        / query_tag(query)
                        / f"range_{start}_{end}_slot_{slot}_gpu_{worker_gpu_ids[slot]}.log",
                        slot=slot,
                    )
                )
            run_parallel_jobs(shard_score_jobs, args=args, execute=args.execute, max_parallel=len(worker_gpu_ids))

            shard_dirs = [score_shard_dir(score_dirs[0][1], start, end) for start, end in ranges]
            if args.execute:
                missing_scores = [str(path / "scores.npy") for path in shard_dirs if not (path / "scores.npy").is_file()]
                if missing_scores:
                    raise FileNotFoundError(f"Missing TrajTracIn score shard(s) for {query_tag(query)}: {missing_scores[:3]}")
            run(
                [python_bin, str(args.root.parent / "common" / "merge_score_shards.py"), "--output-dir", str(score_dirs[0][1]), *map(str, shard_dirs)],
                env0,
                cwd=args.root,
                execute=args.execute,
            )

    if not args.skip_lds_eval:
        if args.eval_algorithms:
            algorithms = parse_csv(args.eval_algorithms)
        else:
            algorithms = []
            if not args.skip_das:
                algorithms.append("das")
            if not args.skip_traj_tracin:
                algorithms.append("traj_tracin")
        for query in all_queries:
            for algorithm in algorithms:
                mode, base_query_env = query_env(args, env0, query)
                lds_dirs = lds_model_dirs(args.root, args, mode)
                for target in TARGET_FUNCTIONS:
                    for score_tag, score_dir in attribution_score_dirs(args.root, args, mode=mode, query=query, algorithm=algorithm):
                        out_dir = lds_eval_out_dir(
                            args.root,
                            args,
                            mode=mode,
                            query=query,
                            algorithm=algorithm,
                            score_tag=score_tag,
                            target=target,
                        )
                        if eval_complete(out_dir):
                            print(f"[skip] LDS eval already complete: {out_dir}")
                            continue
                        if not (score_dir / "scores.npy").is_file():
                            print(f"[skip] missing score for LDS eval: {score_dir}")
                            continue
                        cmd = [
                            python_bin,
                            "lds/run_eval.py",
                            "--algorithm",
                            algorithm,
                            "--lds-model-dirs",
                            lds_dirs,
                            "--score-file",
                            str(score_dir),
                            "--target-function",
                            target,
                            "--out-dir",
                            str(out_dir),
                        ]
                        if query == "unprompted":
                            cmd.insert(2, "--unprompted")
                        run(cmd, base_query_env, cwd=args.root, execute=args.execute)


if __name__ == "__main__":
    main()
