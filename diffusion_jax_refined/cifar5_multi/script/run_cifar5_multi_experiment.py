#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


LABELS = ("bird", "horse", "automobile", "dog", "cat")
TARGET_FUNCTIONS = (
    "noise_trajectory",
    "endpoint_counterfactual",
    "traj_counterfactual",
    "simple_loss",
)


def query_tag(query: str) -> str:
    return "unprompted" if query == "unprompted" else query.replace(",", "_")


def damping_tag(value: float) -> str:
    return f"{float(value):g}".replace("+", "").replace("-", "neg_").replace(".", "p")


def das_damping_values() -> tuple[float, ...]:
    text = os.environ.get("DAS_DAMPING_SWEEP_VALUES")
    if text:
        return tuple(float(part) for part in text.replace(",", " ").split() if part.strip())
    return (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)


def lds_model_dirs(root: Path, args: argparse.Namespace, mode: str) -> str:
    fraction = args.lds_percentage if args.lds_percentage <= 1 else args.lds_percentage / 100.0
    k = round(args.size * fraction)
    pct_tag = f"pct_{args.lds_percentage:g}".replace(".", "p")
    seeds = [x.strip() for x in args.lds_subset_seeds.split(",") if x.strip()]
    dirs = [
        root
        / "result"
        / args.experiment
        / "lds_model"
        / mode
        / f"train_seed_{args.train_seed}"
        / f"m_{args.lds_m}_k_{k}_{pct_tag}_subset_seed_{seed}"
        for seed in seeds
    ]
    return ",".join(str(path) for path in dirs)


def attribution_score_dirs(root: Path, args: argparse.Namespace, *, mode: str, query: str, algorithm: str) -> list[tuple[str, Path]]:
    query_component = "unprompted" if query == "unprompted" else f"query_{query_tag(query)}"
    base = (
        root
        / "result"
        / args.experiment
        / "attribution_score"
        / mode
        / f"train_seed_{args.train_seed}"
        / query_component
        / f"initial_seed_{args.sample_seeds.split(',')[0]}"
        / algorithm
        / "score"
    )
    if algorithm == "das":
        return [(f"lambda_{damping_tag(v)}", base / f"lambda_{damping_tag(v)}") for v in das_damping_values()]
    return [("default", base)]


def lds_eval_out_dir(
    root: Path,
    args: argparse.Namespace,
    *,
    mode: str,
    query: str,
    algorithm: str,
    score_tag: str,
    target: str,
) -> Path:
    query_component = "unprompted" if query == "unprompted" else f"query_{query_tag(query)}"
    lds_component = "lds_unprompted" if query == "unprompted" else "lds"
    alg_component = algorithm if score_tag == "default" else f"{algorithm}_{score_tag}"
    return (
        root
        / "result"
        / args.experiment
        / "eval"
        / mode
        / query_component
        / f"initial_seed_{args.sample_seeds.split(',')[0]}"
        / lds_component
        / alg_component
        / target
    )


def choose_prompted_queries(seed: int, count: int = 2) -> list[str]:
    import random

    combos = [",".join(c) for c in itertools.combinations(LABELS, 3)]
    rng = random.Random(seed)
    rng.shuffle(combos)
    return combos[:count]


def run(cmd: list[str], env: dict[str, str], *, cwd: Path, execute: bool) -> None:
    printable = " ".join(cmd)
    prefix = "RUN" if execute else "DRY"
    print(f"[{prefix}] {printable}")
    if execute:
        subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


@dataclass
class Job:
    name: str
    cmd: list[str]
    cwd: Path
    env: dict[str, str]
    log_path: Path
    slot: int = 0


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in text.replace(" ", ",").split(",") if part.strip()]


def parse_gpus(args: argparse.Namespace) -> list[str]:
    if args.gpus:
        return parse_csv(args.gpus)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return parse_csv(visible)
    return ["0"]


def worker_gpus(args: argparse.Namespace, gpus: list[str]) -> list[str]:
    slots = int(args.slots) if args.slots is not None else len(gpus)
    gpu_per_node = int(args.gpu_per_node)
    return [gpus[i % min(len(gpus), gpu_per_node)] for i in range(slots)]


def log_root(args: argparse.Namespace) -> Path:
    return args.root / "result" / args.experiment / "logs"


def gpu_env(env: dict[str, str], gpu: str, *, single_device: bool = True) -> dict[str, str]:
    child = env.copy()
    child["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if single_device:
        child["JAX_DATA_PARALLEL"] = "0"
        child["JAX_NUM_DEVICES"] = "1"
        child["LDS_NUM_DEVICES"] = "1"
    return child


def launch_cmd(args: argparse.Namespace, job: Job) -> list[str]:
    if args.slot_backend == "ibrun":
        cmd = ["ibrun", "-n", "1", "-o", str(job.slot)]
        if args.use_task_affinity:
            cmd.append("task_affinity")
        return cmd + job.cmd
    if args.slot_backend == "srun":
        return [
            "srun",
            "--nodes=1",
            "--ntasks=1",
            "--exclusive",
            "--gres=gpu:1",
            "--cpus-per-task",
            str(args.cpus_per_worker),
        ] + job.cmd
    return job.cmd


def run_parallel_jobs(jobs: list[Job], *, args: argparse.Namespace, execute: bool, max_parallel: int) -> None:
    if not jobs:
        return
    if max_parallel <= 0:
        raise ValueError("max_parallel must be positive")

    prefix = "RUN" if execute else "DRY"
    for job in jobs:
        printable = " ".join(launch_cmd(args, job))
        gpu = job.env.get("CUDA_VISIBLE_DEVICES", "?")
        print(f"[{prefix}][slot={job.slot}][gpu={gpu}][log={job.log_path}] {job.name}: {printable}")
    if not execute:
        return

    active: list[tuple[Job, subprocess.Popen, object]] = []
    pending = list(jobs)
    failures: list[tuple[Job, int]] = []
    while pending or active:
        while pending and len(active) < max_parallel:
            job = pending.pop(0)
            job.log_path.parent.mkdir(parents=True, exist_ok=True)
            log_f = job.log_path.open("ab")
            header = (
                f"\n\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} | {job.name} | "
                f"slot={job.slot} | gpu={job.env.get('CUDA_VISIBLE_DEVICES', '?')} =====\n"
            )
            log_f.write(header.encode("utf-8"))
            log_f.flush()
            proc = subprocess.Popen(
                launch_cmd(args, job),
                cwd=str(job.cwd),
                env=job.env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
            active.append((job, proc, log_f))

        time.sleep(5)
        still_active: list[tuple[Job, subprocess.Popen, object]] = []
        for job, proc, log_f in active:
            rc = proc.poll()
            if rc is None:
                still_active.append((job, proc, log_f))
                continue
            log_f.close()
            if rc != 0:
                failures.append((job, rc))
                print(f"[FAIL][{job.name}] exit={rc} log={job.log_path}")
            else:
                print(f"[DONE][{job.name}] log={job.log_path}")
        active = still_active
        if failures:
            for _, proc, log_f in active:
                proc.terminate()
                log_f.close()
            first, rc = failures[0]
            raise subprocess.CalledProcessError(rc, first.cmd)


def subset_index_chunks(m: int, num_chunks: int) -> list[list[int]]:
    chunks = [[] for _ in range(num_chunks)]
    for subset_id in range(m):
        chunks[subset_id % num_chunks].append(subset_id)
    return chunks


def slot_for(index: int, worker_count: int) -> int:
    return index % worker_count


def subset_indices_text(indices: list[int]) -> str:
    return ",".join(str(i) for i in indices)


def query_env(args: argparse.Namespace, env0: dict[str, str], query: str) -> tuple[str, dict[str, str]]:
    if query == "unprompted":
        mode = "unprompted_solo"
        return mode, env0 | {
            "INITIAL_SEED": args.sample_seeds.split(",")[0],
            "SAMPLE_MODEL_MODE": mode,
            "ATTRIBUTION_SCORE_MODEL_MODE": mode,
            "DATAPOINT_MODEL_MODE": mode,
            "UNPROMPTED": "1",
            "TRACIN_USE_SHARED_TRAIN_GRADIENT": "1",
        }
    mode = "prompted_solo"
    return mode, env0 | {
        "QUERY": query,
        "INITIAL_SEED": args.sample_seeds.split(",")[0],
        "SAMPLE_MODEL_MODE": mode,
        "ATTRIBUTION_SCORE_MODEL_MODE": mode,
        "DATAPOINT_MODEL_MODE": mode,
        "TRACIN_USE_SHARED_TRAIN_GRADIENT": "1",
    }


def attribution_job_cmd(python_bin: str, algorithm: str) -> list[str]:
    return [
        "bash",
        "-lc",
        f"{python_bin} 02_query_gradient.py && {python_bin} 01_train_datapoint_gradient.py && {python_bin} 03_score.py",
    ]


def base_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("EXPERIMENT_TAG", args.experiment)
    env.setdefault("TRAIN_SEED", str(args.train_seed))
    env.setdefault("JAX_EPOCHS", str(args.epochs))
    env.setdefault("JAX_BFLOAT16", "1")
    env.setdefault("CIFAR5_MULTI_SIZE", str(args.size))
    env.setdefault("CIFAR5_MULTI_DATA_ROOT", str(args.root.parent / "dataset" / "cifar5_multi" / str(args.size)))
    env.setdefault("DAS_PROJ_DIM", "4096")
    env.setdefault("DAS_DAMPING_SWEEP", "1")
    env.setdefault("DTRAK_PROJ_DIM", "4096")
    env.setdefault("TRAJ_TRACIN_PROJ_DIM", "4096")
    env.setdefault("PROJECTED_CACHE_DIM", "4096")
    env.setdefault("PROJECTED_DIMS", "4096")
    env.setdefault("TRACIN_USE_SHARED_TRAIN_GRADIENT", "1")
    env.setdefault("LDS_M", str(args.lds_m))
    env.setdefault("LDS_DATASET_PERCENTAGE", str(args.lds_percentage))
    env.setdefault("LDS_EPOCHS", str(args.lds_epochs))
    env.setdefault("LDS_SAVE_EVERY_EPOCHS", str(args.lds_epochs))
    env.setdefault("LDS_KEEP_LAST_K", "1")
    env.setdefault("LDS_NUM_DEVICES", "1")
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cifar5_multi end-to-end experiment.")
    parser.add_argument("--execute", action="store_true", help="Actually run commands. Default prints the plan.")
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--size", type=int, default=10000)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--sample-seeds", default="0")
    parser.add_argument("--query-seed", type=int, default=0)
    parser.add_argument("--lds-m", type=int, default=64)
    parser.add_argument("--lds-percentage", type=float, default=25)
    parser.add_argument("--lds-subset-seeds", default="0,1,2")
    parser.add_argument("--lds-epochs", type=int, default=200)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-attribution", action="store_true")
    parser.add_argument("--skip-lds", action="store_true")
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated per-node GPU ids for independent jobs. Defaults to CUDA_VISIBLE_DEVICES, then 0.",
    )
    parser.add_argument(
        "--slots",
        type=int,
        default=None,
        help="Number of independent worker slots. For TACC H100 use 16 for 4 nodes x 4 GPUs.",
    )
    parser.add_argument("--gpu-per-node", type=int, default=4)
    parser.add_argument("--cpus-per-worker", type=int, default=8)
    parser.add_argument("--slot-backend", choices=("local", "ibrun", "srun"), default=os.environ.get("TACC_SLOT_BACKEND", "local"))
    parser.add_argument("--use-task-affinity", action="store_true")
    parser.add_argument("--no-parallel", action="store_true", help="Use the old sequential schedule.")
    parser.add_argument(
        "--attribution-algorithms",
        default="das,traj_tracin",
        help="Comma-separated attribution algorithms. Use das,traj_tracin,end_tracin to include End TracIn.",
    )
    args = parser.parse_args()

    args.root = Path(__file__).resolve().parents[1]
    env0 = base_env(args)
    queries = choose_prompted_queries(args.query_seed, 2)
    all_queries = queries + ["unprompted"]
    gpus = parse_gpus(args)
    worker_gpu_ids = worker_gpus(args, gpus)
    use_parallel = (not args.no_parallel) and len(worker_gpu_ids) > 1
    algorithms = tuple(parse_csv(args.attribution_algorithms))
    python_bin = os.environ.get("PYTHON_BIN", "python3")
    print(f"prompted queries: {queries}")
    print("unprompted query: unprompted")
    print(f"job GPUs: {gpus} | worker_gpus={worker_gpu_ids} | backend={args.slot_backend} | parallel={use_parallel}")
    print(f"attribution algorithms: {algorithms}")

    if not args.skip_generate:
        run(
            [
                os.environ.get("PYTHON_BIN", "python3"),
                str(args.root / "script" / "generate_cifar5_multi.py"),
                "--size",
                str(args.size),
                "--seed",
                str(args.data_seed),
            ],
            env0,
            cwd=args.root,
            execute=args.execute,
        )

    if not args.skip_train:
        if use_parallel:
            jobs = [
                Job(
                    name="train_prompted_base",
                    cmd=["bash", "scripts/00_train_prompted_solo.sh"],
                    cwd=args.root,
                    env=gpu_env(env0, worker_gpu_ids[0]),
                    log_path=log_root(args) / "base" / "prompted.log",
                    slot=0,
                ),
                Job(
                    name="train_unprompted_base",
                    cmd=["bash", "scripts/00_train_unprompted_solo.sh"],
                    cwd=args.root,
                    env=gpu_env(env0, worker_gpu_ids[1 % len(worker_gpu_ids)]),
                    log_path=log_root(args) / "base" / "unprompted.log",
                    slot=1 % len(worker_gpu_ids),
                ),
            ]
            run_parallel_jobs(jobs, args=args, execute=args.execute, max_parallel=min(len(worker_gpu_ids), len(jobs)))
        else:
            run(["bash", "scripts/00_train_prompted_solo.sh"], env0, cwd=args.root, execute=args.execute)
            run(["bash", "scripts/00_train_unprompted_solo.sh"], env0, cwd=args.root, execute=args.execute)

    sample_jobs = []
    for i, query in enumerate(queries):
        env = env0 | {"QUERY": query, "SAMPLE_SEEDS": args.sample_seeds, "SAMPLE_MODEL_MODE": "prompted_solo"}
        if use_parallel:
            sample_jobs.append(
                Job(
                    name=f"sample_{query_tag(query)}",
                    cmd=["bash", "scripts/00_sample.sh"],
                    cwd=args.root,
                    env=gpu_env(env, worker_gpu_ids[i % len(worker_gpu_ids)]),
                    log_path=log_root(args) / "sample" / f"{query_tag(query)}.log",
                    slot=slot_for(i, len(worker_gpu_ids)),
                )
            )
        else:
            run(["bash", "scripts/00_sample.sh"], env, cwd=args.root, execute=args.execute)
    env = env0 | {"SAMPLE_SEEDS": args.sample_seeds, "SAMPLE_MODEL_MODE": "unprompted_solo", "UNPROMPTED": "1"}
    if use_parallel:
        sample_jobs.append(
            Job(
                name="sample_unprompted",
                cmd=["bash", "scripts/00_sample_unprompted.sh"],
                cwd=args.root,
                env=gpu_env(env, worker_gpu_ids[len(sample_jobs) % len(worker_gpu_ids)]),
                log_path=log_root(args) / "sample" / "unprompted.log",
                slot=slot_for(len(sample_jobs), len(worker_gpu_ids)),
            )
        )
        run_parallel_jobs(sample_jobs, args=args, execute=args.execute, max_parallel=min(len(worker_gpu_ids), len(sample_jobs)))
    else:
        run(["bash", "scripts/00_sample_unprompted.sh"], env, cwd=args.root, execute=args.execute)

    if not args.skip_lds:
        if use_parallel:
            jobs = []
            chunks = subset_index_chunks(args.lds_m, len(worker_gpu_ids))
            for subset_seed in [x.strip() for x in args.lds_subset_seeds.split(",") if x.strip()]:
                for mode, script_name in (
                    ("prompted_solo", "scripts/03_lds_training.sh"),
                    ("unprompted_solo", "scripts/03_lds_training_unprompted.sh"),
                ):
                    for slot, (gpu, chunk) in enumerate(zip(worker_gpu_ids, chunks)):
                        if not chunk:
                            continue
                        env = env0 | {
                            "LDS_SAMPLE_RANDOM_SEED": subset_seed,
                            "SAMPLE_MODEL_MODE": mode,
                            "LDS_SUBSET_INDICES": subset_indices_text(chunk),
                        }
                        jobs.append(
                            Job(
                                name=f"lds_train_{mode}_subset_seed_{subset_seed}_slot_{slot}",
                                cmd=["bash", script_name],
                                cwd=args.root,
                                env=gpu_env(env, gpu),
                                log_path=log_root(args)
                                / "lds"
                                / mode
                                / f"subset_seed_{subset_seed}"
                                / f"slot_{slot}_gpu_{gpu}.log",
                                slot=slot,
                            )
                        )
            run_parallel_jobs(jobs, args=args, execute=args.execute, max_parallel=len(worker_gpu_ids))
        else:
            for subset_seed in [x.strip() for x in args.lds_subset_seeds.split(",") if x.strip()]:
                env = env0 | {"LDS_SAMPLE_RANDOM_SEED": subset_seed, "SAMPLE_MODEL_MODE": "prompted_solo"}
                run(["bash", "scripts/03_lds_training.sh"], env, cwd=args.root, execute=args.execute)
                env = env0 | {"LDS_SAMPLE_RANDOM_SEED": subset_seed, "SAMPLE_MODEL_MODE": "unprompted_solo"}
                run(["bash", "scripts/03_lds_training_unprompted.sh"], env, cwd=args.root, execute=args.execute)

    if not args.skip_attribution:
        if use_parallel:
            jobs = []
            for job_i, (query, algorithm) in enumerate(itertools.product(all_queries, algorithms)):
                _, env = query_env(args, env0, query)
                stage_cwd = args.root / "data_attribution" / algorithm
                slot = slot_for(job_i, len(worker_gpu_ids))
                jobs.append(
                    Job(
                        name=f"attr_{algorithm}_{query_tag(query)}",
                        cmd=attribution_job_cmd(python_bin, algorithm),
                        cwd=stage_cwd,
                        env=gpu_env(env, worker_gpu_ids[slot]),
                        log_path=log_root(args) / "attribution" / algorithm / f"{query_tag(query)}.log",
                        slot=slot,
                    )
                )
            run_parallel_jobs(jobs, args=args, execute=args.execute, max_parallel=len(worker_gpu_ids))
        else:
            for query in all_queries:
                _, env = query_env(args, env0, query)
                for algorithm in algorithms:
                    stage_cwd = args.root / "data_attribution" / algorithm
                    run([python_bin, "02_query_gradient.py"], env, cwd=stage_cwd, execute=args.execute)
                    run([python_bin, "01_train_datapoint_gradient.py"], env, cwd=stage_cwd, execute=args.execute)
                    run([python_bin, "03_score.py"], env, cwd=stage_cwd, execute=args.execute)

    if not args.skip_lds:
        for query in all_queries:
            for algorithm in algorithms:
                for target in TARGET_FUNCTIONS:
                    if query == "unprompted":
                        mode = "unprompted_solo"
                        env = env0 | {
                            "INITIAL_SEED": args.sample_seeds.split(",")[0],
                            "UNPROMPTED": "1",
                            "SAMPLE_MODEL_MODE": mode,
                        }
                        lds_dirs = lds_model_dirs(args.root, args, mode)
                    else:
                        mode = "prompted_solo"
                        env = env0 | {
                            "QUERY": query,
                            "INITIAL_SEED": args.sample_seeds.split(",")[0],
                            "SAMPLE_MODEL_MODE": mode,
                        }
                        lds_dirs = lds_model_dirs(args.root, args, mode)

                    for score_tag, score_dir in attribution_score_dirs(
                        args.root,
                        args,
                        mode=mode,
                        query=query,
                        algorithm=algorithm,
                    ):
                        cmd = [
                            os.environ.get("PYTHON_BIN", "python3"),
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
                            str(
                                lds_eval_out_dir(
                                    args.root,
                                    args,
                                    mode=mode,
                                    query=query,
                                    algorithm=algorithm,
                                    score_tag=score_tag,
                                    target=target,
                                )
                            ),
                        ]
                        if query == "unprompted":
                            cmd.insert(2, "--unprompted")
                        run(cmd, env, cwd=args.root, execute=args.execute)


if __name__ == "__main__":
    main()
