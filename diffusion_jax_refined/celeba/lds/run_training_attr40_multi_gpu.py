#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_gpus(text: str) -> list[str]:
    values = [part.strip() for part in text.replace(" ", ",").split(",") if part.strip()]
    if not values:
        raise ValueError("--gpus must contain at least one GPU id")
    if len(set(values)) != len(values):
        raise ValueError("--gpus must not contain duplicate GPU ids")
    return values


def subset_chunks(m: int, workers: int) -> list[list[int]]:
    chunks = [[] for _ in range(workers)]
    for subset_id in range(m):
        chunks[subset_id % workers].append(subset_id)
    return chunks


def run_checked(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def terminate_workers(processes: list[tuple[str, subprocess.Popen, object]]) -> None:
    for _, process, _ in processes:
        if process.poll() is None:
            process.terminate()
    for _, process, log_file in processes:
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log_file.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one conditional CelebA-Attr40 LDS seed with one process per GPU."
    )
    parser.add_argument("--gpus", default=os.environ.get("LDS_WORKER_GPUS", "0,1,2,3"))
    parser.add_argument(
        "--subset-seed",
        type=int,
        default=int(os.environ.get("LDS_SAMPLE_RANDOM_SEED", "0")),
    )
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--k", type=int, default=2500)
    args = parser.parse_args()
    if args.m <= 0 or args.k <= 0:
        parser.error("--m and --k must be positive")
    if args.subset_seed < 0:
        parser.error("--subset-seed must be nonnegative")
    try:
        gpus = parse_gpus(args.gpus)
    except ValueError as exc:
        parser.error(str(exc))

    root = Path(__file__).resolve().parents[1]
    python = os.environ.get("PYTHON_BIN", sys.executable)
    base_command = [
        python,
        "lds/run_training_attr40.py",
        "--sample-model-mode",
        "prompted_solo",
        "--m",
        str(args.m),
        "--k",
        str(args.k),
        "--sample-random-seed",
        str(args.subset_seed),
    ]
    base_env = os.environ.copy()
    base_env.update(
        SAMPLE_MODEL_MODE="prompted_solo",
        LDS_NUM_DEVICES="1",
        JAX_NUM_DEVICES="1",
        JAX_DATA_PARALLEL="0",
    )

    run_checked(base_command + ["--dry-run"], cwd=root, env=base_env)

    log_root = (
        root
        / "result"
        / os.environ.get("EXPERIMENT_TAG", "attr40_experiment1")
        / "logs"
        / "lds"
        / f"subset_seed_{args.subset_seed}"
    )
    log_root.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[str, subprocess.Popen, object]] = []
    try:
        for gpu, subset_ids in zip(gpus, subset_chunks(args.m, len(gpus))):
            if not subset_ids:
                continue
            worker_env = base_env.copy()
            worker_env["CUDA_VISIBLE_DEVICES"] = gpu
            worker_env["LDS_WORKER_GPU"] = gpu
            ids = ",".join(str(value) for value in subset_ids)
            log_path = log_root / f"gpu_{gpu}.log"
            log_file = log_path.open("ab")
            command = base_command + ["--reuse-prepared", "--subset-indices", ids]
            print(f"[launch] gpu={gpu} subsets={ids} log={log_path}", flush=True)
            process = subprocess.Popen(
                command,
                cwd=root,
                env=worker_env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            processes.append((gpu, process, log_file))

        failures = []
        for gpu, process, log_file in processes:
            return_code = process.wait()
            log_file.close()
            if return_code:
                failures.append((gpu, return_code))
        if failures:
            raise RuntimeError(f"CelebA-Attr40 LDS workers failed: {failures}; inspect {log_root}")
    except BaseException:
        terminate_workers(processes)
        raise

    finalize_env = base_env.copy()
    finalize_env["CUDA_VISIBLE_DEVICES"] = gpus[0]
    run_checked(
        base_command + ["--reuse-prepared", "--finalize-only"],
        cwd=root,
        env=finalize_env,
    )
    print(
        f"Completed CelebA-Attr40 LDS subset seed {args.subset_seed} "
        f"with {len(gpus)} GPU workers"
    )


if __name__ == "__main__":
    main()
