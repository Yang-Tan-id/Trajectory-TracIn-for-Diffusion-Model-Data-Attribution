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
    return values


def subset_chunks(m: int, workers: int) -> list[list[int]]:
    chunks = [[] for _ in range(workers)]
    for subset_id in range(m):
        chunks[subset_id % workers].append(subset_id)
    return chunks


def run_checked(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one LDS subset-seed run with one independent process per GPU."
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
    try:
        gpus = parse_gpus(args.gpus)
    except ValueError as exc:
        parser.error(str(exc))

    root = Path(__file__).resolve().parents[1]
    python = os.environ.get("PYTHON_BIN", sys.executable)
    base_cmd = [
        python,
        "lds/run_training.py",
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

    # Generate the deterministic subset files exactly once before workers start.
    run_checked(base_cmd + ["--dry-run"], cwd=root, env=base_env)

    log_root = (
        root
        / "result"
        / os.environ.get("EXPERIMENT_TAG", "experiment1")
        / "logs"
        / "lds"
        / f"subset_seed_{args.subset_seed}"
    )
    log_root.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[str, subprocess.Popen, object]] = []
    for gpu, subset_ids in zip(gpus, subset_chunks(args.m, len(gpus))):
        if not subset_ids:
            continue
        worker_env = base_env.copy()
        worker_env["CUDA_VISIBLE_DEVICES"] = gpu
        worker_env["LDS_WORKER_GPU"] = gpu
        ids = ",".join(str(value) for value in subset_ids)
        log_path = log_root / f"gpu_{gpu}.log"
        log_file = log_path.open("ab")
        cmd = base_cmd + ["--reuse-prepared", "--subset-indices", ids]
        print(f"[launch] gpu={gpu} subsets={ids} log={log_path}", flush=True)
        process = subprocess.Popen(
            cmd,
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
        raise RuntimeError(f"LDS GPU workers failed: {failures}; inspect {log_root}")

    finalize_env = base_env.copy()
    finalize_env["CUDA_VISIBLE_DEVICES"] = gpus[0]
    run_checked(base_cmd + ["--reuse-prepared", "--finalize-only"], cwd=root, env=finalize_env)
    print(f"Completed LDS subset seed {args.subset_seed} with {len(gpus)} independent GPU workers")


if __name__ == "__main__":
    main()

