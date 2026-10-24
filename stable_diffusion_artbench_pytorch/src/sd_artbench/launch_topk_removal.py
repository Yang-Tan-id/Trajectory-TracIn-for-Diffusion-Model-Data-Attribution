"""Launch AdamW Top-K removal retraining tasks across explicit GPU IDs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .topk_removal import METHODS


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.replace(",", " ").split() if item]


def parse_strings(value: str) -> list[str]:
    return [item for item in value.replace(",", " ").split() if item]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--query-indices", type=parse_ints, default=list(range(10)))
    parser.add_argument("--methods", type=parse_strings, default=list(METHODS))
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/topk-removal-adamw-q0-9"),
    )
    parser.add_argument(
        "--stage", choices=("prepare", "train", "all"), default="all"
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpu_ids = [item for item in args.gpu_ids.replace(",", " ").split() if item]
    if not gpu_ids:
        raise ValueError("--gpu-ids must contain at least one GPU")
    unknown = set(args.methods).difference(METHODS)
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")

    tasks = [
        (method, query_index)
        for method in args.methods
        for query_index in args.query_indices
    ]
    args.output_root.mkdir(parents=True, exist_ok=True)
    log_root = args.output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    shards = [tasks[slot:: len(gpu_ids)] for slot in range(len(gpu_ids))]

    def run_shard(slot: int) -> None:
        gpu_id = gpu_ids[slot]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu_id
        for method, query_index in shards[slot]:
            log_path = log_root / f"{method}__q{query_index:03d}.log"
            command = [
                sys.executable,
                "-m",
                "sd_artbench.topk_removal",
                "--method",
                method,
                "--query-index",
                str(query_index),
                "--top-k",
                str(args.top_k),
                "--stage",
                args.stage,
                "--output-root",
                str(args.output_root),
            ]
            print(
                f"[launch] slot={slot} gpu={gpu_id} method={method} "
                f"query={query_index} log={log_path}",
                flush=True,
            )
            if args.dry_run:
                continue
            with log_path.open("a", encoding="utf-8") as log:
                process = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment,
                )
            if process.returncode != 0:
                raise RuntimeError(
                    f"task failed with exit={process.returncode}: {log_path}"
                )

    if args.dry_run:
        for slot in range(len(gpu_ids)):
            run_shard(slot)
        return
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = [executor.submit(run_shard, slot) for slot in range(len(gpu_ids))]
        failures = []
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:  # report every failed static shard
                failures.append(error)
        if failures:
            raise RuntimeError("; ".join(str(error) for error in failures))
    print(f"[done] completed {len(tasks)} Top-K removal tasks", flush=True)


if __name__ == "__main__":
    main()
