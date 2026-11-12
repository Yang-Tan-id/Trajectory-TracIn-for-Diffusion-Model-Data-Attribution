"""Launch DAS lambda=1000 ArtBench Top-1000 removal retraining on multiple GPUs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.replace(",", " ").split() if item]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--query-indices", type=parse_ints, default=list(range(60)))
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/topk-removal-das-lambda1000-q0-59"),
    )
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpu_ids = [item for item in args.gpu_ids.replace(",", " ").split() if item]
    if not gpu_ids:
        raise ValueError("--gpu-ids must contain at least one GPU")
    if len(args.query_indices) != len(set(args.query_indices)):
        raise ValueError("query indices must be unique")
    if any(query < 0 or query >= 60 for query in args.query_indices):
        raise ValueError("this experiment only contains Q0-Q59")
    args.output_root.mkdir(parents=True, exist_ok=True)
    log_root = args.output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    shards = [args.query_indices[slot:: len(gpu_ids)] for slot in range(len(gpu_ids))]

    def run_shard(slot: int) -> None:
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu_ids[slot]
        for query in shards[slot]:
            log_path = log_root / f"das_lambda_1000__q{query:03d}.log"
            command = [
                sys.executable,
                "-m",
                "sd_artbench.das_topk_removal",
                "--query-index",
                str(query),
                "--top-k",
                str(args.top_k),
                "--output-root",
                str(args.output_root),
            ]
            if args.prepare_only:
                command.append("--prepare-only")
            print(
                f"[launch] slot={slot} gpu={gpu_ids[slot]} Q{query} log={log_path}",
                flush=True,
            )
            with log_path.open("a", encoding="utf-8") as log:
                result = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, env=environment
                )
            if result.returncode:
                raise RuntimeError(f"Q{query} failed with exit={result.returncode}: {log_path}")

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = [executor.submit(run_shard, slot) for slot in range(len(gpu_ids))]
        failures = []
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                failures.append(error)
        if failures:
            raise RuntimeError("; ".join(map(str, failures)))
    print(f"[done] completed {len(args.query_indices)} DAS removal tasks", flush=True)


if __name__ == "__main__":
    main()
