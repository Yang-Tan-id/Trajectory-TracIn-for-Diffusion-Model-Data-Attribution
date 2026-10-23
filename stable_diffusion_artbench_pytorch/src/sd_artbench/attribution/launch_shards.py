"""Launch attribution shards on an explicit list of GPU IDs and merge them."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

from .merge_shards import merge_shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("retrac", "endpoint"), required=True)
    parser.add_argument(
        "--gpu-ids",
        required=True,
        help="Comma-separated CUDA device IDs to use, for example 0,1,2,3.",
    )
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def save_merged(inputs: list[Path], output: Path) -> None:
    scores, indices, manifest, transforms, variants = merge_shards(inputs)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    fields = {
        "scores": scores,
        "train_metadata_indices": indices,
        "transform_names": transforms,
        "manifest_json": np.asarray(json.dumps(manifest, sort_keys=True)),
    }
    if variants is not None:
        fields["variant_names"] = variants
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **fields)
    temporary.replace(output)


def main() -> None:
    args = parse_args()
    gpu_ids = [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("--gpu-ids must contain at least one device")
    if args.train_size < len(gpu_ids):
        raise ValueError("--train-size must be at least the number of GPUs")
    extra = list(args.runner_args)
    if extra and extra[0] == "--":
        extra = extra[1:]
    forbidden = {"--train-start", "--train-end", "--output-dir", "--device"}
    if forbidden.intersection(extra):
        raise ValueError(
            "shard ranges, output directory, and device are controlled by the launcher"
        )

    module = (
        "sd_artbench.attribution.run_retrac"
        if args.method == "retrac"
        else "sd_artbench.attribution.run_endpoint"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[subprocess.Popen, object, Path]] = []
    boundaries = np.linspace(0, args.train_size, len(gpu_ids) + 1, dtype=np.int64)
    try:
        for shard, gpu_id in enumerate(gpu_ids):
            start, end = int(boundaries[shard]), int(boundaries[shard + 1])
            shard_dir = args.output_dir / f"shard-{shard:02d}-gpu-{gpu_id}"
            shard_dir.mkdir(parents=True, exist_ok=True)
            log_path = args.output_dir / f"shard-{shard:02d}-gpu-{gpu_id}.log"
            log_handle = log_path.open("a", encoding="utf-8")
            command = [
                sys.executable,
                "-m",
                module,
                "--train-start",
                str(start),
                "--train-end",
                str(end),
                "--output-dir",
                str(shard_dir),
                "--device",
                "cuda",
                *extra,
            ]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu_id
            print(
                f"[launch] shard={shard} gpu={gpu_id} train=[{start},{end}) "
                f"log={log_path}\n  {shlex.join(command)}",
                flush=True,
            )
            if args.dry_run:
                log_handle.close()
                continue
            process = subprocess.Popen(
                command,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            processes.append((process, log_handle, log_path))

        if args.dry_run:
            return

        failures = []
        for process, log_handle, log_path in processes:
            return_code = process.wait()
            log_handle.close()
            if return_code:
                failures.append((return_code, log_path))
        if failures:
            details = ", ".join(f"exit={code} log={path}" for code, path in failures)
            raise RuntimeError(f"one or more attribution shards failed: {details}")
    except KeyboardInterrupt:
        for process, log_handle, _ in processes:
            if process.poll() is None:
                process.terminate()
            log_handle.close()
        raise

    if args.method == "retrac":
        names = ["retrac.npz"]
    else:
        names = [
            "endpoint_tracin.npz",
            "endpoint_pollute_linear.npz",
            "endpoint_pollute_termwise_squared.npz",
            "endpoint_pollute_timestamp_sum_squared.npz",
            "endpoint_pollute_total_sum_squared.npz",
        ]
    for name in names:
        inputs = [
            args.output_dir / f"shard-{shard:02d}-gpu-{gpu_id}" / name
            for shard, gpu_id in enumerate(gpu_ids)
        ]
        if all(path.is_file() for path in inputs):
            output = args.output_dir / name
            save_merged(inputs, output)
            print(f"[merge] {name} -> {output}", flush=True)


if __name__ == "__main__":
    main()
