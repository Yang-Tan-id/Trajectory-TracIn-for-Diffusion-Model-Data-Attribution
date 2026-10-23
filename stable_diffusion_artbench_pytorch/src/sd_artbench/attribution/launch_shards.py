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

from .merge_shards import load_shard, merge_shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("retrac", "endpoint"), required=True)
    parser.add_argument(
        "--gpu-ids",
        required=True,
        help="Comma-separated CUDA device IDs to use, for example 0,1,2,3.",
    )
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument(
        "--checkpoint-count",
        type=int,
        default=24,
        help="Number of ReTrac source checkpoint intervals to divide across GPUs.",
    )
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


def save_checkpoint_sum(inputs: list[Path], output: Path) -> None:
    """Sum ReTrac score shards covering disjoint checkpoint intervals."""

    shards = [load_shard(path) for path in inputs]
    reference = shards[0]
    scores = np.zeros_like(reference["scores"], dtype=np.float64)
    checkpoint_indices: list[int] = []
    for shard in shards:
        if shard["scores"].shape != reference["scores"].shape:
            raise ValueError(f"{shard['path']}: incompatible checkpoint-shard shape")
        if not np.array_equal(shard["indices"], reference["indices"]):
            raise ValueError(f"{shard['path']}: training index order differs")
        if not np.array_equal(shard["transforms"], reference["transforms"]):
            raise ValueError(f"{shard['path']}: transform names differ")
        if (shard["variants"] is None) != (reference["variants"] is None):
            raise ValueError(f"{shard['path']}: variant schema differs")
        if reference["variants"] is not None and not np.array_equal(
            shard["variants"], reference["variants"]
        ):
            raise ValueError(f"{shard['path']}: variant names differ")
        for key in (
            "method",
            "parameter_source",
            "model",
            "rank",
            "learning_rate",
            "weight_decay",
            "epochs",
            "batch_size",
            "gradient_accumulation_steps",
            "conditioning_dropout",
            "attention_dropout",
            "per_example_chunk_size",
            "per_example_backend",
            "timesteps",
            "timestep_batch_size",
            "proj_dim",
            "query_ids",
            "query_manifest",
            "max_grad_norm",
            "replay_mode",
            "debug_max_replay_batches",
            "train_positions",
        ):
            if shard["manifest"].get(key) != reference["manifest"].get(key):
                raise ValueError(f"{shard['path']}: manifest field {key!r} differs")
        scores += shard["scores"]
        checkpoint_indices.extend(
            int(value) for value in shard["manifest"]["checkpoint_indices"]
        )
    if len(checkpoint_indices) != len(set(checkpoint_indices)):
        raise ValueError("checkpoint shards overlap")
    checkpoint_indices.sort()
    manifest = dict(reference["manifest"])
    manifest["checkpoint_indices"] = checkpoint_indices
    manifest["completed_through_checkpoint_slot"] = len(checkpoint_indices) - 1
    manifest["merged_checkpoint_shards"] = [str(path) for path in inputs]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    fields = {
        "scores": scores,
        "train_metadata_indices": reference["indices"],
        "transform_names": reference["transforms"],
        "manifest_json": np.asarray(json.dumps(manifest, sort_keys=True)),
    }
    if reference["variants"] is not None:
        fields["variant_names"] = reference["variants"]
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **fields)
    temporary.replace(output)


def main() -> None:
    args = parse_args()
    gpu_ids = [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("--gpu-ids must contain at least one device")
    if args.method == "endpoint" and args.train_size < len(gpu_ids):
        raise ValueError("--train-size must be at least the number of GPUs")
    if args.method == "retrac" and args.checkpoint_count < len(gpu_ids):
        raise ValueError("--checkpoint-count must be at least the number of GPUs")
    extra = list(args.runner_args)
    if extra and extra[0] == "--":
        extra = extra[1:]
    forbidden = {"--train-start", "--train-end", "--output-dir", "--device"}
    if args.method == "retrac":
        forbidden.add("--checkpoint-indices")
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
    item_count = args.checkpoint_count if args.method == "retrac" else args.train_size
    boundaries = np.linspace(0, item_count, len(gpu_ids) + 1, dtype=np.int64)
    try:
        for shard, gpu_id in enumerate(gpu_ids):
            start, end = int(boundaries[shard]), int(boundaries[shard + 1])
            shard_dir = args.output_dir / f"shard-{shard:02d}-gpu-{gpu_id}"
            shard_dir.mkdir(parents=True, exist_ok=True)
            log_path = args.output_dir / f"shard-{shard:02d}-gpu-{gpu_id}.log"
            log_handle = log_path.open("a", encoding="utf-8")
            if args.method == "retrac":
                shard_arguments = [
                    "--checkpoint-indices",
                    ",".join(str(value) for value in range(start, end)),
                ]
                range_description = f"checkpoints={list(range(start, end))}"
            else:
                shard_arguments = [
                    "--train-start",
                    str(start),
                    "--train-end",
                    str(end),
                ]
                range_description = f"train=[{start},{end})"
            command = [
                sys.executable,
                "-m",
                module,
                *shard_arguments,
                "--output-dir",
                str(shard_dir),
                "--device",
                "cuda",
                *extra,
            ]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu_id
            print(
                f"[launch] shard={shard} gpu={gpu_id} {range_description} "
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
            if args.method == "retrac":
                save_checkpoint_sum(inputs, output)
            else:
                save_merged(inputs, output)
            print(f"[merge] {name} -> {output}", flush=True)


if __name__ == "__main__":
    main()
