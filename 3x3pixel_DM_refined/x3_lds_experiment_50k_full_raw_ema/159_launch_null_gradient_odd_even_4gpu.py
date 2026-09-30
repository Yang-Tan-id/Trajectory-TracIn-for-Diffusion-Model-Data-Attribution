"""Launch the null-checkpoint odd/even gradient experiment on several GPUs."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from null_gradient_cross_direction_config import (
    ngcd_odd_even_output_paths,
    nsdl_datapoint_indices,
)


PREDICTORS = (
    "plus_loss_sgd_jvp",
    "minus_loss_sgd_jvp",
    "even_loss_sgd_jvp",
    "odd_loss_sgd_jvp",
    "checkpoint_parameter_delta_jvp",
)
METRICS = (
    "global_cosine",
    "per_timestamp_cosine_mean",
    "per_timestamp_positive_fraction",
    "best_scaled_relative_residual",
)


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def aggregate(point_dir):
    point_results = []
    for datapoint_index in nsdl_datapoint_indices():
        path = point_dir / f"i{datapoint_index:05d}" / "result.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        with open(path) as handle:
            point_results.append(json.load(handle))

    summary = {
        "datapoint_indices": [entry["datapoint_index"] for entry in point_results],
        "evaluation_prompt_mode": point_results[0]["evaluation_prompt_mode"],
        "null_epoch": point_results[0]["null_epoch"],
        "next_epoch": point_results[0]["next_epoch"],
        "directions": {},
        "parameter_cosines_vs_checkpoint_delta": {},
    }
    for direction in ("same", "opposite"):
        summary["directions"][direction] = {}
        for predictor in PREDICTORS:
            predictor_summary = {}
            for metric in METRICS:
                values = np.asarray(
                    [entry["directions"][direction][predictor][metric] for entry in point_results],
                    dtype=np.float64,
                )
                predictor_summary[metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "per_datapoint": values.tolist(),
                }
            summary["directions"][direction][predictor] = predictor_summary
    for predictor in PREDICTORS[:-1]:
        values = np.asarray(
            [entry["parameter_cosines_vs_checkpoint_delta"][predictor] for entry in point_results],
            dtype=np.float64,
        )
        summary["parameter_cosines_vs_checkpoint_delta"][predictor] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "per_datapoint": values.tolist(),
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument(
        "--evaluation-prompt", choices=("original", "random"), default="random"
    )
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU")
    _, point_dir, log_dir, summary_path = ngcd_odd_even_output_paths(
        args.evaluation_prompt
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    for shard_index, gpu in enumerate(gpus):
        log_path = log_dir / f"shard_{shard_index:02d}_of_{len(gpus):02d}.log"
        command = [
            sys.executable,
            "-u",
            "158_run_null_gradient_odd_even_worker.py",
            "--gpu",
            str(gpu),
            "--shard-index",
            str(shard_index),
            "--shard-count",
            str(len(gpus)),
            "--evaluation-prompt",
            args.evaluation_prompt,
        ]
        handle = open(log_path, "w")
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
        processes.append((gpu, process, handle, log_path))
        print(f"[launcher] gpu={gpu} pid={process.pid} log={log_path}", flush=True)
    failures = []
    for gpu, process, handle, log_path in processes:
        code = process.wait()
        handle.close()
        print(f"[launcher] gpu={gpu} code={code}", flush=True)
        if code:
            failures.append((gpu, code, str(log_path)))
    if failures:
        raise RuntimeError(f"worker failures: {failures}")

    summary = aggregate(point_dir)
    atomic_json(summary_path, summary)
    for direction in ("same", "opposite"):
        print(f"[{direction}]", flush=True)
        for predictor in PREDICTORS:
            metric = summary["directions"][direction][predictor]["global_cosine"]
            print(
                f"  {predictor:34s} cosine={metric['mean']:+.6f} ± {metric['std']:.6f}",
                flush=True,
            )
    print(f"[saved] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
