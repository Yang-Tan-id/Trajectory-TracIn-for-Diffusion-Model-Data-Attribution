"""Launch and summarize source-gradient transfer to different datapoints."""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

from null_gradient_cross_direction_config import *


PREDICTORS = (
    "plus_loss_sgd_jvp",
    "even_loss_sgd_jvp",
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


def load_results(pair_dir):
    results = []
    for source_index in nsdl_datapoint_indices():
        target_index = ngcd_cross_datapoint_target_index(source_index)
        path = (
            pair_dir
            / f"source_{source_index:05d}_target_{target_index:05d}"
            / "result.json"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        with open(path) as handle:
            results.append(json.load(handle))
    return results


def aggregate(results):
    summary = {
        "null_epoch": NGCD_NULL_EPOCH,
        "next_epoch": NGCD_NEXT_EPOCH,
        "pairs": [
            {
                "source": entry["source_datapoint_index"],
                "target": entry["target_datapoint_index"],
            }
            for entry in results
        ],
        "target_uses_own_prompt": True,
        "directions": {},
    }
    for direction in ("same", "opposite"):
        summary["directions"][direction] = {}
        for predictor in PREDICTORS:
            summary["directions"][direction][predictor] = {}
            for metric in METRICS:
                values = np.asarray(
                    [entry["directions"][direction][predictor][metric] for entry in results],
                    dtype=np.float64,
                )
                summary["directions"][direction][predictor][metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "per_pair": values.tolist(),
                }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU")
    _, pair_dir, log_dir, summary_path = ngcd_cross_datapoint_output_paths()
    log_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    for shard_index, gpu in enumerate(gpus):
        log_path = log_dir / f"shard_{shard_index:02d}_of_{len(gpus):02d}.log"
        command = [
            sys.executable,
            "-u",
            "161_run_null_gradient_even_cross_datapoint_worker.py",
            "--gpu",
            str(gpu),
            "--shard-index",
            str(shard_index),
            "--shard-count",
            str(len(gpus)),
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

    summary = aggregate(load_results(pair_dir))
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
