"""Launch and summarize null-gradient cross-direction prediction."""

import argparse
import json
import subprocess
import sys
import time

import numpy as np

from null_gradient_cross_direction_config import *


def aggregate(results, direction_name, predictor):
    keys = (
        "global_cosine",
        "per_timestamp_cosine_mean",
        "per_timestamp_positive_fraction",
        "best_scaled_relative_residual",
    )
    return {
        key: {
            "mean": float(
                np.mean(
                    [
                        result["directions"][direction_name][predictor][key]
                        for result in results
                    ]
                )
            ),
            "std": float(
                np.std(
                    [
                        result["directions"][direction_name][predictor][key]
                        for result in results
                    ]
                )
            ),
            "per_datapoint": [
                float(result["directions"][direction_name][predictor][key])
                for result in results
            ],
        }
        for key in keys
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument(
        "--evaluation-prompt", choices=("original", "random"), default="original"
    )
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("at least one GPU is required")
    root, point_dir, log_dir, summary_path = ngcd_output_paths(
        args.evaluation_prompt
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "null_gradient_cross_direction_4gpu.log"
    active = []
    with open(log_path, "a", buffering=1) as stream:
        stream.write("\n[launcher] null-gradient cross-direction JVP\n")
        for shard_index, gpu in enumerate(gpus):
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "151_run_null_gradient_cross_direction_worker.py",
                    "--gpu",
                    str(gpu),
                    "--shard-index",
                    str(shard_index),
                    "--shard-count",
                    str(len(gpus)),
                    "--evaluation-prompt",
                    args.evaluation_prompt,
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            active.append((gpu, process))
            print(f"[launcher] gpu={gpu} pid={process.pid}", flush=True)
        print(f"[launcher] log={log_path}", flush=True)
        while active:
            for item in list(active):
                gpu, process = item
                code = process.poll()
                if code is None:
                    continue
                active.remove(item)
                print(f"[launcher] gpu={gpu} code={code}", flush=True)
                if code:
                    for _, other in active:
                        other.terminate()
                    raise SystemExit(code)
            if active:
                time.sleep(2)
    results = []
    for datapoint_index in nsdl_datapoint_indices():
        with open(
            point_dir / f"i{datapoint_index:05d}" / "result.json"
        ) as handle:
            results.append(json.load(handle))
    summary = {
        "null_epoch": NGCD_NULL_EPOCH,
        "next_epoch": NGCD_NEXT_EPOCH,
        "evaluation_prompt_mode": args.evaluation_prompt,
        "datapoint_indices": list(nsdl_datapoint_indices()),
        "same": {
            predictor: aggregate(results, "same", predictor)
            for predictor in (
                "single_point_sgd_jvp",
                "checkpoint_parameter_delta_jvp",
            )
        },
        "opposite": {
            predictor: aggregate(results, "opposite", predictor)
            for predictor in (
                "single_point_sgd_jvp",
                "checkpoint_parameter_delta_jvp",
            )
        },
        "parameter_cosine_single_point_sgd_vs_checkpoint_delta": {
            "mean": float(
                np.mean(
                    [
                        result[
                            "parameter_cosine_single_point_sgd_vs_checkpoint_delta"
                        ]
                        for result in results
                    ]
                )
            ),
            "per_datapoint": [
                result["parameter_cosine_single_point_sgd_vs_checkpoint_delta"]
                for result in results
            ],
        },
        "per_datapoint_results": results,
    }
    root.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    for direction_name in ("same", "opposite"):
        primary = summary[direction_name]["single_point_sgd_jvp"]
        control = summary[direction_name]["checkpoint_parameter_delta_jvp"]
        print(
            f"{direction_name:8s} single-point J(-g) cosine="
            f"{primary['global_cosine']['mean']:+.6f} ± "
            f"{primary['global_cosine']['std']:.6f} | "
            f"checkpoint-delta JVP control="
            f"{control['global_cosine']['mean']:+.6f} ± "
            f"{control['global_cosine']['std']:.6f}",
            flush=True,
        )
    print(f"[saved] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
