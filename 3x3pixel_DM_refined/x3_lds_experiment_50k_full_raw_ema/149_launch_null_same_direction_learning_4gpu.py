"""Launch and summarize the ten-point null-model direction experiment."""

import argparse
import json
import subprocess
import sys
import time

import numpy as np

from null_same_direction_learning_config import *


def aggregate_group(results, name):
    keys = (
        "before_mean",
        "after_mean",
        "decrease_mean",
        "relative_decrease_mean",
        "fraction_improved",
    )
    return {
        key: {
            "mean_across_datapoints": float(
                np.mean([result[name][key] for result in results])
            ),
            "std_across_datapoints": float(
                np.std([result[name][key] for result in results])
            ),
            "per_datapoint": [
                float(result[name][key]) for result in results
            ],
        }
        for key in keys
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("at least one GPU is required")
    NSDL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = NSDL_LOG_DIR / "null_same_direction_learning_4gpu.log"
    active = []
    with open(log_path, "a", buffering=1) as stream:
        stream.write("\n[launcher] null-model same-direction learning\n")
        for shard_index, gpu in enumerate(gpus):
            command = [
                sys.executable,
                "-u",
                "148_run_null_same_direction_learning_worker.py",
                "--gpu",
                str(gpu),
                "--shard-index",
                str(shard_index),
                "--shard-count",
                str(len(gpus)),
            ]
            process = subprocess.Popen(
                command, stdout=stream, stderr=subprocess.STDOUT
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
        path = NSDL_POINT_DIR / f"i{datapoint_index:05d}" / "result.json"
        with open(path) as handle:
            results.append(json.load(handle))
    summary = {
        "null_epoch": NSDL_NULL_EPOCH,
        "datapoint_indices": list(nsdl_datapoint_indices()),
        "updates": NSDL_UPDATE_COUNT,
        "timestamps_per_update": NSDL_UPDATE_BATCH_SIZE,
        "random_direction_count_per_datapoint": NSDL_RANDOM_DIRECTION_COUNT,
        "same_direction": aggregate_group(results, "same_direction"),
        "opposite_direction": aggregate_group(results, "opposite_direction"),
        "random_directions": aggregate_group(results, "random_directions"),
        "per_datapoint_results": results,
    }
    NSDL_ROOT.mkdir(parents=True, exist_ok=True)
    with open(NSDL_SUMMARY_PATH, "w") as handle:
        json.dump(summary, handle, indent=2)
    for name in ("same_direction", "opposite_direction", "random_directions"):
        decrease = summary[name]["decrease_mean"]
        relative = summary[name]["relative_decrease_mean"]
        print(
            f"{name:20s} loss_decrease="
            f"{decrease['mean_across_datapoints']:+.6e} ± "
            f"{decrease['std_across_datapoints']:.6e} | relative="
            f"{relative['mean_across_datapoints']:+.6e}",
            flush=True,
        )
    print(f"[saved] {NSDL_SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
