"""Launch the 40-model timestamp-block cross-direction experiment."""

import argparse
import json
import subprocess
import sys
import time

import numpy as np

from null_tblock_cross_direction_config import *


METRICS = (
    "delta_l2_mean",
    "delta_rmse_mean",
    "delta_max_abs",
    "global_off_diagonal_delta_cosine_mean",
    "per_timestamp_off_diagonal_delta_cosine_mean",
    "noise_cosine_vs_delta_cosine_correlation",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU")
    NTCD_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = NTCD_LOG_DIR / "null_tblock_cross_direction_4gpu.log"
    active = []
    with open(log_path, "a", buffering=1) as stream:
        stream.write("\n[launcher] null timestamp-block cross-direction experiment\n")
        for shard_index, gpu in enumerate(gpus):
            command = [
                sys.executable,
                "-u",
                "167_run_null_tblock_cross_direction_worker.py",
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
    for source_index in nsdl_datapoint_indices():
        path = ntcd_source_dir(source_index) / "result.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        with open(path) as handle:
            results.append(json.load(handle))
    summary = {
        "null_epoch": NSDL_NULL_EPOCH,
        "source_count": len(results),
        "updated_model_count": len(results) * len(NTCD_TIMESTAMP_BLOCKS),
        "target_direction_count": NTCD_TARGET_DIRECTION_COUNT,
        "timestamp_blocks_are_independent_null_branches": True,
        "pairs": [
            {
                "source": result["source_datapoint_index"],
                "target": result["target_datapoint_index"],
            }
            for result in results
        ],
        "blocks": [],
    }
    for block_index, timestamps in enumerate(NTCD_TIMESTAMP_BLOCKS):
        block_summary = {
            "block_index": block_index,
            "timestamp_start": timestamps[0],
            "timestamp_end": timestamps[-1],
            "metrics": {},
        }
        for metric in METRICS:
            values = np.asarray(
                [result["blocks"][block_index]["metrics"][metric] for result in results],
                dtype=np.float64,
            )
            block_summary["metrics"][metric] = {
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
                "per_source": values.tolist(),
            }
        summary["blocks"].append(block_summary)
    NTCD_ROOT.mkdir(parents=True, exist_ok=True)
    with open(NTCD_SUMMARY_PATH, "w") as handle:
        json.dump(summary, handle, indent=2)
    print("block       target delta L2        cross-direction delta cosine", flush=True)
    for block in summary["blocks"]:
        l2 = block["metrics"]["delta_l2_mean"]
        cosine = block["metrics"]["global_off_diagonal_delta_cosine_mean"]
        print(
            f"{block['timestamp_start']:04d}-{block['timestamp_end']:04d}  "
            f"{l2['mean']:.6e} ± {l2['std']:.6e}   "
            f"{cosine['mean']:+.6f} ± {cosine['std']:.6f}",
            flush=True,
        )
    print(f"[saved] {NTCD_SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
