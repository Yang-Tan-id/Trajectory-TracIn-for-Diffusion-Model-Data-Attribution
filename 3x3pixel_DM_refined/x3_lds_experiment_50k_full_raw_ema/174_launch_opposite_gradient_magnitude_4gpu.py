"""Launch and aggregate opposite-side magnitude prediction."""

import argparse
import json
import subprocess
import sys
import time

import numpy as np

from opposite_gradient_magnitude_config import *


METRICS = (
    "magnitude_pearson",
    "magnitude_spearman",
    "magnitude_ratio_mean",
    "magnitude_ratio_median",
    "magnitude_relative_error_mean",
    "magnitude_relative_error_median",
    "vector_global_cosine",
    "vector_point_cosine_mean",
    "vector_point_cosine_positive_fraction",
    "actual_l2_mean",
    "predicted_l2_mean",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU")
    OGM_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OGM_LOG_DIR / "opposite_gradient_magnitude_4gpu.log"
    active = []
    with open(log_path, "a", buffering=1) as stream:
        for shard_index, gpu in enumerate(gpus):
            command = [
                sys.executable,
                "-u",
                "173_run_opposite_gradient_magnitude_worker.py",
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
        path = ogm_source_dir(source_index) / "result.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        with open(path) as handle:
            results.append(json.load(handle))
    output = {
        "actual_update": "fresh_sgd_plus_direction",
        "source_count": len(results),
        "target_direction_count": NTCD_TARGET_DIRECTION_COUNT,
        "target_timestamps": T,
        "blocks": [],
    }
    for block_index, timestamps in enumerate(NTCD_TIMESTAMP_BLOCKS):
        block_output = {
            "block_index": block_index,
            "timestamp_start": timestamps[0],
            "timestamp_end": timestamps[-1],
            "candidates": {},
        }
        block_output["plus_step_vs_saved_fresh_sgd"] = {}
        for field in ("relative_l2_error", "cosine"):
            values = np.asarray(
                [
                    result["blocks"][block_index][
                        "plus_step_vs_saved_fresh_sgd"
                    ][field]
                    for result in results
                ],
                dtype=np.float64,
            )
            block_output["plus_step_vs_saved_fresh_sgd"][field] = {
                "mean": float(np.mean(values)),
                "max": float(np.max(values)),
                "per_source": values.tolist(),
            }
        for candidate in OGM_CANDIDATES:
            candidate_output = {}
            for metric in METRICS:
                values = np.asarray(
                    [
                        result["blocks"][block_index]["candidates"][candidate][
                            "metrics"
                        ][metric]
                        for result in results
                    ],
                    dtype=np.float64,
                )
                candidate_output[metric] = {
                    "mean": float(np.nanmean(values)),
                    "std": float(np.nanstd(values)),
                    "per_source": values.tolist(),
                }
            block_output["candidates"][candidate] = candidate_output
        output["blocks"].append(block_output)
    OGM_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OGM_SUMMARY_PATH, "w") as handle:
        json.dump(output, handle, indent=2)
    print(
        "block      candidate                    mag_spearman  mag_ratio  "
        "relative_error  vector_cos",
        flush=True,
    )
    for block in output["blocks"]:
        agreement = block["plus_step_vs_saved_fresh_sgd"]
        print(
            f"[step check {block['timestamp_start']:04d}-{block['timestamp_end']:04d}] "
            f"cos={agreement['cosine']['mean']:+.8f} "
            f"relative_l2_error={agreement['relative_l2_error']['mean']:.3e}",
            flush=True,
        )
        for candidate in OGM_CANDIDATES:
            metrics = block["candidates"][candidate]
            print(
                f"{block['timestamp_start']:04d}-{block['timestamp_end']:04d}  "
                f"{candidate:28s} "
                f"{metrics['magnitude_spearman']['mean']:+.5f}  "
                f"{metrics['magnitude_ratio_median']['mean']:.5f}  "
                f"{metrics['magnitude_relative_error_median']['mean']:.5f}  "
                f"{metrics['vector_global_cosine']['mean']:+.5f}",
                flush=True,
            )
    print(f"[saved] {OGM_SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
