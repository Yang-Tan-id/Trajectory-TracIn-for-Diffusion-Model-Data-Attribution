"""Launch endpoint direction-MC evaluation and summarize proxy reliability."""

import argparse
import json
import subprocess
import sys
import time

import numpy as np
from scipy.stats import pearsonr, spearmanr

from endpoint_direction_mc_config import *


def correlation(left, right, rank=False):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if np.std(left) <= 0 or np.std(right) <= 0:
        return float("nan")
    function = spearmanr if rank else pearsonr
    return float(function(left, right).statistic)


def curve_metrics(proxy, truth, calibrate=False):
    proxy = np.asarray(proxy, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    scale = 1.0
    if calibrate:
        valid = proxy > NSDL_EPS
        if np.any(valid):
            scale = float(
                np.median(truth[valid]) / max(np.median(proxy[valid]), NSDL_EPS)
            )
    estimate = scale * proxy
    relative_error = np.abs(estimate - truth) / np.maximum(truth, NSDL_EPS)
    high_truth = truth >= np.median(truth)
    false_small = (estimate < 0.5 * truth) & high_truth
    return {
        "pearson": correlation(estimate, truth),
        "spearman": correlation(estimate, truth, rank=True),
        "scale": scale,
        "relative_error_mean": float(np.mean(relative_error)),
        "relative_error_median": float(np.median(relative_error)),
        "false_small_rate_high_truth": float(
            np.mean(false_small[high_truth]) if np.any(high_truth) else np.nan
        ),
    }


def aggregate_metric_records(records):
    keys = tuple(records[0])
    output = {"count": len(records)}
    for key in keys:
        values = np.asarray([record[key] for record in records], dtype=np.float64)
        output[key] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
            "median": float(np.nanmedian(values)),
            "min": float(np.nanmin(values)),
            "max": float(np.nanmax(values)),
        }
    return output


def launch(gpus, batch_size):
    EDMC_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = EDMC_LOG_DIR / "endpoint_direction_mc_4gpu.log"
    active = []
    with open(log_path, "a", buffering=1) as stream:
        for shard_index, gpu in enumerate(gpus):
            command = [
                sys.executable,
                "-u",
                "179_run_endpoint_direction_mc_worker.py",
                "--gpu",
                str(gpu),
                "--shard-index",
                str(shard_index),
                "--shard-count",
                str(len(gpus)),
                "--batch-size",
                str(batch_size),
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


def analyze():
    subset_records = {
        estimator: {count: [] for count in EDMC_SUBSET_COUNTS}
        for estimator in ("finite_response", "jvp_response", "loss_projection")
    }
    leave_one_out_records = {
        "finite_single_direction": [],
        "jvp_single_direction": [],
        "loss_single_direction": [],
    }
    exact_jvp_records = []
    branch_count = 0
    for source_index in nsdl_datapoint_indices():
        source_dir = edmc_source_dir(source_index)
        with open(source_dir / "done.json") as handle:
            metadata = json.load(handle)
        for block in metadata["blocks"]:
            block_index = block["block_index"]
            with np.load(block["responses"], allow_pickle=False) as archive:
                actual = archive["actual_sq_l2"].astype(np.float64)
                jvp_value = archive["jvp_sq_l2"].astype(np.float64)
                loss_value = archive[
                    "loss_abs_directional_derivative"
                ].astype(np.float64)
            truth = actual.mean(axis=0)
            exact_jvp_records.append(
                curve_metrics(jvp_value.reshape(-1), actual.reshape(-1))
            )
            for direction_index in range(EDMC_DIRECTION_COUNT):
                other_truth = (
                    (actual.sum(axis=0) - actual[direction_index])
                    / float(EDMC_DIRECTION_COUNT - 1)
                )
                leave_one_out_records["finite_single_direction"].append(
                    curve_metrics(actual[direction_index], other_truth)
                )
                leave_one_out_records["jvp_single_direction"].append(
                    curve_metrics(jvp_value[direction_index], other_truth)
                )
                leave_one_out_records["loss_single_direction"].append(
                    curve_metrics(
                        loss_value[direction_index], other_truth, calibrate=True
                    )
                )
            generator = np.random.default_rng(
                EDMC_DIRECTION_SEED_BASE + 17 * source_index + block_index
            )
            for count in EDMC_SUBSET_COUNTS:
                repeats = 1 if count == EDMC_DIRECTION_COUNT else EDMC_SUBSET_REPEATS
                for _ in range(repeats):
                    selection = generator.choice(
                        EDMC_DIRECTION_COUNT, size=count, replace=False
                    )
                    subset_records["finite_response"][count].append(
                        curve_metrics(actual[selection].mean(axis=0), truth)
                    )
                    subset_records["jvp_response"][count].append(
                        curve_metrics(jvp_value[selection].mean(axis=0), truth)
                    )
                    subset_records["loss_projection"][count].append(
                        curve_metrics(
                            loss_value[selection].mean(axis=0),
                            truth,
                            calibrate=True,
                        )
                    )
            branch_count += 1
    output = {
        "definition": {
            "endpoint": "paired target datapoint with own prompt",
            "ground_truth": "100-direction mean finite predicted-noise delta squared L2 at each noise level",
            "loss_proxy": "absolute first-order per-example diffusion-MSE change",
            "loss_proxy_calibration": "per-curve median multiplicative calibration",
            "false_small": "calibrated estimate < 0.5 * truth among noise levels at or above median truth",
        },
        "source_count": len(nsdl_datapoint_indices()),
        "branch_count": branch_count,
        "direction_count": EDMC_DIRECTION_COUNT,
        "noise_level_count": T,
        "exact_parameter_delta_jvp_vs_finite": aggregate_metric_records(
            exact_jvp_records
        ),
        "single_direction_leave_one_out": {
            name: aggregate_metric_records(records)
            for name, records in leave_one_out_records.items()
        },
        "mc_subset_sweep": {
            estimator: {
                str(count): aggregate_metric_records(records)
                for count, records in count_records.items()
            }
            for estimator, count_records in subset_records.items()
        },
    }
    EDMC_ROOT.mkdir(parents=True, exist_ok=True)
    with open(EDMC_SUMMARY_PATH, "w") as handle:
        json.dump(output, handle, indent=2)

    print("single direction -> mean of other 99 directions", flush=True)
    print("estimator                   spearman  relerr  false-small", flush=True)
    for name, metrics in output["single_direction_leave_one_out"].items():
        print(
            f"{name:27s} "
            f"{metrics['spearman']['mean']:+.4f}  "
            f"{metrics['relative_error_median']['mean']:.4f}  "
            f"{metrics['false_small_rate_high_truth']['mean']:.4f}",
            flush=True,
        )
    print("\nMC directions -> 100-direction truth", flush=True)
    print("estimator          R    spearman  relerr  false-small", flush=True)
    for estimator, counts in output["mc_subset_sweep"].items():
        for count in EDMC_SUBSET_COUNTS:
            metrics = counts[str(count)]
            print(
                f"{estimator:18s} {count:3d}  "
                f"{metrics['spearman']['mean']:+.4f}  "
                f"{metrics['relative_error_median']['mean']:.4f}  "
                f"{metrics['false_small_rate_high_truth']['mean']:.4f}",
                flush=True,
            )
    print(f"[saved] {EDMC_SUMMARY_PATH}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--batch-size", type=int, default=EDMC_DEFAULT_BATCH_SIZE)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not args.analyze_only:
        if not gpus:
            raise ValueError("--gpus must contain at least one GPU")
        launch(gpus, args.batch_size)
    analyze()


if __name__ == "__main__":
    main()
