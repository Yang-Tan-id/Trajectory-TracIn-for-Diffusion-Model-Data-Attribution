"""Launch and compare aligned versus non-aligned target loss noises."""

import argparse
import importlib
import json
import subprocess
import sys
import time

import numpy as np

from endpoint_direction_nonaligned_config import *


analysis = importlib.import_module("180_launch_endpoint_direction_mc_4gpu")


def launch(gpus, batch_size):
    EDNA_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = EDNA_LOG_DIR / "endpoint_direction_nonaligned_4gpu.log"
    active = []
    with open(log_path, "a", buffering=1) as stream:
        for shard_index, gpu in enumerate(gpus):
            command = [
                sys.executable,
                "-u",
                "182_run_endpoint_direction_nonaligned_worker.py",
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
    estimators = ("aligned", "nonaligned")
    subset_records = {
        estimator: {count: [] for count in EDMC_SUBSET_COUNTS}
        for estimator in estimators
    }
    leave_one_out = {estimator: [] for estimator in estimators}
    mismatch_cosines = []
    branch_count = 0
    for source_index in nsdl_datapoint_indices():
        aligned_dir = edmc_source_dir(source_index)
        nonaligned_dir = edna_source_dir(source_index)
        with open(nonaligned_dir / "done.json") as handle:
            nonaligned_metadata = json.load(handle)
        mismatch_cosines.append(
            nonaligned_metadata["pollution_to_loss_noise_cosine_mean"]
        )
        for block_index in range(len(NTCD_TIMESTAMP_BLOCKS)):
            with np.load(
                aligned_dir / f"block_{block_index}_responses.npz",
                allow_pickle=False,
            ) as archive:
                actual = archive["actual_sq_l2"].astype(np.float64)
                aligned = archive[
                    "loss_abs_directional_derivative"
                ].astype(np.float64)
            with np.load(
                nonaligned_dir / f"block_{block_index}_nonaligned_loss.npz",
                allow_pickle=False,
            ) as archive:
                nonaligned = archive[
                    "loss_abs_directional_derivative"
                ].astype(np.float64)
            truth = actual.mean(axis=0)
            banks = {"aligned": aligned, "nonaligned": nonaligned}
            for direction_index in range(EDMC_DIRECTION_COUNT):
                other_truth = (
                    actual.sum(axis=0) - actual[direction_index]
                ) / float(EDMC_DIRECTION_COUNT - 1)
                for estimator, bank in banks.items():
                    leave_one_out[estimator].append(
                        analysis.curve_metrics(
                            bank[direction_index], other_truth, calibrate=True
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
                    for estimator, bank in banks.items():
                        subset_records[estimator][count].append(
                            analysis.curve_metrics(
                                bank[selection].mean(axis=0),
                                truth,
                                calibrate=True,
                            )
                        )
            branch_count += 1
    output = {
        "definition": {
            "aligned": "pollution noise epsilon_r is also the diffusion-loss target",
            "nonaligned": "pollution uses epsilon_r while loss target uses epsilon_(r+1 mod 100)",
            "ground_truth": "100-direction mean finite predicted-noise delta squared L2",
            "calibration": "per-curve median multiplicative calibration for both loss estimators",
        },
        "branch_count": branch_count,
        "pollution_to_nonaligned_loss_noise_cosine": {
            "mean": float(np.mean(mismatch_cosines)),
            "std": float(np.std(mismatch_cosines)),
        },
        "single_direction_leave_one_out": {
            estimator: analysis.aggregate_metric_records(records)
            for estimator, records in leave_one_out.items()
        },
        "mc_subset_sweep": {
            estimator: {
                str(count): analysis.aggregate_metric_records(records)
                for count, records in count_records.items()
            }
            for estimator, count_records in subset_records.items()
        },
    }
    EDNA_ROOT.mkdir(parents=True, exist_ok=True)
    with open(EDNA_SUMMARY_PATH, "w") as handle:
        json.dump(output, handle, indent=2)
    cosine = output["pollution_to_nonaligned_loss_noise_cosine"]
    print(
        f"pollution/loss-noise cosine={cosine['mean']:+.4f}±{cosine['std']:.4f}",
        flush=True,
    )
    print("single direction -> mean other 99", flush=True)
    print("mode          spearman  relerr  false-small", flush=True)
    for estimator, metrics in output["single_direction_leave_one_out"].items():
        print(
            f"{estimator:12s}  "
            f"{metrics['spearman']['mean']:+.4f}  "
            f"{metrics['relative_error_median']['mean']:.4f}  "
            f"{metrics['false_small_rate_high_truth']['mean']:.4f}",
            flush=True,
        )
    print("\nR-direction loss average -> finite 100-direction truth", flush=True)
    print("R    aligned rho/error/false      nonaligned rho/error/false", flush=True)
    for count in EDMC_SUBSET_COUNTS:
        aligned = output["mc_subset_sweep"]["aligned"][str(count)]
        nonaligned = output["mc_subset_sweep"]["nonaligned"][str(count)]
        print(
            f"{count:3d}  "
            f"{aligned['spearman']['mean']:+.4f}/"
            f"{aligned['relative_error_median']['mean']:.4f}/"
            f"{aligned['false_small_rate_high_truth']['mean']:.4f}      "
            f"{nonaligned['spearman']['mean']:+.4f}/"
            f"{nonaligned['relative_error_median']['mean']:.4f}/"
            f"{nonaligned['false_small_rate_high_truth']['mean']:.4f}",
            flush=True,
        )
    print(f"[saved] {EDNA_SUMMARY_PATH}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--batch-size", type=int, default=EDMC_DEFAULT_BATCH_SIZE)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if not args.analyze_only:
        gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
        if not gpus:
            raise ValueError("--gpus must contain at least one GPU")
        launch(gpus, args.batch_size)
    analyze()


if __name__ == "__main__":
    main()
