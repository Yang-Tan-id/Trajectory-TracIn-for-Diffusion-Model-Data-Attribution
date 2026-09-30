"""Launch and summarize fixed-point path-integrated JVP validation."""

import argparse
import json
import subprocess
import sys
import time

import numpy as np
from scipy.stats import spearmanr

from path_integrated_jvp_config import *


def safe_spearman(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if np.std(left) <= 0.0 or np.std(right) <= 0.0:
        return float("nan")
    return float(spearmanr(left, right).statistic)


def curve_correlations(predicted, actual, axis):
    if axis == "direction":
        predicted, actual = predicted, actual
    elif axis == "timestamp":
        predicted, actual = predicted.T, actual.T
    else:
        raise ValueError(axis)
    return [safe_spearman(left, right) for left, right in zip(predicted, actual)]


def summarize_values(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
        "median": float(np.nanmedian(values)),
        "p90": float(np.nanquantile(values, 0.90)),
        "p95": float(np.nanquantile(values, 0.95)),
    }


def launch(gpus, batch_size):
    PIJVP_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = PIJVP_LOG_DIR / "path_integrated_jvp_4gpu.log"
    active = []
    with open(log_path, "a", buffering=1) as stream:
        for shard_index, gpu in enumerate(gpus):
            command = [
                sys.executable,
                "-u",
                "188_run_path_integrated_jvp_worker.py",
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
    values = {
        method: {
            "pooled_spearman": [],
            "fixed_direction_spearman": [],
            "fixed_t_spearman": [],
            "magnitude_ratio": [],
            "magnitude_relative_error": [],
            "vector_relative_error": [],
            "vector_cosine": [],
        }
        for method in PIJVP_METHODS
    }
    branch_actual = []
    branch_prediction = {method: [] for method in PIJVP_METHODS}
    for source_index in nsdl_datapoint_indices():
        source_dir = pijvp_source_dir(source_index)
        for block_index in range(len(NTCD_TIMESTAMP_BLOCKS)):
            path = source_dir / f"block_{block_index}_path_jvp.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as archive:
                actual = archive["actual_l2"].astype(np.float64)
                branch_actual.append(float(actual.mean()))
                for method in PIJVP_METHODS:
                    predicted = archive[f"{method}_predicted_l2"].astype(np.float64)
                    ratio = predicted / np.maximum(actual, NSDL_EPS)
                    values[method]["pooled_spearman"].append(
                        safe_spearman(predicted.reshape(-1), actual.reshape(-1))
                    )
                    values[method]["fixed_direction_spearman"].extend(
                        curve_correlations(predicted, actual, "direction")
                    )
                    values[method]["fixed_t_spearman"].extend(
                        curve_correlations(predicted, actual, "timestamp")
                    )
                    values[method]["magnitude_ratio"].extend(ratio.reshape(-1))
                    values[method]["magnitude_relative_error"].extend(
                        archive[f"{method}_magnitude_relative_error"].reshape(-1)
                    )
                    values[method]["vector_relative_error"].extend(
                        archive[f"{method}_vector_relative_error"].reshape(-1)
                    )
                    values[method]["vector_cosine"].extend(
                        archive[f"{method}_vector_cosine"].reshape(-1)
                    )
                    branch_prediction[method].append(float(predicted.mean()))
        print(f"[analyzed] source={source_index}", flush=True)

    output = {
        "definition": {
            "target": "finite predicted-noise change vector at one fixed endpoint, pollution direction, timestep, and prompt",
            "parameter_path": "theta(s)=theta+s*Delta-theta for the exact saved fresh-SGD parameter delta",
            "precision_mode": PIJVP_PRECISION_MODE,
            "selection": {
                "directions": list(PIJVP_DIRECTION_INDICES),
                "timestamps": list(PIJVP_TIMESTAMPS),
                "branches": len(branch_actual),
            },
        },
        "methods": {},
    }
    for method in PIJVP_METHODS:
        summary = {
            name: summarize_values(metric_values)
            for name, metric_values in values[method].items()
        }
        summary["branch_mean_magnitude_spearman"] = safe_spearman(
            branch_prediction[method], branch_actual
        )
        output["methods"][method] = summary
    PIJVP_ROOT.mkdir(parents=True, exist_ok=True)
    with open(PIJVP_SUMMARY_PATH, "w") as handle:
        json.dump(output, handle, indent=2)

    print("\nfixed-point predicted-noise change prediction")
    print(f"precision={PIJVP_PRECISION_MODE}")
    print(
        "method                    pooled-rho  dir-rho  fixed-t-rho  "
        "mag-ratio  mag-relerr  vector-relerr  vector-cos  branch-rho"
    )
    for method, summary in output["methods"].items():
        print(
            f"{method:25s} "
            f"{summary['pooled_spearman']['mean']:+.4f}     "
            f"{summary['fixed_direction_spearman']['mean']:+.4f}    "
            f"{summary['fixed_t_spearman']['mean']:+.4f}       "
            f"{summary['magnitude_ratio']['median']:.4f}     "
            f"{summary['magnitude_relative_error']['median']:.4f}      "
            f"{summary['vector_relative_error']['median']:.4f}         "
            f"{summary['vector_cosine']['mean']:+.4f}     "
            f"{summary['branch_mean_magnitude_spearman']:+.4f}"
        )
    print(f"\n[saved] {PIJVP_SUMMARY_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--batch-size", type=int, default=PIJVP_DEFAULT_BATCH_SIZE)
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
