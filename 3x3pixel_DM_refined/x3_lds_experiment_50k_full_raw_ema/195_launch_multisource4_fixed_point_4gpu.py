"""Launch and summarize four-source output-vector composition."""

import argparse
import json
import subprocess
import sys
import time

import numpy as np
from scipy.stats import spearmanr

from multisource4_fixed_point_config import *


def safe_spearman(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if np.std(left) <= 0.0 or np.std(right) <= 0.0:
        return float("nan")
    return float(spearmanr(left, right).statistic)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
        "median": float(np.nanmedian(values)),
        "p05": float(np.nanquantile(values, 0.05)),
        "p95": float(np.nanquantile(values, 0.95)),
    }


def launch(gpus, batch_size):
    MS4_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = MS4_LOG_DIR / "multisource4_fixed_point_4gpu.log"
    active = []
    with open(log_path, "a", buffering=1) as stream:
        for shard_index, gpu in enumerate(gpus):
            command = [
                sys.executable,
                "-u",
                "194_run_multisource4_fixed_point_worker.py",
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
    method_values = {
        method: {
            "pooled_sequence_spearman": [],
            "magnitude_ratio": [],
            "magnitude_relative_error": [],
            "vector_relative_error": [],
            "vector_cosine": [],
        }
        for method in MS4_METHODS
    }
    termwise_ratio = []
    termwise_relative_error = []
    termwise_spearman = []
    cross_term_fraction = []
    sequence_actual = []
    sequence_prediction = {method: [] for method in MS4_METHODS}
    for sequence_index in range(MS4_SEQUENCE_COUNT):
        sequence_dir = ms4_sequence_dir(sequence_index)
        with open(sequence_dir / "done.json") as handle:
            metadata = json.load(handle)
        with np.load(metadata["responses"], allow_pickle=False) as archive:
            actual_l2 = archive["actual_l2"].astype(np.float64)
            actual_squared = np.square(actual_l2)
            sequence_actual.append(float(actual_l2.mean()))
            for method in MS4_METHODS:
                predicted = archive[f"{method}_predicted_l2"].astype(np.float64)
                ratio = predicted / np.maximum(actual_l2, NSDL_EPS)
                method_values[method]["pooled_sequence_spearman"].append(
                    safe_spearman(predicted.reshape(-1), actual_l2.reshape(-1))
                )
                method_values[method]["magnitude_ratio"].extend(ratio.reshape(-1))
                method_values[method]["magnitude_relative_error"].extend(
                    archive[f"{method}_magnitude_relative_error"].reshape(-1)
                )
                method_values[method]["vector_relative_error"].extend(
                    archive[f"{method}_vector_relative_error"].reshape(-1)
                )
                method_values[method]["vector_cosine"].extend(
                    archive[f"{method}_vector_cosine"].reshape(-1)
                )
                sequence_prediction[method].append(float(predicted.mean()))
            termwise = archive["gauss2_termwise_squared"].astype(np.float64)
            ratio = termwise / np.maximum(actual_squared, NSDL_EPS)
            termwise_ratio.extend(ratio.reshape(-1))
            termwise_relative_error.extend(np.abs(ratio - 1.0).reshape(-1))
            termwise_spearman.append(
                safe_spearman(termwise.reshape(-1), actual_squared.reshape(-1))
            )
            cross_term_fraction.extend(
                archive["gauss2_cross_term_fraction"].reshape(-1)
            )
        print(f"[analyzed] sequence={sequence_index}", flush=True)

    output = {
        "definition": {
            "updates": "four sequential restored-AdamW updates from four distinct datapoints and four distinct fixed noise directions",
            "composition": "preserve each 27-dimensional step response, add vectors, then take the norm",
            "termwise_control": "sum the four squared step norms and discard all cross terms",
        },
        "methods": {},
        "termwise_square_control": {
            "pooled_sequence_spearman": summarize(termwise_spearman),
            "squared_magnitude_ratio": summarize(termwise_ratio),
            "squared_magnitude_relative_error": summarize(termwise_relative_error),
        },
        "cross_term_fraction_of_vector_sum_squared": summarize(
            cross_term_fraction
        ),
    }
    for method in MS4_METHODS:
        result = {
            name: summarize(values)
            for name, values in method_values[method].items()
        }
        result["sequence_mean_magnitude_spearman"] = safe_spearman(
            sequence_prediction[method], sequence_actual
        )
        output["methods"][method] = result
    MS4_ROOT.mkdir(parents=True, exist_ok=True)
    with open(MS4_SUMMARY_PATH, "w") as handle:
        json.dump(output, handle, indent=2)

    print("\nfour distinct datapoints/directions: fixed-point response")
    print(
        "method                            pooled-rho  mag-ratio  mag-relerr  "
        "vector-relerr  vector-cos  sequence-rho"
    )
    for method, result in output["methods"].items():
        print(
            f"{method:33s} "
            f"{result['pooled_sequence_spearman']['mean']:+.4f}     "
            f"{result['magnitude_ratio']['median']:.6f}   "
            f"{result['magnitude_relative_error']['median']:.6e}  "
            f"{result['vector_relative_error']['median']:.6e}    "
            f"{result['vector_cosine']['mean']:+.6f}   "
            f"{result['sequence_mean_magnitude_spearman']:+.4f}"
        )
    termwise = output["termwise_square_control"]
    cross = output["cross_term_fraction_of_vector_sum_squared"]
    print("\ntermwise-square control (cross terms discarded)")
    print(
        f"rho={termwise['pooled_sequence_spearman']['mean']:+.4f} "
        f"squared-ratio={termwise['squared_magnitude_ratio']['median']:.4f} "
        f"relative-error={termwise['squared_magnitude_relative_error']['median']:.4f}"
    )
    print(
        "cross-term fraction of final squared norm: "
        f"median={cross['median']:+.4f} "
        f"p05={cross['p05']:+.4f} p95={cross['p95']:+.4f}"
    )
    print(f"\n[saved] {MS4_SUMMARY_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--batch-size", type=int, default=MS4_DEFAULT_BATCH_SIZE)
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
