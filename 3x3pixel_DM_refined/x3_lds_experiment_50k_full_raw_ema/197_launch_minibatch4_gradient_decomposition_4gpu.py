"""Launch and summarize the one-minibatch gradient decomposition test."""

import argparse
import json
import subprocess
import sys
import time

import numpy as np

from checkpoint_counterfactual_metrics import spearman_correlation
from minibatch4_gradient_decomposition_config import *


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.nanmean(values)),
        "median": float(np.nanmedian(values)),
        "p05": float(np.nanquantile(values, 0.05)),
        "p95": float(np.nanquantile(values, 0.95)),
    }


def launch(gpus, batch_size):
    MB4_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = MB4_LOG_DIR / "minibatch4_gradient_decomposition_4gpu.log"
    active = []
    with open(log_path, "a", buffering=1) as stream:
        for shard_index, gpu in enumerate(gpus):
            command = [
                sys.executable, "-u", "196_run_minibatch4_gradient_decomposition_worker.py",
                "--gpu", str(gpu), "--shard-index", str(shard_index),
                "--shard-count", str(len(gpus)), "--batch-size", str(batch_size),
            ]
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
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
    collected = {method: {metric: [] for metric in (
        "predicted_l2", "vector_cosine", "vector_relative_error",
        "magnitude_relative_error",
    )} for method in MB4_METHODS}
    parameter = {"sgd": [], "adam": [], "adam_baseline_fraction": []}
    cross = {"sgd": [], "adam_data": []}
    for sequence_index in range(MB4_SEQUENCE_COUNT):
        with open(mb4_sequence_dir(sequence_index) / "done.json") as handle:
            metadata = json.load(handle)
        parameter["sgd"].append(metadata["sgd_parameter_reconstruction"])
        parameter["adam"].append(metadata["adam_parameter_reconstruction"])
        parameter["adam_baseline_fraction"].append(
            metadata["adam_history_baseline_norm_over_actual"]
        )
        with np.load(metadata["responses"], allow_pickle=False) as archive:
            for method in MB4_METHODS:
                for metric in collected[method]:
                    collected[method][metric].extend(archive[f"{method}_{metric}"].reshape(-1))
            for prefix in cross:
                termwise = archive[f"{prefix}_termwise_squared"].astype(np.float64)
                vector_sum = archive[f"{prefix}_vector_sum_squared"].astype(np.float64)
                cross[prefix].extend(
                    ((vector_sum - termwise) / np.maximum(vector_sum, NSDL_EPS)).reshape(-1)
                )
        print(f"[analyzed] sequence={sequence_index}", flush=True)

    output = {
        "definition": "four datapoint losses averaged into one minibatch and one update",
        "methods": {
            method: {metric: summarize(values) for metric, values in metrics.items()}
            for method, metrics in collected.items()
        },
        "parameter_reconstruction": {
            mode: {
                key: summarize([entry[key] for entry in entries])
                for key in ("cosine", "relative_l2_error")
            }
            for mode, entries in (("sgd", parameter["sgd"]), ("adam", parameter["adam"]))
        },
        "adam_history_baseline_norm_over_actual": summarize(parameter["adam_baseline_fraction"]),
        "cross_term_fraction": {prefix: summarize(values) for prefix, values in cross.items()},
    }
    MB4_ROOT.mkdir(parents=True, exist_ok=True)
    with open(MB4_SUMMARY_PATH, "w") as handle:
        json.dump(output, handle, indent=2)

    print("\none minibatch = mean of four datapoint losses")
    print("method                                  mag-ratio  mag-relerr  vector-relerr  vector-cos")
    for method, result in output["methods"].items():
        actual_key = "sgd_actual_l2" if method.startswith("sgd_") else "adam_actual_l2"
        predicted = np.asarray(collected[method]["predicted_l2"])
        actual = []
        for sequence_index in range(MB4_SEQUENCE_COUNT):
            with open(mb4_sequence_dir(sequence_index) / "done.json") as handle:
                metadata = json.load(handle)
            with np.load(metadata["responses"], allow_pickle=False) as archive:
                actual.extend(archive[actual_key].reshape(-1))
        ratio = predicted / np.maximum(np.asarray(actual), NSDL_EPS)
        rho = spearman_correlation(predicted, actual)
        print(
            f"{method:39s} rho={rho:+.4f} ratio={np.median(ratio):.6f} "
            f"magerr={result['magnitude_relative_error']['median']:.3e} "
            f"vecerr={result['vector_relative_error']['median']:.3e} "
            f"cos={result['vector_cosine']['mean']:+.6f}"
        )
    for mode, result in output["parameter_reconstruction"].items():
        print(
            f"parameter {mode}: cos={result['cosine']['mean']:+.8f} "
            f"relerr={result['relative_l2_error']['median']:.3e}"
        )
    print(
        "cross-term fraction: "
        f"sgd={output['cross_term_fraction']['sgd']['median']:+.4f} "
        f"adam-data={output['cross_term_fraction']['adam_data']['median']:+.4f}"
    )
    print(f"[saved] {MB4_SUMMARY_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--batch-size", type=int, default=MB4_DEFAULT_BATCH_SIZE)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not args.analyze_only:
        launch(gpus, args.batch_size)
    analyze()


if __name__ == "__main__":
    main()
