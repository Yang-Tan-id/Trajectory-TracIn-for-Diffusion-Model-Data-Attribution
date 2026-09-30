"""Launch and summarize sequential four-update fixed-point validation."""

import argparse
import json
import subprocess
import sys
import time

import numpy as np
from scipy.stats import spearmanr

from sequential4_fixed_point_config import *


def safe_spearman(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if np.std(left) <= 0.0 or np.std(right) <= 0.0:
        return float("nan")
    return float(spearmanr(left, right).statistic)


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
        "median": float(np.nanmedian(values)),
        "p95": float(np.nanquantile(values, 0.95)),
        "max": float(np.nanmax(values)),
    }


def launch(gpus, batch_size):
    SEQ4_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = SEQ4_LOG_DIR / "sequential4_fixed_point_4gpu.log"
    active = []
    with open(log_path, "a", buffering=1) as stream:
        for shard_index, gpu in enumerate(gpus):
            command = [
                sys.executable,
                "-u",
                "192_run_sequential4_fixed_point_worker.py",
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
            "pooled_branch_spearman": [],
            "magnitude_ratio": [],
            "magnitude_relative_error": [],
            "vector_relative_error": [],
            "vector_cosine": [],
        }
        for method in SEQ4_METHODS
    }
    branch_actual = []
    branch_predicted = {method: [] for method in SEQ4_METHODS}
    replay_cosines = []
    replay_errors = []
    for source_index in nsdl_datapoint_indices():
        source_dir = seq4_source_dir(source_index)
        with open(source_dir / "done.json") as handle:
            metadata = json.load(handle)
        agreement = metadata["replayed_final_vs_saved_final"]
        replay_cosines.append(agreement["cosine"])
        replay_errors.append(agreement["relative_l2_error"])
        with np.load(metadata["responses"], allow_pickle=False) as archive:
            actual = archive["actual_l2"].astype(np.float64)
            branch_actual.append(float(actual.mean()))
            for method in SEQ4_METHODS:
                predicted = archive[f"{method}_predicted_l2"].astype(np.float64)
                ratio = predicted / np.maximum(actual, NSDL_EPS)
                values[method]["pooled_branch_spearman"].append(
                    safe_spearman(predicted.reshape(-1), actual.reshape(-1))
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
                branch_predicted[method].append(float(predicted.mean()))
        print(f"[analyzed] source={source_index}", flush=True)

    output = {
        "definition": {
            "source_update": "four sequential restored-AdamW updates; each averages one disjoint 250-timestamp block",
            "target": "finite predicted-noise change at each fixed target endpoint, direction, timestep, and prompt",
            "precision": PIJVP_PRECISION_MODE,
        },
        "replayed_final_vs_saved_final": {
            "cosine": summary(replay_cosines),
            "relative_l2_error": summary(replay_errors),
        },
        "methods": {},
    }
    for method in SEQ4_METHODS:
        result = {
            name: summary(metric_values)
            for name, metric_values in values[method].items()
        }
        result["branch_mean_magnitude_spearman"] = safe_spearman(
            branch_predicted[method], branch_actual
        )
        output["methods"][method] = result
    SEQ4_ROOT.mkdir(parents=True, exist_ok=True)
    with open(SEQ4_SUMMARY_PATH, "w") as handle:
        json.dump(output, handle, indent=2)

    replay = output["replayed_final_vs_saved_final"]
    print("\nsequential four-update replay")
    print(
        f"final parameter cosine={replay['cosine']['mean']:+.8f} "
        f"relative-error={replay['relative_l2_error']['mean']:.6e}"
    )
    print("\nfixed-point predicted-noise response")
    print(
        "method                                  pooled-rho  mag-ratio  "
        "mag-relerr  vector-relerr  vector-cos  branch-rho"
    )
    for method, result in output["methods"].items():
        print(
            f"{method:39s} "
            f"{result['pooled_branch_spearman']['mean']:+.4f}     "
            f"{result['magnitude_ratio']['median']:.6f}   "
            f"{result['magnitude_relative_error']['median']:.6e}  "
            f"{result['vector_relative_error']['median']:.6e}    "
            f"{result['vector_cosine']['mean']:+.6f}   "
            f"{result['branch_mean_magnitude_spearman']:+.4f}"
        )
    print(f"\n[saved] {SEQ4_SUMMARY_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--batch-size", type=int, default=SEQ4_DEFAULT_BATCH_SIZE)
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
