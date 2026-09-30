"""Launch selected two-checkpoint frozen-gradient AdamW diagnostics."""

import argparse
import json
import subprocess
import sys
import time

import numpy as np

from checkpoint_endpoint_adam_diagnostic_config import *


def summarize(pair_indices, family, loss_mc):
    output_root = cead_root(loss_mc)
    records = []
    parameter = {method: [] for method in CEAD_METHODS if method != CEAD_METHODS[0]}
    for pair_index in pair_indices:
        path = output_root / family / f"pair_{pair_index:02d}.npz"
        metadata_path = output_root / family / f"pair_{pair_index:02d}.json"
        with np.load(path, allow_pickle=False) as payload:
            records.append({name: payload[name] for name in payload.files})
        with open(metadata_path) as handle:
            metadata = json.load(handle)
        for method, values in metadata["parameter_agreement"].items():
            parameter[method].append(values)

    actual = np.concatenate([record["actual_l2"] for record in records])
    output = {
        "family": family,
        "pair_indices": pair_indices,
        "loss_mc": loss_mc,
        "methods": {},
    }
    print("\ntwo-checkpoint frozen-gradient AdamW response", flush=True)
    print(
        "method                                      vector-cos  vector-relerr  "
        "mag-relerr  mag-ratio",
        flush=True,
    )
    for method in CEAD_METHODS:
        cosine = np.concatenate([record[f"{method}_vector_cosine"] for record in records])
        vector_error = np.concatenate(
            [record[f"{method}_vector_relative_error"] for record in records]
        )
        magnitude_error = np.concatenate(
            [record[f"{method}_magnitude_relative_error"] for record in records]
        )
        predicted = np.concatenate([record[f"{method}_predicted_l2"] for record in records])
        values = {
            "vector_cosine_mean": float(np.nanmean(cosine)),
            "vector_relative_error_mean": float(np.nanmean(vector_error)),
            "magnitude_relative_error_mean": float(np.nanmean(magnitude_error)),
            "magnitude_ratio": float(predicted.sum() / np.maximum(actual.sum(), 1e-30)),
        }
        output["methods"][method] = values
        print(
            f"{method:44s} {values['vector_cosine_mean']:+.6f}  "
            f"{values['vector_relative_error_mean']:.6f}       "
            f"{values['magnitude_relative_error_mean']:.6f}   "
            f"{values['magnitude_ratio']:.6f}",
            flush=True,
        )
    output["parameter_agreement"] = {
        method: {
            "cosine_mean": float(np.mean([item["cosine"] for item in items])),
            "relative_error_mean": float(
                np.mean([item["relative_error"] for item in items])
            ),
        }
        for method, items in parameter.items()
    }
    output_path = output_root / f"summary_{family}.json"
    with open(output_path, "w") as handle:
        json.dump(output, handle, indent=2)
    print(f"[saved] {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--pair-indices", default="0,16,32,48")
    parser.add_argument("--family", choices=FAMILIES, default="prompted")
    parser.add_argument("--query-ids", default="0-9")
    parser.add_argument("--query-batch-size", type=int, default=CTD_QUERY_BATCH_SIZE)
    parser.add_argument("--loss-mc", type=int, default=1)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    pair_indices = parse_integer_selection(args.pair_indices, (0, 16, 32, 48))
    if args.loss_mc <= 0:
        raise ValueError("--loss-mc must be positive")
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if len(pair_indices) > len(gpus):
        raise ValueError("this launcher assigns at most one selected pair per GPU")

    if not args.analyze_only:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / "checkpoint_endpoint_adam_diagnostic_4gpu.log"
        with open(log_path, "a", buffering=1) as stream:
            workers = []
            for pair_index, gpu in zip(pair_indices, gpus):
                command = [
                    sys.executable,
                    "-u",
                    "206_run_checkpoint_endpoint_adam_diagnostic.py",
                    "--family",
                    args.family,
                    "--gpu",
                    str(gpu),
                    "--pair-index",
                    str(pair_index),
                    "--query-ids",
                    args.query_ids,
                    "--query-batch-size",
                    str(args.query_batch_size),
                    "--loss-mc",
                    str(args.loss_mc),
                ]
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                workers.append((pair_index, process))
                print(f"[launcher] pair={pair_index:02d} gpu={gpu} pid={process.pid}", flush=True)
            print(f"[launcher] log={log_path}", flush=True)
            active = dict(workers)
            while active:
                for pair_index, process in list(active.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    del active[pair_index]
                    print(f"[launcher] pair={pair_index:02d} code={code}", flush=True)
                    if code != 0:
                        for other in active.values():
                            other.terminate()
                        for other in active.values():
                            other.wait()
                        raise SystemExit(code)
                if active:
                    time.sleep(1)
    summarize(pair_indices, args.family, args.loss_mc)


if __name__ == "__main__":
    main()
