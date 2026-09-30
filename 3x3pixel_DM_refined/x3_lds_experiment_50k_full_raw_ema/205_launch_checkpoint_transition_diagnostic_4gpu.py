"""Launch and summarize the checkpoint-transition diagnostic."""

import argparse
import json
import subprocess
import sys
import time

import numpy as np

from checkpoint_transition_diagnostic_config import *


def launch(commands, log_path, label):
    with open(log_path, "a", buffering=1) as stream:
        stream.write(f"\n[launcher] {label}\n")
        workers = []
        for name, command in commands:
            stream.write(f"[launcher] {name}: {' '.join(command)}\n")
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            workers.append((name, process))
            print(f"[launcher] {name} pid={process.pid}", flush=True)
        active = dict(workers)
        while active:
            for name, process in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                del active[name]
                print(f"[launcher] {name} code={code}", flush=True)
                if code != 0:
                    for other in active.values():
                        other.terminate()
                    for other in active.values():
                        other.wait()
                    raise SystemExit(code)
            if active:
                time.sleep(1)


def analyze(pair_indices, query_ids):
    records = []
    parameter_records = []
    for family in FAMILIES:
        for pair_index in pair_indices:
            path = CTD_ROOT / family / f"pair_{pair_index:02d}.npz"
            metadata_path = CTD_ROOT / family / f"pair_{pair_index:02d}.json"
            if not path.is_file():
                continue
            with np.load(path, allow_pickle=False) as payload:
                selected = np.isin(payload["query_ids"], query_ids)
                if not np.any(selected):
                    continue
                item = {"actual_l2": payload["actual_l2"][selected]}
                for method in CTD_METHODS:
                    for metric in (
                        "predicted_l2",
                        "vector_cosine",
                        "vector_relative_error",
                        "magnitude_relative_error",
                    ):
                        item[f"{method}_{metric}"] = payload[
                            f"{method}_{metric}"
                        ][selected]
                records.append(item)
            with open(metadata_path) as handle:
                metadata = json.load(handle)
            parameter_records.append(metadata["replayed_adamw_parameter_agreement"])
    if not records:
        raise FileNotFoundError("no completed diagnostic pair outputs found")

    summary = {
        "pair_indices": pair_indices,
        "query_ids": query_ids,
        "completed_family_pairs": len(records),
        "methods": {},
        "replayed_adamw_parameter_agreement": {
            "cosine_mean": float(np.mean([x["cosine"] for x in parameter_records])),
            "relative_error_mean": float(
                np.mean([x["relative_error"] for x in parameter_records])
            ),
            "relative_error_max": float(
                np.max([x["relative_error"] for x in parameter_records])
            ),
        },
    }
    print("\ncheckpoint-transition predicted-noise response", flush=True)
    print(
        "method                                  vector-cos  vector-relerr  "
        "mag-relerr  mag-ratio",
        flush=True,
    )
    actual = np.concatenate([record["actual_l2"] for record in records])
    for method in CTD_METHODS:
        cosine = np.concatenate(
            [record[f"{method}_vector_cosine"] for record in records]
        )
        vector_error = np.concatenate(
            [record[f"{method}_vector_relative_error"] for record in records]
        )
        magnitude_error = np.concatenate(
            [record[f"{method}_magnitude_relative_error"] for record in records]
        )
        predicted = np.concatenate(
            [record[f"{method}_predicted_l2"] for record in records]
        )
        values = {
            "vector_cosine_mean": float(np.nanmean(cosine)),
            "vector_cosine_std": float(np.nanstd(cosine)),
            "vector_relative_error_mean": float(np.nanmean(vector_error)),
            "magnitude_relative_error_mean": float(np.nanmean(magnitude_error)),
            "magnitude_ratio": float(predicted.sum() / np.maximum(actual.sum(), 1e-30)),
        }
        summary["methods"][method] = values
        print(
            f"{method:40s} {values['vector_cosine_mean']:+.6f}  "
            f"{values['vector_relative_error_mean']:.6f}       "
            f"{values['magnitude_relative_error_mean']:.6f}   "
            f"{values['magnitude_ratio']:.6f}",
            flush=True,
        )
    agreement = summary["replayed_adamw_parameter_agreement"]
    print(
        "replayed AdamW parameter delta: "
        f"cos={agreement['cosine_mean']:+.8f} "
        f"mean-relerr={agreement['relative_error_mean']:.3e} "
        f"max-relerr={agreement['relative_error_max']:.3e}",
        flush=True,
    )
    CTD_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = CTD_ROOT / "summary.json"
    with open(output_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[saved] {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--pair-indices", default="0-48")
    parser.add_argument("--query-ids", default="0-9")
    parser.add_argument("--query-batch-size", type=int, default=CTD_QUERY_BATCH_SIZE)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    pair_indices = parse_integer_selection(args.pair_indices, CTD_PAIR_INDICES)
    query_ids = parse_integer_selection(args.query_ids, CTD_QUERY_IDS)
    pair_arg = ",".join(str(value) for value in pair_indices)
    query_arg = ",".join(str(value) for value in query_ids)
    if not args.analyze_only:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / "checkpoint_transition_diagnostic_4gpu.log"
        for family in FAMILIES:
            commands = []
            for shard_index, gpu in enumerate(gpus):
                commands.append(
                    (
                        f"{family}-pair-shard-{shard_index}",
                        [
                            sys.executable,
                            "-u",
                            "204_run_checkpoint_transition_diagnostic_shard.py",
                            "--family",
                            family,
                            "--gpu",
                            str(gpu),
                            "--pair-shard-index",
                            str(shard_index),
                            "--pair-shard-count",
                            str(len(gpus)),
                            "--pair-indices",
                            pair_arg,
                            "--query-ids",
                            query_arg,
                            "--query-batch-size",
                            str(args.query_batch_size),
                        ],
                    )
                )
            launch(commands, log_path, f"checkpoint transitions family={family}")
    analyze(pair_indices, query_ids)


if __name__ == "__main__":
    main()
