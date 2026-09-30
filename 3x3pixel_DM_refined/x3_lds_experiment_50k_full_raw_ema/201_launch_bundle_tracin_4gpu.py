"""Run 10-query Bundle TracIn over four checkpoint shards and evaluate LDS."""

import argparse
import json
import subprocess
import sys
import time

from bundle_tracin_config import *


def run_workers(commands, log_path, label):
    with open(log_path, "a", buffering=1) as stream:
        stream.write(f"\n[launcher] {label}\n")
        workers = []
        for worker_label, command in commands:
            stream.write(f"[launcher] {worker_label}: {' '.join(command)}\n")
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            workers.append((worker_label, process))
            print(f"[launcher] {worker_label} pid={process.pid}", flush=True)
        print(f"[launcher] log={log_path}", flush=True)
        active = dict(workers)
        while active:
            for worker_label, process in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                del active[worker_label]
                print(f"[launcher] {worker_label} code={code}", flush=True)
                if code != 0:
                    for other in active.values():
                        other.terminate()
                    for other in active.values():
                        other.wait()
                    raise SystemExit(code)
            if active:
                time.sleep(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--query-ids", default="0-9")
    parser.add_argument("--batch-size", type=int, default=BUNDLE_TRACIN_BATCH_SIZE)
    args = parser.parse_args()
    gpus = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU")
    query_ids = parse_query_ids(args.query_ids)
    query_arg = ",".join(str(value) for value in query_ids)
    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    by_id = {int(record["query_id"]): record for record in manifest}
    families = []
    for query_id in query_ids:
        family = by_id[query_id]["family"]
        if family not in families:
            families.append(family)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / (
        f"bundle_tracin_q{query_ids[0]:02d}_q{query_ids[-1]:02d}_4gpu.log"
    )
    shard_count = len(gpus)
    # Families run sequentially; all available GPUs shard that family's 50
    # checkpoints. This avoids recomputing the same checkpoint's 50k training
    # gradients on multiple GPUs merely because queries have different prompts.
    for family in families:
        commands = []
        for shard_index, gpu in enumerate(gpus):
            commands.append(
                (
                    f"{family}-checkpoint-shard-{shard_index}",
                    [
                        sys.executable,
                        "-u",
                        "199_run_bundle_tracin_checkpoint_shard.py",
                        "--family",
                        family,
                        "--gpu",
                        str(gpu),
                        "--checkpoint-shard-index",
                        str(shard_index),
                        "--checkpoint-shard-count",
                        str(shard_count),
                        "--query-ids",
                        query_arg,
                        "--batch-size",
                        str(args.batch_size),
                    ],
                )
            )
        run_workers(commands, log_path, f"Bundle TracIn family={family}")

    subprocess.run(
        [
            sys.executable,
            "-u",
            "200_merge_eval_bundle_tracin_lds.py",
            "--query-ids",
            query_arg,
            "--checkpoint-shard-count",
            str(shard_count),
        ],
        check=True,
    )
    print("[done] Bundle TracIn vectors merged and LDS evaluated", flush=True)


if __name__ == "__main__":
    main()
