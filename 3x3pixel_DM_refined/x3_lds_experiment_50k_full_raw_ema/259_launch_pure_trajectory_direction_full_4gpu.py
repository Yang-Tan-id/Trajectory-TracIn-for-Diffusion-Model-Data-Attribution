"""Launch q00-q09 full pure trajectory-direction TracIn-DAS."""

import argparse
import subprocess
import sys

from exp_config import LOG_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--grad-microbatch-size", type=int, default=4)
    ap.add_argument("--query-term-batch-size", type=int, default=32)
    args = ap.parse_args()
    gpus = [int(value) for value in args.gpus.split(",")]
    if len(gpus) != 4:
        raise ValueError("provide exactly four GPUs")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    processes, handles = [], []
    for shard_index, gpu in enumerate(gpus):
        log = LOG_DIR / f"pure_trajectory_direction_full_shard_{shard_index:02d}.log"
        handle = open(log, "w")
        command = [
            sys.executable, "-u",
            "257_run_pure_trajectory_direction_full_worker.py",
            "--gpu", str(gpu),
            "--query-shard-index", str(shard_index),
            "--query-shard-count", str(len(gpus)),
            "--batch-size", str(args.batch_size),
            "--grad-microbatch-size", str(args.grad_microbatch_size),
            "--query-term-batch-size", str(args.query_term_batch_size),
        ]
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
        processes.append((process, log))
        handles.append(handle)
        print(f"[launcher] shard={shard_index} gpu={gpu} pid={process.pid} log={log}", flush=True)
    failed = False
    for process, log in processes:
        code = process.wait()
        failed |= code != 0
        print(f"[launcher] {log.name} code={code}", flush=True)
    for handle in handles:
        handle.close()
    if failed:
        raise SystemExit("one or more full pure-direction workers failed")
    subprocess.run(
        [sys.executable, "-u", "258_merge_eval_pure_trajectory_direction_full.py",
         "--query-shard-count", str(len(gpus))],
        check=True,
    )


if __name__ == "__main__":
    main()
