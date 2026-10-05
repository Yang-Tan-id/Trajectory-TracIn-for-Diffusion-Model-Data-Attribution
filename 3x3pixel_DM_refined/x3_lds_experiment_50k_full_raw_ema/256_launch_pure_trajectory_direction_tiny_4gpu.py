"""Launch q00-q03 tiny pure trajectory-direction experiment on four GPUs."""

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
        raise ValueError("provide exactly four GPUs for q00-q03")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    processes = []
    handles = []
    for query_id, gpu in enumerate(gpus):
        log = LOG_DIR / f"pure_trajectory_direction_tiny_q{query_id:02d}.log"
        handle = open(log, "w")
        command = [
            sys.executable, "-u",
            "254_run_pure_trajectory_direction_small_worker.py",
            "--gpu", str(gpu),
            "--query-id", str(query_id),
            "--batch-size", str(args.batch_size),
            "--grad-microbatch-size", str(args.grad_microbatch_size),
            "--query-term-batch-size", str(args.query_term_batch_size),
        ]
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
        processes.append((process, log))
        handles.append(handle)
        print(
            f"[launcher] q{query_id:02d} gpu={gpu} pid={process.pid} log={log}",
            flush=True,
        )
    failed = False
    for process, log in processes:
        code = process.wait()
        failed |= code != 0
        print(f"[launcher] {log.name} code={code}", flush=True)
    for handle in handles:
        handle.close()
    if failed:
        raise SystemExit("one or more tiny workers failed")
    subprocess.run(
        [sys.executable, "-u", "255_merge_eval_pure_trajectory_direction_tiny.py"],
        check=True,
    )


if __name__ == "__main__":
    main()
