"""Run trajectory-aligned then trajectory-independent controlled tests."""

import argparse
import subprocess
import sys

from exp_config import LOG_DIR


def run_mode(mode, gpus, args):
    processes, handles = [], []
    print(f"[launcher] starting trajectory mode={mode}", flush=True)
    for shard, gpu in enumerate(gpus):
        log = LOG_DIR / f"trajectory_pairing_{mode}_shard_{shard:02d}.log"
        handle = open(log, "w")
        command = [
            sys.executable, "-u", "260_run_trajectory_pairing_small_worker.py",
            "--gpu", str(gpu), "--mode", mode,
            "--timestamp-shard-index", str(shard),
            "--timestamp-shard-count", str(len(gpus)),
            "--batch-size", str(args.batch_size),
            "--grad-microbatch-size", str(args.grad_microbatch_size),
            "--query-term-batch-size", str(args.query_term_batch_size),
        ]
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
        processes.append((process, log))
        handles.append(handle)
        print(f"[launcher] mode={mode} shard={shard} gpu={gpu} pid={process.pid}", flush=True)
    failed = False
    for process, log in processes:
        code = process.wait()
        failed |= code != 0
        print(f"[launcher] {log.name} code={code}", flush=True)
    for handle in handles:
        handle.close()
    if failed:
        raise SystemExit(f"trajectory mode {mode} failed")


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
    for mode in ("aligned", "independent"):
        run_mode(mode, gpus, args)
    subprocess.run(
        [sys.executable, "-u", "261_merge_eval_trajectory_pairing_small.py",
         "--timestamp-shard-count", str(len(gpus))],
        check=True,
    )


if __name__ == "__main__":
    main()
