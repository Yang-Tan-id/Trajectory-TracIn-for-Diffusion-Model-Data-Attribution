#!/usr/bin/env python3
"""Launch the x3 Top-K simple-loss 100-timestamp x MC4 recomputation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from exp_config import LOG_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--batch-size", type=int, default=200)
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("no GPUs provided")

    log_dir = LOG_DIR / "topk_simple_loss_100t_mc4"
    log_dir.mkdir(parents=True, exist_ok=True)
    workers = []
    streams = []
    for shard_index, gpu in enumerate(gpus):
        log_path = log_dir / f"shard_{shard_index:02d}.log"
        stream = log_path.open("w", buffering=1)
        command = [
            sys.executable,
            "-u",
            "302_recompute_topk_simple_loss_100t_mc4_shard.py",
            "--gpu",
            str(gpu),
            "--query-shard-index",
            str(shard_index),
            "--query-shard-count",
            str(len(gpus)),
            "--batch-size",
            str(args.batch_size),
        ]
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        workers.append((process, log_path))
        streams.append(stream)
        print(
            f"[launcher] shard={shard_index} gpu={gpu} pid={process.pid} "
            f"log={log_path}",
            flush=True,
        )

    failed = []
    for (process, log_path), stream in zip(workers, streams):
        code = process.wait()
        stream.close()
        print(f"[launcher] {log_path.name} code={code}", flush=True)
        if code:
            failed.append((str(log_path), code))
    if failed:
        raise SystemExit(f"workers failed: {failed}")
    subprocess.run(
        [sys.executable, "-u", "301_export_topk_removal_results_60q.py"],
        check=True,
    )
    print("[done] simple loss updated to 100 timestamps x MC4", flush=True)


if __name__ == "__main__":
    main()
