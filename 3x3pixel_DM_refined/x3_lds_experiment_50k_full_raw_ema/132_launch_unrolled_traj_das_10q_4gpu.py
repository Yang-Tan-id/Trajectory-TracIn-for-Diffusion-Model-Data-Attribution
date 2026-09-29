"""Cache full-unroll query features and run trajectory DAS on four GPUs."""

import argparse
import subprocess
import sys
import time

from unrolled_traj_das_config import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default=",".join(str(value) for value in CUDA_IDS[:4]))
    parser.add_argument("--batch-size", type=int, default=DAS_FEATURE_BATCH_SIZE)
    args = parser.parse_args()
    gpus = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    if len(gpus) != 4 or len(set(gpus)) != 4:
        raise ValueError("--gpus must contain exactly four distinct GPU ids")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "unrolled_trajectory_das_probe4_10q_4gpu.log"
    active = {}
    print(
        f"[launcher] caching fully-unrolled q00-q09 query features on "
        f"gpu={gpus[0]} log={log_path}",
        flush=True,
    )
    with open(log_path, "a", buffering=1) as stream:
        subprocess.run(
            [
                sys.executable,
                "-u",
                "129_cache_unrolled_traj_das_query_features.py",
                "--gpu",
                str(gpus[0]),
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
        stream.write(
            "\n[launcher] fully-unrolled trajectory DAS q00-q09; final EMA, "
            "probe4, projected4096, DAS 100x10, train-gradient MC10\n"
        )
        for shard_index, gpu in enumerate(gpus):
            label = f"term-shard-{shard_index}"
            command = [
                sys.executable,
                "-u",
                "130_run_unrolled_traj_das_shard.py",
                "--gpu",
                str(gpu),
                "--shard-index",
                str(shard_index),
                "--shard-count",
                "4",
                "--batch-size",
                str(args.batch_size),
            ]
            stream.write(f"[launcher] {label}: {' '.join(command)}\n")
            process = subprocess.Popen(
                command, stdout=stream, stderr=subprocess.STDOUT
            )
            active[label] = process
            print(f"[launcher] started {label} pid={process.pid}", flush=True)
        print(f"[launcher] log={log_path}", flush=True)
        while active:
            for label, process in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                del active[label]
                print(f"[launcher] {label} exited code={code}", flush=True)
                if code:
                    for other in active.values():
                        other.terminate()
                    for other in active.values():
                        other.wait()
                    raise SystemExit(code)
            if active:
                time.sleep(2)
        subprocess.run(
            [
                sys.executable,
                "-u",
                "131_merge_unrolled_traj_das_shards.py",
                "--shard-count",
                "4",
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print("[done] fully-unrolled trajectory DAS q00-q09", flush=True)


if __name__ == "__main__":
    main()
