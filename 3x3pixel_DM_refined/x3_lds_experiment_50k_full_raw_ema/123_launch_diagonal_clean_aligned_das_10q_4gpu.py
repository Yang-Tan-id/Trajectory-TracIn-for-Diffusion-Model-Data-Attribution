"""Build 100-query clean estimates and run diagonal DAS on four GPUs."""

import argparse
import subprocess
import sys
import time

from diagonal_clean_das_config import CUDA_IDS, LOG_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpus", default=",".join(str(value) for value in CUDA_IDS[:4]))
    parser.add_argument("--timestamp-count", type=int, choices=(10, 100), default=100)
    args = parser.parse_args()
    gpus = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if len(gpus) != 4 or len(set(gpus)) != 4:
        raise ValueError("--gpus must contain exactly four distinct GPU ids")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / (
        f"diagonal_clean_aligned_das_100q_{args.timestamp_count}timestamps_4gpu.log"
    )
    active = {}
    with open(log_path, "a", buffering=1) as stream:
        subprocess.run(
            [
                sys.executable,
                "-u",
                "120_build_predicted_clean_100timestamps.py",
                "--gpu",
                str(gpus[0]),
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
        stream.write(
            f"\n[launcher] q00-q99 predicted-clean {args.timestamp_count}-timestamp "
            "diagonal aligned DAS; MC10; all lambdas\n"
        )
        stream.write(
            f"[launcher] gpus={gpus} feature_batch_size={args.batch_size}\n"
        )
        assignments = (
            ("prompted", 0, gpus[0]),
            ("prompted", 1, gpus[1]),
            ("unprompted", 0, gpus[2]),
            ("unprompted", 1, gpus[3]),
        )
        for family, shard_index, gpu in assignments:
            label = f"{family}-timestamp-shard-{shard_index}"
            command = [
                sys.executable,
                "-u",
                "121_run_diagonal_clean_aligned_das_shard.py",
                "--family",
                family,
                "--gpu",
                str(gpu),
                "--timestamp-shard-index",
                str(shard_index),
                "--timestamp-shard-count",
                "2",
                "--timestamp-count",
                str(args.timestamp_count),
                "--batch-size",
                str(args.batch_size),
            ]
            stream.write(f"[launcher] {label}: {' '.join(command)}\n")
            process = subprocess.Popen(
                command, stdout=stream, stderr=subprocess.STDOUT
            )
            active[label] = process
            print(f"[launcher] started {label} gpu={gpu} pid={process.pid}", flush=True)
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
                "122_merge_diagonal_clean_aligned_das.py",
                "--timestamp-shard-count",
                "2",
                "--timestamp-count",
                str(args.timestamp_count),
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print(
        f"[done] {args.timestamp_count}-timestamp diagonal predicted-clean DAS q00-q99",
        flush=True,
    )


if __name__ == "__main__":
    main()
