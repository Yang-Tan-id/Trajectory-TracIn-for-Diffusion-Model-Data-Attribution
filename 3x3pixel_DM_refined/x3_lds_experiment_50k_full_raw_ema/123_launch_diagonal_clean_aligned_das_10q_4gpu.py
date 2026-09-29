"""Build 100 clean estimates and run timestamp-diagonal DAS on four GPUs."""

import argparse
import subprocess
import sys
import time

from diagonal_clean_das_config import CUDA_IDS, LOG_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpus", default=",".join(str(value) for value in CUDA_IDS[:4]))
    args = parser.parse_args()
    gpus = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if len(gpus) != 4 or len(set(gpus)) != 4:
        raise ValueError("--gpus must contain exactly four distinct GPU ids")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "diagonal_clean_aligned_das_10q_4gpu.log"
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
            "\n[launcher] q00-q09 predicted-clean 100-timestamp diagonal aligned "
            "DAS; MC10; all lambdas\n"
        )
        stream.write(
            f"[launcher] gpus={gpus} feature_batch_size={args.batch_size}\n"
        )
        for shard_index, gpu in enumerate(gpus):
            label = f"timestamp-shard-{shard_index}"
            command = [
                sys.executable,
                "-u",
                "121_run_diagonal_clean_aligned_das_shard.py",
                "--gpu",
                str(gpu),
                "--timestamp-shard-index",
                str(shard_index),
                "--timestamp-shard-count",
                "4",
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
                "4",
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print("[done] timestamp-diagonal predicted-clean DAS q00-q09", flush=True)


if __name__ == "__main__":
    main()
