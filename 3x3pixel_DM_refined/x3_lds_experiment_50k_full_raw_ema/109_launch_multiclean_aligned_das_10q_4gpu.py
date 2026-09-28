"""Build ten clean estimates, run aligned DAS on four GPUs, and merge."""

import argparse
import subprocess
import sys
import time

from multiclean_das_config import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=DAS_FEATURE_BATCH_SIZE)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if len(CUDA_IDS) < 4:
        raise ValueError("four CUDA_IDS are required")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "multiclean_aligned_das_10q_4gpu.log"
    active = {}
    with open(log_path, "a", buffering=1) as stream:
        subprocess.run(
            [sys.executable, "-u", "106_build_predicted_clean_10anchors.py"],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
        stream.write(
            "\n[launcher] q00-q09 predicted-clean 10-anchor aligned DAS; "
            "100 DAS timestamps, MC10, all lambdas, train pass reused\n"
        )
        for shard_index, gpu in enumerate(CUDA_IDS[:4]):
            label = f"timestamp-shard-{shard_index}"
            command = [
                sys.executable,
                "-u",
                "107_run_multiclean_aligned_das_shard.py",
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
                "108_merge_multiclean_aligned_das.py",
                "--timestamp-shard-count",
                "4",
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print("[done] predicted-clean 10-anchor aligned DAS q00-q09", flush=True)


if __name__ == "__main__":
    main()
