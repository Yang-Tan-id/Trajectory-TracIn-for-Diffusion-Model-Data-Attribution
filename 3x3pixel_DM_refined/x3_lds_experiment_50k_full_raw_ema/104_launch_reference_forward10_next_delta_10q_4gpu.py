"""Launch q00-q09 reference-forward-10 next-delta TracIn on four GPUs."""

import argparse
import subprocess
import sys
import time

from reference_forward10_config import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=REF_FORWARD10_BATCH_SIZE)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if len(CUDA_IDS) < 4:
        raise ValueError("four CUDA_IDS are required")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "reference_forward10_next_delta_aligned_loss_10q_4gpu.log"
    active = {}
    with open(log_path, "a", buffering=1) as stream:
        stream.write(
            "\n[launcher] q00-q09 reference forward-10 next-delta/aligned-loss; "
            "50 checkpoints, 49 pairs, 100 timestamps, CountSketch4096\n"
        )
        for shard_index, gpu in enumerate(CUDA_IDS[:4]):
            label = f"prompted-timestamp-shard-{shard_index}"
            command = [
                sys.executable,
                "-u",
                "100_run_reference_forward10_aligned_shard.py",
                "--gpu",
                str(gpu),
                "--family",
                "prompted",
                "--query-scope",
                "ten",
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
                "101_merge_reference_forward10_aligned.py",
                "--family",
                "prompted",
                "--query-scope",
                "ten",
                "--timestamp-shard-count",
                "4",
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print("[done] merged reference-forward-10 next-delta q00-q09", flush=True)


if __name__ == "__main__":
    main()
