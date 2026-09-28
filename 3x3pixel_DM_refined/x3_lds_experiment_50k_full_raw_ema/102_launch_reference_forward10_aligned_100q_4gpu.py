"""Launch all-100-query reference-forward-10 aligned-loss TracIn on four GPUs."""

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
    log_path = LOG_DIR / "reference_forward10_prednoise_aligned_loss_100q_4gpu.log"
    assignments = (
        ("prompted", 0, CUDA_IDS[0]),
        ("prompted", 1, CUDA_IDS[1]),
        ("unprompted", 0, CUDA_IDS[2]),
        ("unprompted", 1, CUDA_IDS[3]),
    )
    active = {}
    with open(log_path, "a", buffering=1) as stream:
        stream.write(
            "\n[launcher] q00-q99 reference forward-10 predicted-noise versus "
            "aligned train-loss TracIn; "
            "50 checkpoints, 100 timestamps, CountSketch4096\n"
        )
        for family, shard, gpu in assignments:
            label = f"{family}-timestamp-shard-{shard}"
            command = [
                sys.executable,
                "-u",
                "100_run_reference_forward10_aligned_shard.py",
                "--gpu",
                str(gpu),
                "--family",
                family,
                "--timestamp-shard-index",
                str(shard),
                "--timestamp-shard-count",
                "2",
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
        for family in FAMILIES:
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "101_merge_reference_forward10_aligned.py",
                    "--family",
                    family,
                    "--timestamp-shard-count",
                    "2",
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
    print(
        "[done] merged reference-forward-10 prednoise/aligned-loss q00-q99",
        flush=True,
    )


if __name__ == "__main__":
    main()
