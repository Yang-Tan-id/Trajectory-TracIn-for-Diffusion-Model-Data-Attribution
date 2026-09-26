"""Run Adam/clipping-aware raw SOURCE-DAS on four GPUs, merge, and evaluate."""

import argparse
import subprocess
import sys
import time

from adam_clip_source_das_config import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-batch-size", type=int, default=ADAM_CLIP_SOURCE_TRAIN_BATCH_SIZE
    )
    args = parser.parse_args()
    if len(CUDA_IDS) < 4:
        raise ValueError("Adam/clipping SOURCE launcher requires four GPUs")
    if args.train_batch_size <= 0:
        raise ValueError("--train-batch-size must be positive")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "source_das_adam_clip_raw_11h50p_100q_100t_mc10_4gpu.log"
    assignments = (
        ("prompted", 0, 2, CUDA_IDS[0]),
        ("prompted", 1, 2, CUDA_IDS[1]),
        ("unprompted", 0, 2, CUDA_IDS[2]),
        ("unprompted", 1, 2, CUDA_IDS[3]),
    )
    with open(log_path, "a", buffering=1) as stream:
        stream.write("\n[launcher] Adam/clipping-aware raw SOURCE-DAS start\n")
        active = {}
        for family, shard, count, gpu in assignments:
            label = f"adam-clip-source-{family}-{shard}"
            command = [
                sys.executable,
                "-u",
                "21_run_adam_clip_source_das_timestamp_shard.py",
                "--family",
                family,
                "--gpu",
                str(gpu),
                "--timestamp-shard-index",
                str(shard),
                "--timestamp-shard-count",
                str(count),
                "--train-batch-size",
                str(args.train_batch_size),
            ]
            stream.write(f"[launcher] {label}: {' '.join(command)}\n")
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            active[label] = process
            print(f"[launcher] started {label} pid={process.pid}", flush=True)
        print(f"[launcher] detailed log: {log_path}", flush=True)
        while active:
            for label, process in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                del active[label]
                print(f"[launcher] {label} exited code={code}", flush=True)
                if code != 0:
                    for other in active.values():
                        other.terminate()
                    for other in active.values():
                        other.wait()
                    raise SystemExit(code)
            if active:
                time.sleep(1)

    for family in FAMILIES:
        subprocess.run(
            [
                sys.executable,
                "21_merge_adam_clip_source_das_shards.py",
                "--family",
                family,
                "--timestamp-shard-count",
                "2",
            ],
            check=True,
        )
    for method in ADAM_CLIP_SOURCE_METHODS.values():
        subprocess.run(
            [
                sys.executable,
                "15_eval_last_noise_delta_direction_lds.py",
                "--method",
                method,
            ],
            check=True,
        )
    print(
        f"[done] Adam/clipping SOURCE methods={tuple(ADAM_CLIP_SOURCE_METHODS.values())}",
        flush=True,
    )


if __name__ == "__main__":
    main()
