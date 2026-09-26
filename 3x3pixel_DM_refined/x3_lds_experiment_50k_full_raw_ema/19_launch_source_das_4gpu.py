"""Run timestamp-aligned SOURCE-DAS on four GPUs, then merge and evaluate LDS."""

import argparse
import subprocess
import sys
import time

from exp_config import CUDA_IDS, LOG_DIR
from source_das_config import SOURCE_DAS_METHOD, SOURCE_DAS_TRAIN_SCORE_BATCH_SIZE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-batch-size", type=int, default=SOURCE_DAS_TRAIN_SCORE_BATCH_SIZE
    )
    args = parser.parse_args()
    if len(CUDA_IDS) < 4:
        raise ValueError("SOURCE-DAS launcher requires four GPUs")
    if args.train_batch_size <= 0:
        raise ValueError("--train-batch-size must be positive")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "source_das_10ckpt_100q_100t_mc10_4gpu.log"
    assignments = (
        ("prompted", 0, 2, CUDA_IDS[0]),
        ("prompted", 1, 2, CUDA_IDS[1]),
        ("unprompted", 0, 2, CUDA_IDS[2]),
        ("unprompted", 1, 2, CUDA_IDS[3]),
    )
    with open(log_path, "a", buffering=1) as stream:
        stream.write("\n[launcher] timestamp-aligned SOURCE-DAS start\n")
        active = {}
        for family, shard, count, gpu in assignments:
            label = f"source-das-{family}-{shard}"
            command = [
                sys.executable,
                "-u",
                "18_run_source_das_timestamp_shard.py",
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
            process = subprocess.Popen(
                command, stdout=stream, stderr=subprocess.STDOUT
            )
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
    for family in ("prompted", "unprompted"):
        subprocess.run(
            [
                sys.executable,
                "18_merge_source_das_shards.py",
                "--family",
                family,
                "--timestamp-shard-count",
                "2",
            ],
            check=True,
        )
    subprocess.run(
        [
            sys.executable,
            "15_eval_last_noise_delta_direction_lds.py",
            "--method",
            SOURCE_DAS_METHOD,
        ],
        check=True,
    )
    print(f"[done] SOURCE-DAS method={SOURCE_DAS_METHOD}", flush=True)


if __name__ == "__main__":
    main()
