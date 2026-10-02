"""Launch q00-q09 direction-integrated TracIn-DAS on four GPUs."""

import argparse
import subprocess
import sys
import time

from direction_integrated_tracin_das_config import *


def parse_gpus(text):
    values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    if len(values) != 4 or len(set(values)) != 4:
        raise ValueError("--gpus must contain four distinct CUDA IDs")
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-t-chunk-size", type=int, default=50)
    parser.add_argument(
        "--train-t-count", type=int, default=DITD_DEFAULT_TRAIN_T_COUNT
    )
    parser.add_argument("--query-term-batch-size", type=int, default=100)
    parser.add_argument("--direction-count", type=int, default=DITD_DIRECTION_COUNT)
    args = parser.parse_args()
    gpus = parse_gpus(args.gpus)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / (
        f"direction_integrated_tracin_das_{args.train_t_count}traint_"
        "100dir_q00_q09_4gpu.log"
    )
    active = {}
    with open(log_path, "a", buffering=1) as stream:
        stream.write(
            f"\n[launcher] direction-aligned {args.train_t_count}-t train loss; "
            "100 endpoint timestamps; next delta; projected4096; q00-q09\n"
        )
        for shard_index, gpu in enumerate(gpus):
            command = [
                sys.executable,
                "-u",
                "219_run_direction_integrated_tracin_das_shard.py",
                "--gpu",
                str(gpu),
                "--direction-shard-index",
                str(shard_index),
                "--direction-shard-count",
                "4",
                "--batch-size",
                str(args.batch_size),
                "--train-t-chunk-size",
                str(args.train_t_chunk_size),
                "--train-t-count",
                str(args.train_t_count),
                "--query-term-batch-size",
                str(args.query_term_batch_size),
                "--direction-count",
                str(args.direction_count),
                "--train-t-count",
                str(args.train_t_count),
            ]
            label = f"direction-shard-{shard_index}"
            stream.write(f"[launcher] {label}: {' '.join(command)}\n")
            process = subprocess.Popen(
                command, stdout=stream, stderr=subprocess.STDOUT
            )
            active[label] = process
            print(f"[launcher] {label} gpu={gpu} pid={process.pid}", flush=True)
        print(f"[launcher] log={log_path}", flush=True)
        while active:
            for label, process in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                del active[label]
                print(f"[launcher] {label} code={code}", flush=True)
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
                "220_merge_eval_direction_integrated_tracin_das.py",
                "--direction-shard-count",
                "4",
                "--direction-count",
                str(args.direction_count),
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print("[done] direction-integrated TracIn-DAS merged and evaluated", flush=True)


if __name__ == "__main__":
    main()
