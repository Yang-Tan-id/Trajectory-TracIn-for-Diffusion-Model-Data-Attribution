"""Run q00-q09 projected Traj with ten antithetic training-noise pairs."""

import argparse
import subprocess
import sys
import time

from exp_config import LOG_DIR, TRACIN_PROJECTED_BATCH_SIZE


QUERY_IDS = ",".join(str(value) for value in range(10))
OUTPUT_SUFFIX = "antithetic10pairs_q00_q09"
PAIR_COUNT = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument(
        "--batch-size", type=int, default=TRACIN_PROJECTED_BATCH_SIZE
    )
    args = parser.parse_args()
    gpus = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    if len(gpus) != 4 or len(set(gpus)) != 4:
        raise ValueError("--gpus must contain exactly four distinct GPU IDs")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "projected_traj_antithetic10pairs_q00_q09_4gpu.log"
    active = {}
    with open(log_path, "a", buffering=1) as stream:
        stream.write(
            "\n[launcher] q00-q09 projected first-order raw next-checkpoint "
            "Traj; 10 antithetic train-noise pairs; query/train align t only\n"
        )
        for shard_index, gpu in enumerate(gpus):
            label = f"timestamp-shard-{shard_index}"
            command = [
                sys.executable,
                "-u",
                "run_projected_traj_bank.py",
                "--family",
                "prompted",
                "--gpu",
                str(gpu),
                "--timestamp-shard-index",
                str(shard_index),
                "--timestamp-shard-count",
                "4",
                "--checkpoint-direction",
                "forward",
                "--first-order-only",
                "--query-ids",
                QUERY_IDS,
                "--train-noise-sampling",
                "antithetic",
                "--train-mc-pairs",
                str(PAIR_COUNT),
                "--output-suffix",
                OUTPUT_SUFFIX,
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
                "merge_projected_traj_shards.py",
                "--family",
                "prompted",
                "--timestamp-shard-count",
                "4",
                "--checkpoint-direction",
                "forward",
                "--first-order-only",
                "--query-ids",
                QUERY_IDS,
                "--train-noise-sampling",
                "antithetic",
                "--train-mc-pairs",
                str(PAIR_COUNT),
                "--output-suffix",
                OUTPUT_SUFFIX,
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-u",
                "165_eval_projected_traj_antithetic10pairs_10q_lds.py",
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print(f"[done] q00-q09 projected antithetic Traj; log={log_path}")


if __name__ == "__main__":
    main()
