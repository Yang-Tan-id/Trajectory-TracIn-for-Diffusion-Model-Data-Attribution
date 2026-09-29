"""Launch final-EMA trajectory inverse-noise DAS on four GPUs."""

import argparse
import subprocess
import sys
import time

from exp_config import LOG_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--condition-batch-size", type=int, default=64)
    args = parser.parse_args()
    gpus = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    if len(gpus) != 4 or len(set(gpus)) != 4:
        raise ValueError("--gpus must contain exactly four distinct GPU IDs")
    if args.condition_batch_size <= 0:
        raise ValueError("condition batch size must be positive")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "trajectory_inverse_noise_das_q00_q09_4gpu.log"
    active = {}
    with open(log_path, "a", buffering=1) as stream:
        stream.write(
            "\n[launcher] final-EMA query-dependent trajectory inverse-noise "
            "DAS q00-q09; 99 timestamps x probe10; projected4096\n"
        )
        for shard_index, gpu in enumerate(gpus):
            label = f"timestamp-shard-{shard_index}"
            command = [
                sys.executable,
                "-u",
                "140_run_trajectory_inverse_noise_das_shard.py",
                "--gpu",
                str(gpu),
                "--timestamp-shard-index",
                str(shard_index),
                "--timestamp-shard-count",
                "4",
                "--condition-batch-size",
                str(args.condition_batch_size),
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
                "141_merge_trajectory_inverse_noise_das.py",
                "--timestamp-shard-count",
                "4",
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-u",
                "143_eval_trajectory_inverse_noise_das_10q_lds.py",
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print(f"[done] trajectory inverse-noise DAS; log={log_path}")


if __name__ == "__main__":
    main()
