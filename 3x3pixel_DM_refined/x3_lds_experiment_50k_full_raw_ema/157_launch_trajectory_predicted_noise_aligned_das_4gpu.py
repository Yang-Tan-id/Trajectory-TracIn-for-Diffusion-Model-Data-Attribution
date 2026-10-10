"""Launch q00-q09 trajectory-predicted-noise aligned DAS on four GPUs."""

import argparse
import subprocess
import sys
import time

from exp_config import LOG_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--feature-batch-size", type=int, default=64)
    args = parser.parse_args()
    gpus = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    if len(gpus) != 4 or len(set(gpus)) != 4:
        raise ValueError("--gpus must contain exactly four distinct GPU IDs")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "trajectory_predicted_noise_aligned_das_10q_4gpu.log"
    active = {}
    with open(log_path, "a", buffering=1) as stream:
        stream.write(
            "\n[launcher] q00-q09 final-EMA trajectory-predicted-noise "
            "aligned DAS; 99 timestamps x probe10; projected4096\n"
        )
        for shard_index, gpu in enumerate(gpus):
            label = f"timestamp-shard-{shard_index}"
            command = [
                sys.executable,
                "-u",
                "154_run_trajectory_predicted_noise_aligned_das_shard.py",
                "--gpu",
                str(gpu),
                "--timestamp-shard-index",
                str(shard_index),
                "--timestamp-shard-count",
                "4",
                "--feature-batch-size",
                str(args.feature_batch_size),
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
                "155_merge_trajectory_predicted_noise_aligned_das.py",
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
                "156_eval_trajectory_predicted_noise_aligned_das_lds.py",
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print(f"[done] trajectory-predicted-noise aligned DAS; log={log_path}")


if __name__ == "__main__":
    main()
