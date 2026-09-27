"""Launch the q00-q19 twelve-probe Traj experiment on four GPUs."""

import argparse
import os
import subprocess
import sys
import threading

from traj_probe12_config import LOG_DIR


def stream_process(label, command, log_path, errors):
    try:
        with open(log_path, "a", buffering=1) as log:
            log.write("\n$ " + " ".join(command) + "\n")
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            for line in process.stdout:
                message = f"[{label}] {line}"
                print(message, end="", flush=True)
                log.write(message)
            code = process.wait()
        if code != 0:
            errors.append((label, code))
    except Exception as exc:
        errors.append((label, repr(exc)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    gpus = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    if len(gpus) != 4:
        raise ValueError("this launcher requires exactly four GPU ids")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    subprocess.run([sys.executable, "42_verify_traj_probe12.py"], check=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "traj_probe12_20q_4gpu.log"
    errors = []
    threads = []
    for shard_index, gpu in enumerate(gpus):
        command = [
            sys.executable,
            "-u",
            "43_run_traj_probe12_timestamp_shard.py",
            "--gpu",
            str(gpu),
            "--timestamp-shard-index",
            str(shard_index),
            "--timestamp-shard-count",
            "4",
            "--batch-size",
            str(args.batch_size),
        ]
        thread = threading.Thread(
            target=stream_process,
            args=(f"gpu{gpu}-shard{shard_index}", command, log_path, errors),
            daemon=False,
        )
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    if errors:
        raise RuntimeError(
            "; ".join(f"{label}: {detail}" for label, detail in errors)
        )

    for label, command in (
        (
            "merge",
            [
                sys.executable,
                "-u",
                "44_merge_traj_probe12_timestamp_shards.py",
                "--timestamp-shard-count",
                "4",
            ],
        ),
        ("lds", [sys.executable, "-u", "45_eval_traj_probe12_lds.py"]),
    ):
        stream_process(label, command, log_path, errors)
        if errors:
            raise RuntimeError(
                "; ".join(f"{name}: {detail}" for name, detail in errors)
            )
    print(f"[done] probe12 Traj scores, timestamp artifacts, and LDS; log={log_path}")


if __name__ == "__main__":
    main()
