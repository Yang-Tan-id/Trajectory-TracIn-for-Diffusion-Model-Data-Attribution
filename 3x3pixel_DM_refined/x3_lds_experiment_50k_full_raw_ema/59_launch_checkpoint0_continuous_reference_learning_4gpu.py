"""Launch ten checkpoint-0 continuous reference-learning queries on four GPUs."""

import argparse
import os
import subprocess
import sys
import threading

from checkpoint0_continuous_reference_learning_config import CUDA_IDS, LOG_DIR


def stream_process(label, command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
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
        raise RuntimeError(f"{label} exited with code {code}")


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    if len(CUDA_IDS) < 4:
        raise ValueError("continuous reference-learning launcher requires four GPUs")
    subprocess.run(
        [
            sys.executable,
            "56_verify_checkpoint0_continuous_reference_learning.py",
        ],
        check=True,
    )
    errors = []
    threads = []

    def worker(shard_index, gpu):
        try:
            stream_process(
                f"gpu{gpu}",
                [
                    sys.executable,
                    "-u",
                    "57_run_checkpoint0_continuous_reference_learning_shard.py",
                    "--gpu",
                    str(gpu),
                    "--shard-index",
                    str(shard_index),
                    "--shard-count",
                    "4",
                ],
                LOG_DIR
                / f"checkpoint0_continuous_reference_learning_gpu{gpu}.log",
            )
        except Exception as exc:
            errors.append((gpu, exc))

    for shard_index, gpu in enumerate(CUDA_IDS[:4]):
        thread = threading.Thread(
            target=worker, args=(shard_index, gpu), daemon=False
        )
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    if errors:
        raise RuntimeError(
            "; ".join(f"gpu{gpu}: {error}" for gpu, error in errors)
        )
    subprocess.run(
        [
            sys.executable,
            "58_eval_checkpoint0_continuous_reference_learning_lds.py",
        ],
        check=True,
    )
    print(
        "[done] checkpoint-0 continuous reference-learning LDS",
        flush=True,
    )


if __name__ == "__main__":
    main()
