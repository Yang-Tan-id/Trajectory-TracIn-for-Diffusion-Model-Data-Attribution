"""Launch q00-q09 checkpoint-AdamW reference learning on four GPUs."""

import argparse
import os
import subprocess
import sys
import threading

from checkpoint_adamw_reference_learning_config import CUDA_IDS, LOG_DIR


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
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{label} exited with code {return_code}")


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    if len(CUDA_IDS) < 4:
        raise ValueError("checkpoint-AdamW launcher requires four CUDA_IDS")
    subprocess.run(
        [sys.executable, "52_verify_checkpoint_adamw_reference_learning.py"],
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
                    "53_run_checkpoint_adamw_reference_learning_shard.py",
                    "--gpu",
                    str(gpu),
                    "--shard-index",
                    str(shard_index),
                    "--shard-count",
                    "4",
                ],
                LOG_DIR
                / f"checkpoint_adamw_reference_learning_gpu{gpu}.log",
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
        [sys.executable, "54_eval_checkpoint_adamw_reference_learning_lds.py"],
        check=True,
    )
    print(
        "[done] checkpoint-AdamW reference-learning four-score LDS",
        flush=True,
    )


if __name__ == "__main__":
    main()
