"""Launch four-step trajectory unlearning on four GPUs, then evaluate LDS."""

import json
import os
import subprocess
import sys
import threading

from forward_loss_alignment_config import *


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


def verify_preparation():
    required = (
        replay_t_path(),
        replay_noise_path(),
        FLA_REPLAY_DIR / "metadata.json",
        baseline_event_path(),
        FLA_BASELINE_DIR / "done.json",
    )
    required += tuple(
        path
        for qid in FLA_QUERY_IDS
        for path in (
            FLA_QUERY_DIR / f"q{qid:02d}" / "trajectory_xt_1000.npy",
            FLA_QUERY_DIR / f"q{qid:02d}" / "target_eps_ema_1000.npy",
            FLA_QUERY_DIR / f"q{qid:02d}" / "trajectory_t_1000.npy",
        )
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Run `python -u 24_prepare_forward_loss_alignment.py --gpu 0` first. "
            f"Missing: {missing}"
        )
    with open(FLA_BASELINE_DIR / "done.json") as handle:
        done = json.load(handle)
    if int(done.get("format_version", 0)) < 2:
        raise RuntimeError("Event-level baseline cache is outdated; rerun script 24.")


def main():
    verify_preparation()
    errors = []
    threads = []

    def worker(gpu):
        try:
            stream_process(
                f"gpu{gpu}",
                [
                    sys.executable,
                    "28_run_trajectory_unlearning_4step_shard.py",
                    "--gpu", str(gpu),
                    "--shard-index", str(gpu),
                    "--shard-count", "4",
                ],
                LOG_DIR / f"trajectory_unlearning_4step_gpu{gpu}.log",
            )
        except Exception as exc:
            errors.append((gpu, exc))

    for gpu in range(4):
        thread = threading.Thread(target=worker, args=(gpu,), daemon=False)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    if errors:
        raise RuntimeError("; ".join(f"gpu{gpu}: {exc}" for gpu, exc in errors))
    subprocess.run(
        [sys.executable, "29_eval_trajectory_unlearning_4step_lds.py"],
        check=True,
    )
    print("[done] four-step trajectory unlearning and LDS complete", flush=True)


if __name__ == "__main__":
    main()
