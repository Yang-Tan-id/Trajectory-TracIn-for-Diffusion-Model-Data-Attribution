"""Launch q00-q09 endpoint-MC100 AdamW unlearning on exactly four GPUs."""

import os
import subprocess
import sys
import threading

from mucs_endpoint_unlearning_config import *


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


def verify_inputs():
    required = [
        mucs_checkpoint_path(MUCS_NULL_EPOCH),
        mucs_checkpoint_path(MUCS_INITIAL_EPOCH),
        MASK_DIR / "membership.npy",
        QUERY_DIR / "manifest.json",
    ]
    for qid in MUCS_QUERY_IDS:
        required.extend(
            [
                QUERY_DIR / f"q{qid:02d}" / "final_state.npy",
                QUERY_DIR / f"q{qid:02d}" / "query.json",
            ]
        )
    for metric in MUCS_LDS_METRICS:
        required.append(LDS_DIR / f"observed_{metric}.npy")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} inputs; first={missing[0]}")
    print(
        f"[verified] q00-q09 final raw -> AdamW ascent lr={MUCS_UNLEARNING_LR:.3e}; "
        f"null=epoch{MUCS_NULL_EPOCH}; target={MUCS_NULL_GAP_FRACTION:.0%} gap",
        flush=True,
    )


def main():
    verify_inputs()
    errors = []
    threads = []

    def worker(gpu):
        try:
            stream_process(
                f"gpu{gpu}",
                [
                    sys.executable,
                    "-u",
                    "33_run_mucs_endpoint_unlearning_shard.py",
                    "--gpu", str(gpu),
                    "--shard-index", str(gpu),
                    "--shard-count", "4",
                ],
                LOG_DIR / f"mucs_endpoint_unlearning_gpu{gpu}.log",
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
        [sys.executable, "34_eval_mucs_endpoint_unlearning_lds.py"],
        check=True,
    )
    print("[done] endpoint-unlearning MUCS scores and LDS complete", flush=True)


if __name__ == "__main__":
    main()
