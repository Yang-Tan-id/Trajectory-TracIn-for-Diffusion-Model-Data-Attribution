"""Prepare, train, and evaluate 200 top-1000 removal models."""

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from exp_config import LOG_DIR, ROOT


OUT_ROOT = ROOT / "topk_removal_retrain"


def run_logged(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", buffering=1) as stream:
        stream.write("\n$ " + " ".join(command) + "\n")
        return subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    gpus = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")

    subprocess.run([sys.executable, "09_prepare_topk_removal.py"], check=True)
    with open(OUT_ROOT / "jobs.json") as handle:
        jobs = json.load(handle)
    if len(jobs) != 200:
        raise ValueError(f"expected 200 jobs, found {len(jobs)}")

    pending = queue.Queue()
    for job in jobs:
        pending.put(job)
    lock = threading.Lock()
    state = {"done": 0, "failed": []}
    started = time.perf_counter()

    def worker(gpu):
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            job_dir = Path(job["job_dir"])
            label = f"{job['method_tag']}/q{int(job['query_id']):02d}"
            log_path = LOG_DIR / "topk_removal" / job["method_tag"] / f"q{int(job['query_id']):02d}.log"
            try:
                print(f"[gpu {gpu}] START {label}", flush=True)
                train_command = [
                    sys.executable, "-u", "topk_removal_train_worker.py",
                    "--job-dir", str(job_dir), "--gpu", str(gpu),
                ]
                code = run_logged(train_command, log_path)
                if code == 0:
                    eval_command = [
                        sys.executable, "-u", "topk_removal_eval_worker.py",
                        "--job-dir", str(job_dir), "--gpu", str(gpu),
                    ]
                    code = run_logged(eval_command, log_path)
                with lock:
                    if code != 0:
                        state["failed"].append({"label": label, "code": code})
                    else:
                        state["done"] += 1
                    elapsed = time.perf_counter() - started
                    finished = state["done"] + len(state["failed"])
                    eta = elapsed / max(finished, 1) * (len(jobs) - finished)
                    print(
                        f"[overall] {'DONE' if code == 0 else 'FAIL'} {label} | "
                        f"finished={finished}/{len(jobs)} successful={state['done']} "
                        f"elapsed={elapsed/3600:.2f}h eta≈{eta/3600:.2f}h",
                        flush=True,
                    )
            finally:
                pending.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if state["failed"]:
        failure_path = OUT_ROOT / "failed_jobs.json"
        with open(failure_path, "w") as handle:
            json.dump(state["failed"], handle, indent=2)
        raise SystemExit(f"{len(state['failed'])} jobs failed; rerun to resume. See {failure_path}")
    subprocess.run([sys.executable, "11_summarize_topk_removal.py"], check=True)
    print("[done] all 200 top-k removal models trained and evaluated", flush=True)


if __name__ == "__main__":
    main()
