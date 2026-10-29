"""Train/evaluate the 180 raw-gradient PINT/Endpoint Top-K jobs."""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import importlib.util

from exp_config import LOG_DIR


def load_prepare():
    path = Path(__file__).with_name(
        "291_prepare_raw_gradient_pint_endpoint_60q_10pct.py"
    )
    spec = importlib.util.spec_from_file_location("raw_gradient_topk_prepare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.configure_base()
    return module


def run_logged(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", buffering=1) as stream:
        stream.write("\n$ " + " ".join(command) + "\n")
        return subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus or args.workers_per_gpu <= 0:
        raise ValueError("invalid GPU/worker configuration")

    prepare = load_prepare()
    if not args.skip_prepare:
        subprocess.run(
            [sys.executable, "-u", "291_prepare_raw_gradient_pint_endpoint_60q_10pct.py"],
            check=True,
        )
    with open(prepare.OUT_ROOT / "jobs.json") as handle:
        jobs = json.load(handle)
    if len(jobs) != prepare.EXPECTED_JOB_COUNT:
        raise ValueError(f"expected {prepare.EXPECTED_JOB_COUNT} jobs, found {len(jobs)}")

    pending = queue.Queue()
    for job in jobs:
        pending.put(job)
    lock = threading.Lock()
    state = {"done": 0, "failed": []}
    started = time.perf_counter()

    def worker(gpu, slot):
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            label = f"{job['method_tag']}/q{int(job['query_id']):02d}"
            job_dir = Path(job["job_dir"])
            log_path = (
                LOG_DIR
                / "topk_raw_gradient_pint_endpoint_mc1_60q_10pct"
                / job["method_tag"]
                / f"q{int(job['query_id']):02d}.log"
            )
            try:
                print(f"[gpu {gpu} slot {slot}] START {label}", flush=True)
                code = 0
                if not args.eval_only:
                    code = run_logged(
                        [sys.executable, "-u", "topk_removal_train_worker.py", "--job-dir", str(job_dir), "--gpu", str(gpu)],
                        log_path,
                    )
                if code == 0:
                    code = run_logged(
                        [sys.executable, "-u", "topk_removal_eval_worker.py", "--job-dir", str(job_dir), "--gpu", str(gpu)],
                        log_path,
                    )
                with lock:
                    if code:
                        state["failed"].append({"label": label, "code": code, "log": str(log_path)})
                    else:
                        state["done"] += 1
                    finished = state["done"] + len(state["failed"])
                    elapsed = time.perf_counter() - started
                    eta = elapsed / max(finished, 1) * (len(jobs) - finished)
                    print(
                        f"[overall] {'FAIL' if code else 'DONE'} {label} "
                        f"finished={finished}/{len(jobs)} elapsed={elapsed/3600:.2f}h "
                        f"eta≈{eta/3600:.2f}h",
                        flush=True,
                    )
            finally:
                pending.task_done()

    threads = [
        threading.Thread(target=worker, args=(gpu, slot))
        for gpu in gpus
        for slot in range(args.workers_per_gpu)
    ]
    print(
        f"[launcher] jobs={len(jobs)} GPUs={gpus} "
        f"workers_per_gpu={args.workers_per_gpu} concurrent={len(threads)}",
        flush=True,
    )
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if state["failed"]:
        path = prepare.OUT_ROOT / "failed_jobs.json"
        with open(path, "w") as handle:
            json.dump(state["failed"], handle, indent=2)
        raise SystemExit(f"{len(state['failed'])} jobs failed; see {path}")
    subprocess.run(
        [sys.executable, "-u", "293_summarize_raw_gradient_pint_endpoint_60q_10pct.py"],
        check=True,
    )


if __name__ == "__main__":
    main()
