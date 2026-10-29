"""Train/evaluate the 360 q00-q59, 10% top-k/random removal jobs."""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from exp_config import LOG_DIR


def load_prepare_module():
    path = Path(__file__).with_name("287_prepare_topk5_random_60q_10pct.py")
    spec = spec_from_file_location("topk5_random_prepare", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_logged(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", buffering=1) as stream:
        stream.write("\n$ " + " ".join(command) + "\n")
        return subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="reuse trained checkpoints and only refresh evaluation metrics",
    )
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("at least one GPU is required")
    if args.workers_per_gpu <= 0:
        raise ValueError("--workers-per-gpu must be positive")

    prepare = load_prepare_module()
    if not args.skip_prepare:
        subprocess.run(
            [sys.executable, "-u", "287_prepare_topk5_random_60q_10pct.py"],
            check=True,
        )
    with open(prepare.OUT_ROOT / "jobs.json") as handle:
        jobs = json.load(handle)
    if len(jobs) != prepare.EXPECTED_JOB_COUNT:
        raise ValueError(
            f"expected {prepare.EXPECTED_JOB_COUNT} jobs, found {len(jobs)}"
        )

    pending: queue.Queue = queue.Queue()
    for job in jobs:
        pending.put(job)
    lock = threading.Lock()
    state = {"done": 0, "failed": []}
    started = time.perf_counter()

    def worker(gpu: int, slot: int) -> None:
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            job_dir = Path(job["job_dir"])
            label = f"{job['method_tag']}/q{int(job['query_id']):02d}"
            log_path = (
                LOG_DIR
                / "topk5_random_60q_10pct_correct_direction"
                / job["method_tag"]
                / f"q{int(job['query_id']):02d}.log"
            )
            try:
                print(f"[gpu {gpu} slot {slot}] START {label}", flush=True)
                code = 0
                if not args.eval_only:
                    code = run_logged(
                        [
                            sys.executable,
                            "-u",
                            "topk_removal_train_worker.py",
                            "--job-dir",
                            str(job_dir),
                            "--gpu",
                            str(gpu),
                        ],
                        log_path,
                    )
                if code == 0:
                    code = run_logged(
                        [
                            sys.executable,
                            "-u",
                            "topk_removal_eval_worker.py",
                            "--job-dir",
                            str(job_dir),
                            "--gpu",
                            str(gpu),
                        ],
                        log_path,
                    )
                with lock:
                    if code:
                        state["failed"].append(
                            {"label": label, "code": code, "log": str(log_path)}
                        )
                    else:
                        state["done"] += 1
                    finished = state["done"] + len(state["failed"])
                    elapsed = time.perf_counter() - started
                    eta = elapsed / max(finished, 1) * (len(jobs) - finished)
                    print(
                        f"[overall] {'FAIL' if code else 'DONE'} gpu={gpu} "
                        f"slot={slot} {label} finished={finished}/{len(jobs)} "
                        f"successful={state['done']} elapsed={elapsed/3600:.2f}h "
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
        f"workers_per_gpu={args.workers_per_gpu} concurrent_models={len(threads)}",
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
        raise SystemExit(
            f"{len(state['failed'])} jobs failed; rerun to resume. See {path}"
        )
    subprocess.run(
        [sys.executable, "-u", "289_summarize_topk5_random_60q_10pct.py"],
        check=True,
    )
    print("[done] all 360 removal models evaluated and summarized", flush=True)


if __name__ == "__main__":
    main()
