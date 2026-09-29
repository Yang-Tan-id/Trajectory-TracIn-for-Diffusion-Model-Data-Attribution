"""Prepare, train, evaluate, and summarize all 900 removal jobs on four GPUs."""

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from exp_config import LOG_DIR
from importlib.util import module_from_spec, spec_from_file_location


def load_prepare_module():
    path = Path(__file__).with_name("117_prepare_retrac_tracindas_topk_100q.py")
    spec = spec_from_file_location("retrac_topk_prepare", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_logged(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", buffering=1) as stream:
        stream.write("\n$ " + " ".join(command) + "\n")
        return subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--skip-prepare", action="store_true")
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("at least one GPU is required")
    prepare = load_prepare_module()
    if not args.skip_prepare:
        subprocess.run(
            [sys.executable, "-u", "117_prepare_retrac_tracindas_topk_100q.py"],
            check=True,
        )
    with open(prepare.OUT_ROOT / "jobs.json") as handle:
        jobs = json.load(handle)
    if len(jobs) != 900:
        raise ValueError(f"expected 900 jobs, found {len(jobs)}")

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
            label = (
                f"{job['method_tag']}/{job['removal_fraction_tag']}/"
                f"q{int(job['query_id']):02d}"
            )
            log_path = (
                LOG_DIR
                / "topk_retrac_tracindas_100q"
                / job["method_tag"]
                / job["removal_fraction_tag"]
                / f"q{int(job['query_id']):02d}.log"
            )
            try:
                print(f"[gpu {gpu}] START {label}", flush=True)
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
                        f"[overall] {'FAIL' if code else 'DONE'} {label} "
                        f"finished={finished}/900 successful={state['done']} "
                        f"elapsed={elapsed/3600:.2f}h eta≈{eta/3600:.2f}h",
                        flush=True,
                    )
            finally:
                pending.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
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
        [sys.executable, "-u", "119_summarize_retrac_tracindas_topk_100q.py"],
        check=True,
    )
    print("[done] all 900 removal models evaluated and summarized", flush=True)


if __name__ == "__main__":
    main()
