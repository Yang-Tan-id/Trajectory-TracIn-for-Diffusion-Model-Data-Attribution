import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from exp_config import *

with open(QUERY_DIR / "manifest.json") as f:
    queries = json.load(f)

jobs = []
for q in queries:
    for method in (
        "traj_ref_raw",
        "traj_next_raw",
        "traj_ref_ema",
        "traj_next_ema",
        "das",
    ):
        jobs.append((q, method))

qwork = queue.Queue()
for j in jobs:
    qwork.put(j)

TOTAL_JOBS = len(jobs)
done_jobs = 0
lock = threading.Lock()
started_at = time.perf_counter()


def fmt(sec):
    sec = max(0, int(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def output_exists(qid, method):
    root = ATTR_DIR / method / f"q{qid:02d}"
    if method == "das":
        for lam in DAS_LAMBDAS:
            tag = str(float(lam)).replace(".", "p")
            if not (root / f"lambda_{tag}" / "scores.npy").exists():
                return False
        return True
    return (root / "scores.npy").exists()


def mark_done(method, qid, skipped=False):
    global done_jobs
    with lock:
        done_jobs += 1
        elapsed = time.perf_counter() - started_at
        rate = done_jobs / elapsed if elapsed > 0 else 0
        eta = (TOTAL_JOBS - done_jobs) / rate if rate > 0 else 0
        print(
            f"[attr overall] {'SKIP' if skipped else 'DONE'} {method} q{qid:02d} | "
            f"{done_jobs}/{TOTAL_JOBS} ({100.0*done_jobs/TOTAL_JOBS:.1f}%) | "
            f"elapsed={fmt(elapsed)} | eta≈{fmt(eta)}",
            flush=True,
        )


def worker(gpu):
    while True:
        try:
            rec, method = qwork.get_nowait()
        except queue.Empty:
            return
        try:
            qid = int(rec["query_id"])
            if output_exists(qid, method):
                print(f"[gpu {gpu}] skip {method} q{qid:02d}", flush=True)
                mark_done(method, qid, skipped=True)
                continue

            cmd = [
                sys.executable,
                "attribution_one_query.py",
                "--query-json", str(Path(rec["dir"]) / "query.json"),
                "--method", method,
                "--gpu", str(gpu),
            ]
            log = LOG_DIR / f"attr_{method}_q{qid:02d}.log"
            log.parent.mkdir(parents=True, exist_ok=True)

            print(f"[gpu {gpu}] START {method} q{qid:02d}", flush=True)
            with open(log, "a", buffering=1) as f:
                f.write("\n$ " + " ".join(cmd) + "\n")
                rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)

            if rc != 0:
                print(f"[gpu {gpu}] FAIL {method} q{qid:02d} rc={rc}", flush=True)
            else:
                print(f"[gpu {gpu}] DONE {method} q{qid:02d}", flush=True)
                mark_done(method, qid, skipped=False)
        finally:
            qwork.task_done()


threads = [
    threading.Thread(target=worker, args=(gpu,), daemon=False)
    for gpu in CUDA_IDS
]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"[done] attribution queue exhausted | total_elapsed={fmt(time.perf_counter()-started_at)}")
