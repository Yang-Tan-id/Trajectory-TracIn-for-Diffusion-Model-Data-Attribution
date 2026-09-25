import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from exp_config import *

with open(ROOT / "train_jobs.json") as f:
    JOBS = json.load(f)

q = queue.Queue()
for j in JOBS:
    q.put(j)

TOTAL_MODELS = 0
for j in JOBS:
    if j["kind"] == "base":
        TOTAL_MODELS += 1
    else:
        TOTAL_MODELS += len(j["families"])

state_lock = threading.Lock()
completed_models = 0
started_at = time.perf_counter()


def fmt(sec):
    sec = max(0, int(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def mark_complete(label, skipped=False):
    global completed_models
    with state_lock:
        completed_models += 1
        elapsed = time.perf_counter() - started_at
        rate = completed_models / elapsed if elapsed > 0 else 0.0
        remain = (TOTAL_MODELS - completed_models) / rate if rate > 0 else 0.0
        pct = 100.0 * completed_models / TOTAL_MODELS
        tag = "SKIP" if skipped else "DONE"
        print(
            f"[overall] {tag} {label} | "
            f"{completed_models}/{TOTAL_MODELS} models ({pct:.1f}%) | "
            f"elapsed={fmt(elapsed)} | eta≈{fmt(remain)}",
            flush=True,
        )


def run_cmd(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", buffering=1) as log:
        log.write("\n$ " + " ".join(cmd) + "\n")
        return subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)


def worker(gpu):
    while True:
        try:
            job = q.get_nowait()
        except queue.Empty:
            return

        try:
            if job["kind"] == "base":
                fams = [job["family"]]
                kind = "base"
            else:
                fams = job["families"]
                kind = "subset"

            for fam in fams:
                if kind == "base":
                    out = MODEL_DIR / "base" / fam / f"epoch_{EPOCHS:04d}.pt"
                    log = LOG_DIR / f"base_{fam}.log"
                    label = f"base/{fam}"
                else:
                    out = (
                        MODEL_DIR / "subsets" / f"seed_{int(job['lds_seed']):02d}"
                        / f"subset_{int(job['subset_id']):02d}" / fam / f"epoch_{EPOCHS:04d}.pt"
                    )
                    log = LOG_DIR / (
                        f"subset_s{int(job['lds_seed']):02d}_"
                        f"m{int(job['subset_id']):02d}_{fam}.log"
                    )
                    label = (
                        f"subset s{int(job['lds_seed']):02d}/"
                        f"m{int(job['subset_id']):02d}/{fam}"
                    )

                if out.exists():
                    print(f"[gpu {gpu}] skip existing {out}", flush=True)
                    mark_complete(label, skipped=True)
                    continue

                cmd = [
                    sys.executable, "train_worker.py",
                    "--kind", kind,
                    "--family", fam,
                    "--seed", str(job["seed"]),
                    "--gpu", str(gpu),
                ]
                if kind == "subset":
                    cmd += [
                        "--mask-path", job["mask_path"],
                        "--lds-seed", str(job["lds_seed"]),
                        "--subset-id", str(job["subset_id"]),
                    ]

                print(f"[gpu {gpu}] START {kind}/{fam} seed={job['seed']}", flush=True)
                rc = run_cmd(cmd, log)
                if rc != 0:
                    print(f"[gpu {gpu}] FAIL rc={rc}: {' '.join(cmd)}", flush=True)
                    break

                print(f"[gpu {gpu}] DONE {kind}/{fam}", flush=True)
                mark_complete(label, skipped=False)
        finally:
            q.task_done()


threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in CUDA_IDS]
for t in threads:
    t.start()
for t in threads:
    t.join()

elapsed = time.perf_counter() - started_at
print(f"[done] all training jobs exhausted | total_elapsed={fmt(elapsed)}")
