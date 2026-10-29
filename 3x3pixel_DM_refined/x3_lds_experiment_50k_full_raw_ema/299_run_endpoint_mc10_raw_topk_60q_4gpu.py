"""Run raw-gradient Endpoint MC10/RAW removals and compare with AdamW."""

from __future__ import annotations

import argparse
import importlib.util
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from endpoint_tracin_adamw_mc10_config import eta_raw_gradient_method
from exp_config import LOG_DIR, ROOT


QUERY_IDS = tuple(range(60))
FRACTION = "remove_10pct"
RAW_ROOT = ROOT / "topk_removal_endpoint_raw_gradient_mc10_raw_60q_10pct"
ADAMW_ROOT = ROOT / (
    "topk_removal_pint_endpoint_retrac_random_60q_10pct_"
    "final_endpoint_linear_adamw_sign"
)
RAW_TAG = "endpoint_mc10_raw_gradient_raw"
ADAMW_TAG = "endpoint_tracin_raw"
METHODS = (
    {
        "tag": RAW_TAG,
        "label": "RAW-GRADIENT ENDPOINT MC10 / RAW",
        "method": eta_raw_gradient_method(
            "raw", "linear", query_mc=10, train_mc=10
        ),
        "prediction_sign": -1,
        "description": (
            "Raw-gradient endpoint-loss TracIn, query100x10/train100x10, "
            "raw normalization, linear"
        ),
    },
)
METRICS = (
    "noise_trajectory_mse",
    "trajectory_mse",
    "endpoint_mse",
    "simple_loss_abs_change",
)


def load_base():
    path = Path(__file__).with_name("287_prepare_topk5_random_60q_10pct.py")
    spec = importlib.util.spec_from_file_location("topk_prepare_base", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.QUERY_IDS = QUERY_IDS
    module.OUT_ROOT = RAW_ROOT
    module.METHODS = METHODS
    module.EXPECTED_JOB_COUNT = len(QUERY_IDS)
    return module


def run_logged(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", buffering=1) as stream:
        stream.write("\n$ " + " ".join(command) + "\n")
        return subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT)


def directory(root, tag, query_id):
    return root / tag / FRACTION / f"q{query_id:02d}"


def load_json(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with open(path) as handle:
        return json.load(handle)


def summarize():
    raw = {
        q: load_json(directory(RAW_ROOT, RAW_TAG, q) / "evaluation.json")
        for q in QUERY_IDS
    }
    adamw = {
        q: load_json(directory(ADAMW_ROOT, ADAMW_TAG, q) / "evaluation.json")
        for q in QUERY_IDS
    }
    overlaps = []
    for q in QUERY_IDS:
        raw_indices = np.load(
            directory(RAW_ROOT, RAW_TAG, q) / "removed_indices.npy"
        )
        adamw_indices = np.load(
            directory(ADAMW_ROOT, ADAMW_TAG, q) / "removed_indices.npy"
        )
        overlaps.append(
            np.intersect1d(raw_indices, adamw_indices).size / raw_indices.size
        )

    output = {
        "query_ids": list(QUERY_IDS),
        "difference": "Full-AdamW minus raw-gradient",
        "topk_overlap": {
            "mean": float(np.mean(overlaps)),
            "std": float(np.std(overlaps)),
            "min": float(np.min(overlaps)),
            "max": float(np.max(overlaps)),
        },
        "metrics": {},
    }
    lines = [
        "ENDPOINT MC10 RAW TRUE 10% REMOVAL: RAW-GRADIENT VS FULL-ADAMW",
        "q00-q59; delta = Full-AdamW minus raw-gradient",
        "raw-gradient ranks descending; AdamW ranks ascending",
        "",
        (
            f"Top-K overlap: {np.mean(overlaps):.4f} +/- {np.std(overlaps):.4f} "
            f"[min={np.min(overlaps):.4f}, max={np.max(overlaps):.4f}]"
        ),
        "",
        (
            f"{'metric':28s} {'raw-gradient':24s} {'Full-AdamW':24s} "
            f"{'delta':>12s} {'ratio':>9s} {'wins':>8s} {'p-W':>12s}"
        ),
    ]
    for metric in METRICS:
        raw_values = np.asarray([raw[q][metric] for q in QUERY_IDS], dtype=np.float64)
        adamw_values = np.asarray(
            [adamw[q][metric] for q in QUERY_IDS], dtype=np.float64
        )
        delta = adamw_values - raw_values
        item = {
            "raw_mean": float(raw_values.mean()),
            "raw_std": float(raw_values.std()),
            "adamw_mean": float(adamw_values.mean()),
            "adamw_std": float(adamw_values.std()),
            "adamw_minus_raw_mean": float(delta.mean()),
            "adamw_over_raw_ratio": float(adamw_values.mean() / raw_values.mean()),
            "adamw_wins": int(np.sum(delta > 0)),
            "raw_wins": int(np.sum(delta < 0)),
            "p_wilcoxon_two_sided": float(wilcoxon(delta).pvalue),
        }
        output["metrics"][metric] = item
        lines.append(
            f"{metric:28s} "
            f"{item['raw_mean']:.6e}+/-{item['raw_std']:.3e} "
            f"{item['adamw_mean']:.6e}+/-{item['adamw_std']:.3e} "
            f"{item['adamw_minus_raw_mean']:+.3e} "
            f"{item['adamw_over_raw_ratio']:9.4f} "
            f"{item['adamw_wins']:02d}/60 "
            f"{item['p_wilcoxon_two_sided']:.6g}"
        )

    json_path = RAW_ROOT / "endpoint_mc10_raw_gradient_vs_adamw_raw.json"
    text_path = RAW_ROOT / "endpoint_mc10_raw_gradient_vs_adamw_raw.txt"
    with open(json_path, "w") as handle:
        json.dump(output, handle, indent=2)
    text_path.write_text("\n".join(lines) + "\n")
    print(text_path.read_text(), flush=True)
    print(f"[saved] {json_path}", flush=True)
    print(f"[saved] {text_path}", flush=True)


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

    prepare = load_base()
    if not args.skip_prepare:
        prepare.main()
    with open(RAW_ROOT / "jobs.json") as handle:
        jobs = json.load(handle)
    if len(jobs) != len(QUERY_IDS):
        raise ValueError(f"expected {len(QUERY_IDS)} jobs, found {len(jobs)}")

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
            query_id = int(job["query_id"])
            label = f"{RAW_TAG}/q{query_id:02d}"
            job_dir = Path(job["job_dir"])
            log_path = (
                LOG_DIR
                / "topk_endpoint_raw_gradient_mc10_raw_60q_10pct"
                / f"q{query_id:02d}.log"
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
                        f"finished={finished}/{len(jobs)} "
                        f"elapsed={elapsed/3600:.2f}h eta≈{eta/3600:.2f}h",
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
        path = RAW_ROOT / "failed_jobs.json"
        with open(path, "w") as handle:
            json.dump(state["failed"], handle, indent=2)
        raise SystemExit(f"{len(state['failed'])} jobs failed; see {path}")
    summarize()


if __name__ == "__main__":
    main()
