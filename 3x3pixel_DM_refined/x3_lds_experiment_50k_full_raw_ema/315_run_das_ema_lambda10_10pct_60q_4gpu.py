#!/usr/bin/env python3
"""Run q00-q59 original DAS-EMA lambda=10 true 10% removals.

The launcher is resumable and, after evaluation, compares DAS against the
existing raw-gradient and Full-AdamW PINT / RAW 10% removal arms.
"""

from __future__ import annotations

import argparse
import csv
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from exp_config import ATTR_DIR, LOG_DIR, N_TRAIN, QUERY_DIR, ROOT, TRAIN_SEED


QUERY_IDS = tuple(range(60))
LAMBDA = 10.0
TOPK = 5000
METHOD = "das_ema"
METHOD_TAG = "das_ema_lambda10"
OUT_ROOT = ROOT / "topk_removal_das_ema_lambda10_60q_10pct"
RAW_PINT_ROOT = (
    ROOT
    / "topk_removal_retrac_endtracin_tracindas_100q"
    / "tracin_das_aligned_timestamp_square"
    / "remove_10pct"
)
ADAMW_PINT_ROOT = (
    ROOT
    / (
        "topk_removal_pint_endpoint_retrac_random_60q_10pct_"
        "final_endpoint_linear_adamw_sign"
    )
    / "pint_raw"
    / "remove_10pct"
)
METRICS = (
    "noise_trajectory_mse",
    "trajectory_mse",
    "endpoint_mse",
    "simple_loss_abs_change",
)


def prepare_jobs() -> list[dict]:
    manifest = json.loads((QUERY_DIR / "manifest.json").read_text())
    by_id = {int(item["query_id"]): item for item in manifest}
    missing_queries = sorted(set(QUERY_IDS) - set(by_id))
    if missing_queries:
        raise ValueError(f"query manifest is missing IDs: {missing_queries}")

    jobs = []
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for query_id in QUERY_IDS:
        query = by_id[query_id]
        score_path = (
            ATTR_DIR
            / METHOD
            / f"q{query_id:02d}"
            / "lambda_10p0"
            / "scores.npy"
        )
        if not score_path.is_file():
            raise FileNotFoundError(score_path)
        scores = np.asarray(np.load(score_path), dtype=np.float64).reshape(-1)
        if scores.shape != (N_TRAIN,):
            raise ValueError(f"{score_path}: shape={scores.shape}")
        if not np.isfinite(scores).all():
            raise ValueError(f"non-finite scores: {score_path}")

        # Original DAS and PINT both use the largest saved scores for removal.
        removed = np.argsort(-scores, kind="stable")[:TOPK].astype(np.int64)
        keep = np.ones(N_TRAIN, dtype=bool)
        keep[removed] = False
        kept = np.flatnonzero(keep).astype(np.int64)
        job_dir = OUT_ROOT / METHOD_TAG / "remove_10pct" / f"q{query_id:02d}"
        job_dir.mkdir(parents=True, exist_ok=True)
        np.save(job_dir / "removed_indices.npy", removed)
        np.save(job_dir / "kept_indices.npy", kept)
        np.save(job_dir / "removed_scores.npy", scores[removed])
        job = {
            "job_id": len(jobs),
            "query_id": query_id,
            "family": query["family"],
            "labels": query.get("labels", []),
            "initial_seed": int(query["initial_seed"]),
            "method_tag": METHOD_TAG,
            "method": METHOD,
            "score_param_source": "ema",
            "eval_param_source": "ema",
            "lambda": LAMBDA,
            "removal_fraction": 0.10,
            "topk": TOPK,
            "ranking": "descending_saved_score_without_lds_sign",
            "score_path": str(score_path),
            "removed_indices_path": str(job_dir / "removed_indices.npy"),
            "kept_indices_path": str(job_dir / "kept_indices.npy"),
            "job_dir": str(job_dir),
            "model_dir": str(job_dir / "model"),
            "train_seed": int(TRAIN_SEED),
        }
        (job_dir / "job.json").write_text(json.dumps(job, indent=2) + "\n")
        jobs.append(job)
    (OUT_ROOT / "jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
    print(f"[prepared] {len(jobs)} DAS-EMA lambda=10 Top-5000 jobs", flush=True)
    return jobs


def run_logged(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", buffering=1) as stream:
        stream.write("\n$ " + " ".join(command) + "\n")
        return subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT)


def load_evaluation(root: Path, query_id: int) -> dict:
    path = root / f"q{query_id:02d}" / "evaluation.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def summarize() -> None:
    arms = {
        "DAS-EMA lambda=10": OUT_ROOT / METHOD_TAG / "remove_10pct",
        "raw-gradient PINT / RAW": RAW_PINT_ROOT,
        "Full-AdamW PINT / RAW": ADAMW_PINT_ROOT,
    }
    values = {
        arm: {query_id: load_evaluation(root, query_id) for query_id in QUERY_IDS}
        for arm, root in arms.items()
    }
    baseline = "DAS-EMA lambda=10"
    lines = [
        "X3 TRUE 10% REMOVAL — DAS-EMA LAMBDA=10 VS PINT",
        "q00-q59; Top-5000/50000; evaluation parameters=EMA",
        "delta = comparison arm - DAS; positive means comparison arm is stronger",
        "",
    ]
    payload = {"query_ids": list(QUERY_IDS), "topk": TOPK, "metrics": {}}
    rows = []
    for metric in METRICS:
        lines.extend(
            [
                f"METRIC: {metric}",
                (
                    f"{'method':30s} {'mean +/- std':28s} {'delta-vs-DAS':>14s} "
                    f"{'ratio':>9s} {'wins':>8s} {'p-W':>12s}"
                ),
            ]
        )
        das = np.asarray(
            [values[baseline][query_id][metric] for query_id in QUERY_IDS],
            dtype=np.float64,
        )
        metric_payload = {}
        for arm in arms:
            current = np.asarray(
                [values[arm][query_id][metric] for query_id in QUERY_IDS],
                dtype=np.float64,
            )
            delta = current - das
            if arm == baseline:
                pvalue = None
                wins = None
            else:
                pvalue = float(wilcoxon(delta, alternative="two-sided").pvalue)
                wins = int(np.sum(delta > 0))
            result = {
                "mean": float(current.mean()),
                "std": float(current.std()),
                "mean_delta_vs_das": float(delta.mean()),
                "ratio_over_das": float(current.mean() / das.mean()),
                "wins_over_das": wins,
                "wilcoxon_two_sided_p": pvalue,
            }
            metric_payload[arm] = result
            wins_text = "--" if wins is None else f"{wins}/60"
            p_text = "--" if pvalue is None else f"{pvalue:.6g}"
            lines.append(
                f"{arm:30s} {current.mean():.6e} +/- {current.std():.3e} "
                f"{delta.mean():+14.3e} {current.mean()/das.mean():9.4f} "
                f"{wins_text:>8s} {p_text:>12s}"
            )
        payload["metrics"][metric] = metric_payload
        lines.append("")

    for query_id in QUERY_IDS:
        row = {"query_id": query_id}
        for arm in arms:
            for metric in METRICS:
                key = arm.lower().replace(" ", "_").replace("/", "").replace("=", "")
                row[f"{key}_{metric}"] = values[arm][query_id][metric]
        rows.append(row)

    text_path = OUT_ROOT / "das_ema_lambda10_vs_pint_10pct.txt"
    json_path = OUT_ROOT / "das_ema_lambda10_vs_pint_10pct.json"
    csv_path = OUT_ROOT / "das_ema_lambda10_vs_pint_10pct_per_query.csv"
    text_path.write_text("\n".join(lines) + "\n")
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\n".join(lines), flush=True)
    print(f"[saved] {text_path}")
    print(f"[saved] {json_path}")
    print(f"[saved] {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus or args.workers_per_gpu <= 0:
        raise ValueError("invalid GPU/worker configuration")

    if args.skip_prepare:
        jobs = json.loads((OUT_ROOT / "jobs.json").read_text())
    else:
        jobs = prepare_jobs()
    if len(jobs) != len(QUERY_IDS):
        raise ValueError(f"expected 60 jobs, found {len(jobs)}")

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
            query_id = int(job["query_id"])
            label = f"{METHOD_TAG}/q{query_id:02d}"
            job_dir = Path(job["job_dir"])
            log_path = LOG_DIR / "das_ema_lambda10_60q_10pct" / f"q{query_id:02d}.log"
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
                        f"finished={finished}/60 elapsed={elapsed/3600:.2f}h "
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
        f"[launcher] jobs=60 GPUs={gpus} workers_per_gpu={args.workers_per_gpu} "
        f"concurrent={len(threads)}",
        flush=True,
    )
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if state["failed"]:
        path = OUT_ROOT / "failed_jobs.json"
        path.write_text(json.dumps(state["failed"], indent=2) + "\n")
        raise SystemExit(f"{len(state['failed'])} jobs failed; see {path}")
    summarize()
    print("[done] DAS-EMA lambda=10 10% removal and PINT comparison", flush=True)


if __name__ == "__main__":
    main()
