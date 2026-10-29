"""Prepare the raw-gradient PINT arm for the q00-q59 10% removal study."""

from __future__ import annotations

import json

import numpy as np

from exp_config import ATTR_DIR, N_TRAIN, QUERY_DIR, ROOT, TRAIN_SEED
from tracin_das_norm_adam_config import tdna_method


QUERY_IDS = tuple(range(60))
REMOVAL_FRACTION = 0.10
TOPK = int(round(N_TRAIN * REMOVAL_FRACTION))
REMOVAL_FRACTION_TAG = "remove_10pct"
METHOD_TAG = "pint_raw_gradient_raw"
METHOD_LABEL = "PINT / RAW-GRADIENT / RAW"
METHOD = tdna_method("gradient", "raw", "timestamp_sum_squared")
PREDICTION_SIGN = -1
OUT_ROOT = ROOT / "topk_removal_pint_endpoint_retrac_random_60q_10pct"
JOBS_PATH = OUT_ROOT / "raw_gradient_jobs.json"
EXPECTED_JOB_COUNT = len(QUERY_IDS)


def main() -> None:
    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    by_id = {int(item["query_id"]): item for item in manifest}
    missing_queries = sorted(set(QUERY_IDS) - set(by_id))
    if missing_queries:
        raise ValueError(f"query manifest is missing IDs: {missing_queries}")

    cached: dict[int, tuple[np.ndarray, str]] = {}
    for query_id in QUERY_IDS:
        path = ATTR_DIR / METHOD / f"q{query_id:02d}" / "scores.npy"
        if not path.is_file():
            raise FileNotFoundError(f"missing q{query_id:02d}: {path}")
        scores = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
        if scores.shape != (N_TRAIN,):
            raise ValueError(f"{path} shape={scores.shape}, expected={(N_TRAIN,)}")
        if not np.isfinite(scores).all():
            raise ValueError(f"non-finite scores in {path}")
        cached[query_id] = (scores, str(path))
    print("[preflight] all 60 raw-gradient PINT vectors verified", flush=True)

    jobs = []
    overlaps = []
    for query_id in QUERY_IDS:
        query = by_id[query_id]
        scores, score_path = cached[query_id]
        oriented = PREDICTION_SIGN * scores
        removed = np.argsort(-oriented, kind="stable")[:TOPK].astype(np.int64)
        keep_mask = np.ones(N_TRAIN, dtype=bool)
        keep_mask[removed] = False
        kept = np.flatnonzero(keep_mask).astype(np.int64)
        job_dir = OUT_ROOT / METHOD_TAG / REMOVAL_FRACTION_TAG / f"q{query_id:02d}"
        job_dir.mkdir(parents=True, exist_ok=True)
        np.save(job_dir / "removed_indices.npy", removed)
        np.save(job_dir / "kept_indices.npy", kept)
        np.save(job_dir / "removed_oriented_scores.npy", oriented[removed])

        adamw_removed_path = (
            OUT_ROOT
            / "pint_raw"
            / REMOVAL_FRACTION_TAG
            / f"q{query_id:02d}"
            / "removed_indices.npy"
        )
        if adamw_removed_path.is_file():
            adamw_removed = np.load(adamw_removed_path).astype(np.int64)
            overlap_count = int(np.intersect1d(removed, adamw_removed).size)
            overlaps.append(
                {
                    "query_id": query_id,
                    "overlap_count": overlap_count,
                    "overlap_fraction": overlap_count / float(TOPK),
                }
            )

        job = {
            "job_id": len(jobs),
            "query_id": query_id,
            "family": query["family"],
            "labels": query.get("labels", []),
            "initial_seed": int(query["initial_seed"]),
            "method_tag": METHOD_TAG,
            "method_label": METHOD_LABEL,
            "method": METHOD,
            "method_description": (
                "Raw-gradient PINT (TracIn-DAS), raw normalization, "
                "timestamp-sum-squared"
            ),
            "prediction_sign": PREDICTION_SIGN,
            "score_param_source": "raw",
            "eval_param_source": "ema",
            "lambda": None,
            "removal_fraction": REMOVAL_FRACTION,
            "removal_fraction_tag": REMOVAL_FRACTION_TAG,
            "topk": TOPK,
            "ranking": "descending_negative_saved_score",
            "random_seed": None,
            "score_path": score_path,
            "removed_indices_path": str(job_dir / "removed_indices.npy"),
            "kept_indices_path": str(job_dir / "kept_indices.npy"),
            "job_dir": str(job_dir),
            "model_dir": str(job_dir / "model"),
            "train_seed": int(TRAIN_SEED),
            "removed_oriented_score_min": float(oriented[removed].min()),
            "removed_oriented_score_max": float(oriented[removed].max()),
            "removed_oriented_score_mean": float(oriented[removed].mean()),
        }
        with open(job_dir / "job.json", "w") as handle:
            json.dump(job, handle, indent=2)
        jobs.append(job)
        print(
            f"[prepared {len(jobs):02d}/{EXPECTED_JOB_COUNT}] {METHOD_TAG} "
            f"q{query_id:02d} removed={TOPK}",
            flush=True,
        )

    with open(JOBS_PATH, "w") as handle:
        json.dump(jobs, handle, indent=2)
    with open(OUT_ROOT / "pint_raw_gradient_vs_adamw_topk_overlap.json", "w") as handle:
        json.dump(overlaps, handle, indent=2)
    print(f"[done] prepared {len(jobs)} jobs -> {JOBS_PATH}", flush=True)


if __name__ == "__main__":
    main()
