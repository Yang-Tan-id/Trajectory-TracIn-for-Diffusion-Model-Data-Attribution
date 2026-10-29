"""Prepare q00-q59 10% removal jobs for five attribution methods and random."""

from __future__ import annotations

import json
from itertools import combinations

import numpy as np

from diffusion_retrac_config import RETRAC_ADAMW_FULL_METHOD
from endpoint_tracin_adamw_mc10_config import eta_method
from exp_config import ATTR_DIR, N_TRAIN, QUERY_DIR, ROOT, TRAIN_SEED
from tracin_das_norm_adam_config import tdna_method


QUERY_IDS = tuple(range(60))
REMOVAL_FRACTION = 0.10
TOPK = int(round(N_TRAIN * REMOVAL_FRACTION))
REMOVAL_FRACTION_TAG = "remove_10pct"
RANDOM_SEED_BASE = 60_000_000
OUT_ROOT = ROOT / (
    "topk_removal_pint_endpoint_retrac_random_60q_10pct_"
    "correct_direction_endpoint_linear"
)

# prediction_sign maps a retained-subset score sum to the LDS target.  Removal
# changes that prediction with the opposite sign, so the ranking quantity is
# -prediction_sign * saved_score.  In particular, LDS sign=-1 means removing
# the largest saved scores, exactly as in the earlier raw-gradient Top-K run.
METHODS = (
    {
        "tag": "pint_raw",
        "label": "PINT / RAW",
        "method": tdna_method("adamw_full", "raw", "timestamp_sum_squared"),
        "prediction_sign": -1,
        "description": "Full-AdamW PINT (TracIn-DAS), raw, timestamp-sum-squared",
    },
    {
        "tag": "pint_both_l2",
        "label": "PINT / BOTH-L2",
        "method": tdna_method(
            "adamw_full", "query_train_l2", "timestamp_sum_squared"
        ),
        "prediction_sign": -1,
        "description": (
            "Full-AdamW PINT (TracIn-DAS), query/train L2, "
            "timestamp-sum-squared"
        ),
    },
    {
        "tag": "endpoint_tracin_raw",
        "label": "ENDPOINT TRACIN / RAW",
        "method": eta_method("raw", "linear"),
        "prediction_sign": -1,
        "description": "Full-AdamW endpoint-loss TracIn, raw, linear",
    },
    {
        "tag": "endpoint_tracin_both_l2",
        "label": "ENDPOINT TRACIN / BOTH-L2",
        "method": eta_method("query_train_l2", "linear"),
        "prediction_sign": -1,
        "description": "Full-AdamW endpoint-loss TracIn, query/train L2, linear",
    },
    {
        "tag": "retrac_both_l2",
        "label": "RETRAC / BOTH-L2",
        "method": RETRAC_ADAMW_FULL_METHOD,
        "prediction_sign": +1,
        "description": "Full-AdamW Diffusion-ReTrac with query/train L2 normalization",
    },
    {
        "tag": "random",
        "label": "RANDOM REMOVAL",
        "method": None,
        "prediction_sign": None,
        "description": "query-specific deterministic uniform random 10% removal",
    },
)
EXPECTED_JOB_COUNT = len(QUERY_IDS) * len(METHODS)


def load_manifest() -> dict[int, dict]:
    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    by_id = {int(item["query_id"]): item for item in manifest}
    missing = sorted(set(QUERY_IDS) - set(by_id))
    if missing:
        raise ValueError(f"query manifest is missing IDs: {missing}")
    return by_id


def load_scores(spec: dict, query_id: int) -> tuple[np.ndarray, str]:
    path = ATTR_DIR / spec["method"] / f"q{query_id:02d}" / "scores.npy"
    if not path.is_file():
        raise FileNotFoundError(f"missing {spec['label']} q{query_id:02d}: {path}")
    scores = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
    if scores.shape != (N_TRAIN,):
        raise ValueError(f"{path} shape={scores.shape}, expected={(N_TRAIN,)}")
    if not np.isfinite(scores).all():
        raise ValueError(f"non-finite scores in {path}")
    return scores, str(path)


def main() -> None:
    by_id = load_manifest()

    # Fail before creating a partial job tree if any requested attribution is absent.
    score_cache: dict[tuple[str, int], tuple[np.ndarray, str]] = {}
    for spec in METHODS:
        if spec["method"] is None:
            continue
        for query_id in QUERY_IDS:
            score_cache[(spec["tag"], query_id)] = load_scores(spec, query_id)
    print("[preflight] all 300 attribution vectors verified", flush=True)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    jobs: list[dict] = []
    removed_sets: dict[tuple[str, int], set[int]] = {}
    for spec in METHODS:
        for query_id in QUERY_IDS:
            query = by_id[query_id]
            random_seed = None
            score_path = None
            if spec["method"] is None:
                random_seed = RANDOM_SEED_BASE + int(TRAIN_SEED) + query_id
                rng = np.random.default_rng(random_seed)
                removed = rng.choice(N_TRAIN, size=TOPK, replace=False).astype(np.int64)
                ranking_values = np.full(TOPK, np.nan, dtype=np.float64)
                ranking = f"uniform_without_replacement_seed_{random_seed}"
            else:
                scores, score_path = score_cache[(spec["tag"], query_id)]
                removal_scores = -float(spec["prediction_sign"]) * scores
                removed = np.argsort(-removal_scores, kind="stable")[:TOPK].astype(
                    np.int64
                )
                ranking_values = removal_scores[removed]
                ranking = (
                    "descending_negative_prediction_sign_times_saved_score;"
                    f"prediction_sign={int(spec['prediction_sign']):+d}"
                )

            keep_mask = np.ones(N_TRAIN, dtype=bool)
            keep_mask[removed] = False
            kept = np.flatnonzero(keep_mask).astype(np.int64)
            job_dir = OUT_ROOT / spec["tag"] / REMOVAL_FRACTION_TAG / f"q{query_id:02d}"
            job_dir.mkdir(parents=True, exist_ok=True)
            np.save(job_dir / "removed_indices.npy", removed)
            np.save(job_dir / "kept_indices.npy", kept)
            np.save(job_dir / "removed_oriented_scores.npy", ranking_values)

            finite_values = ranking_values[np.isfinite(ranking_values)]
            job = {
                "job_id": len(jobs),
                "query_id": query_id,
                "family": query["family"],
                "labels": query.get("labels", []),
                "initial_seed": int(query["initial_seed"]),
                "method_tag": spec["tag"],
                "method_label": spec["label"],
                "method": spec["method"] or "random_removal",
                "method_description": spec["description"],
                "prediction_sign": spec["prediction_sign"],
                "removal_score_definition": (
                    None
                    if spec["prediction_sign"] is None
                    else "-prediction_sign * saved_score"
                ),
                "ranking_saved_score_direction": (
                    None
                    if spec["prediction_sign"] is None
                    else (
                        "descending"
                        if int(spec["prediction_sign"]) == -1
                        else "ascending"
                    )
                ),
                "score_param_source": "raw",
                "eval_param_source": "ema",
                "lambda": None,
                "removal_fraction": REMOVAL_FRACTION,
                "removal_fraction_tag": REMOVAL_FRACTION_TAG,
                "topk": TOPK,
                "ranking": ranking,
                "random_seed": random_seed,
                "score_path": score_path,
                "removed_indices_path": str(job_dir / "removed_indices.npy"),
                "kept_indices_path": str(job_dir / "kept_indices.npy"),
                "job_dir": str(job_dir),
                "model_dir": str(job_dir / "model"),
                "train_seed": int(TRAIN_SEED),
                "removed_oriented_score_min": (
                    float(finite_values.min()) if finite_values.size else None
                ),
                "removed_oriented_score_max": (
                    float(finite_values.max()) if finite_values.size else None
                ),
                "removed_oriented_score_mean": (
                    float(finite_values.mean()) if finite_values.size else None
                ),
            }
            with open(job_dir / "job.json", "w") as handle:
                json.dump(job, handle, indent=2)
            jobs.append(job)
            removed_sets[(spec["tag"], query_id)] = set(removed.tolist())
            print(
                f"[prepared {len(jobs):03d}/{EXPECTED_JOB_COUNT}] {spec['tag']} "
                f"q{query_id:02d} removed={TOPK} kept={len(kept)}",
                flush=True,
            )

    overlaps = []
    for query_id in QUERY_IDS:
        for left, right in combinations(METHODS, 2):
            count = len(
                removed_sets[(left["tag"], query_id)]
                & removed_sets[(right["tag"], query_id)]
            )
            overlaps.append(
                {
                    "query_id": query_id,
                    "family": by_id[query_id]["family"],
                    "removal_fraction": REMOVAL_FRACTION,
                    "topk": TOPK,
                    "left_method": left["tag"],
                    "right_method": right["tag"],
                    "overlap_count": count,
                    "overlap_fraction": count / float(TOPK),
                }
            )

    with open(OUT_ROOT / "jobs.json", "w") as handle:
        json.dump(jobs, handle, indent=2)
    with open(OUT_ROOT / "method_overlap.json", "w") as handle:
        json.dump(overlaps, handle, indent=2)
    with open(OUT_ROOT / "contract.json", "w") as handle:
        json.dump(
            {
                "query_ids": list(QUERY_IDS),
                "removal_fraction": REMOVAL_FRACTION,
                "topk": TOPK,
                "methods": list(METHODS),
                "job_count": len(jobs),
                "evaluation_parameter_source": "ema",
                "random_control": "one deterministic random subset per query",
            },
            handle,
            indent=2,
        )
    if len(jobs) != EXPECTED_JOB_COUNT:
        raise RuntimeError(f"prepared {len(jobs)} jobs, expected {EXPECTED_JOB_COUNT}")
    print(f"[done] prepared {len(jobs)} jobs -> {OUT_ROOT / 'jobs.json'}", flush=True)


if __name__ == "__main__":
    main()
