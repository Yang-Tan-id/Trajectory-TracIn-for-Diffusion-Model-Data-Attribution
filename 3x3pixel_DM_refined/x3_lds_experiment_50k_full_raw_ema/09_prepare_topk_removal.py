"""Prepare 200 query-specific top-1000 removal/retrain jobs."""

import json
from pathlib import Path

import numpy as np

from exp_config import ATTR_DIR, N_TRAIN, QUERY_DIR, ROOT, TRAIN_SEED


TOPK = 1000
OUT_ROOT = ROOT / "topk_removal_retrain"
METHODS = (
    {
        "tag": "traj_first_raw_linear",
        "method": "traj_projected_first_raw_linear",
        "score_param_source": "raw",
        "eval_param_source": "ema",
        "lambda": None,
    },
    {
        "tag": "das_ema_lambda_10",
        "method": "das_ema",
        "score_param_source": "ema",
        "eval_param_source": "ema",
        "lambda": 10.0,
    },
)


def score_path(spec, query_id):
    root = ATTR_DIR / spec["method"] / f"q{query_id:02d}"
    if spec["lambda"] is None:
        return root / "scores.npy"
    tag = str(float(spec["lambda"])).replace(".", "p")
    return root / f"lambda_{tag}" / "scores.npy"


def main():
    with open(QUERY_DIR / "manifest.json") as handle:
        queries = json.load(handle)
    if len(queries) != 100:
        raise ValueError(f"expected 100 queries, found {len(queries)}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = []
    removed_by_query = {}
    for spec in METHODS:
        for query in queries:
            query_id = int(query["query_id"])
            path = score_path(spec, query_id)
            if not path.is_file():
                raise FileNotFoundError(f"missing attribution scores: {path}")
            scores = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
            if scores.shape != (N_TRAIN,):
                raise ValueError(f"{path} has shape {scores.shape}, expected {(N_TRAIN,)}")
            if not np.isfinite(scores).all():
                raise ValueError(f"{path} contains non-finite scores")

            # The requested removal convention is the score itself: descending
            # raw scores, with no LDS prediction-sign multiplication.
            removed = np.argsort(-scores, kind="stable")[:TOPK].astype(np.int64)
            keep_mask = np.ones(N_TRAIN, dtype=bool)
            keep_mask[removed] = False
            kept = np.flatnonzero(keep_mask).astype(np.int64)

            job_dir = OUT_ROOT / spec["tag"] / f"q{query_id:02d}"
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
                "method_tag": spec["tag"],
                "method": spec["method"],
                "score_param_source": spec["score_param_source"],
                "eval_param_source": spec["eval_param_source"],
                "lambda": spec["lambda"],
                "topk": TOPK,
                "ranking": "descending_raw_score_without_lds_sign",
                "score_path": str(path),
                "removed_indices_path": str(job_dir / "removed_indices.npy"),
                "kept_indices_path": str(job_dir / "kept_indices.npy"),
                "job_dir": str(job_dir),
                "model_dir": str(job_dir / "model"),
                "train_seed": int(TRAIN_SEED),
                "removed_score_min": float(scores[removed].min()),
                "removed_score_max": float(scores[removed].max()),
                "removed_score_mean": float(scores[removed].mean()),
            }
            with open(job_dir / "job.json", "w") as handle:
                json.dump(job, handle, indent=2)
            jobs.append(job)
            removed_by_query[(spec["tag"], query_id)] = set(removed.tolist())
            print(
                f"[prepared] {spec['tag']} q{query_id:02d} {query['family']} "
                f"removed={len(removed)} kept={len(kept)} "
                f"score=[{scores[removed].min():.6g}, {scores[removed].max():.6g}]",
                flush=True,
            )

    overlaps = []
    first_tag, second_tag = METHODS[0]["tag"], METHODS[1]["tag"]
    for query in queries:
        query_id = int(query["query_id"])
        overlap = len(
            removed_by_query[(first_tag, query_id)]
            & removed_by_query[(second_tag, query_id)]
        )
        overlaps.append({"query_id": query_id, "overlap": overlap, "fraction": overlap / TOPK})

    with open(OUT_ROOT / "jobs.json", "w") as handle:
        json.dump(jobs, handle, indent=2)
    with open(OUT_ROOT / "method_overlap.json", "w") as handle:
        json.dump(overlaps, handle, indent=2)
    print(f"[done] prepared {len(jobs)} jobs -> {OUT_ROOT / 'jobs.json'}", flush=True)


if __name__ == "__main__":
    main()
