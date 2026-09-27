"""Prepare q00-q09 timestamp-sum-squared Traj top-1000 removal jobs."""

import json

import numpy as np

from exp_config import ATTR_DIR, N_TRAIN, QUERY_DIR, ROOT, TRAIN_SEED


TOPK = 1000
QUERY_IDS = tuple(range(10))
METHOD = "traj_projected_first_raw_timestamp_sum_squared"
METHOD_TAG = "traj_projected_first_raw_timestamp_sum_squared"
OUT_ROOT = ROOT / "topk_removal_traj_first_raw_timestamp_square_q00_q09"


def main():
    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    by_id = {int(query["query_id"]): query for query in manifest}
    queries = [by_id[query_id] for query_id in QUERY_IDS]
    if any(query["family"] != "prompted" for query in queries):
        raise ValueError("q00-q09 must all be prompted queries")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for query in queries:
        query_id = int(query["query_id"])
        path = ATTR_DIR / METHOD / f"q{query_id:02d}" / "scores.npy"
        if not path.is_file():
            raise FileNotFoundError(f"missing attribution scores: {path}")
        scores = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
        if scores.shape != (N_TRAIN,):
            raise ValueError(f"{path} has shape {scores.shape}, expected {(N_TRAIN,)}")
        if not np.isfinite(scores).all():
            raise ValueError(f"{path} contains non-finite scores")

        removed = np.argsort(-scores, kind="stable")[:TOPK].astype(np.int64)
        keep_mask = np.ones(N_TRAIN, dtype=bool)
        keep_mask[removed] = False
        kept = np.flatnonzero(keep_mask).astype(np.int64)
        job_dir = OUT_ROOT / METHOD_TAG / f"q{query_id:02d}"
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
            "score_param_source": "raw",
            "eval_param_source": "ema",
            "lambda": None,
            "topk": TOPK,
            "ranking": "descending_saved_score_without_lds_sign",
            "source_lds_file": (
                "traj_projected_first_raw_timestamp_sum_squared_"
                "traj_ref_raw.json"
            ),
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
        print(
            f"[prepared] {METHOD_TAG} q{query_id:02d} "
            f"removed={len(removed)} kept={len(kept)} "
            f"score=[{scores[removed].min():.6g}, "
            f"{scores[removed].max():.6g}]",
            flush=True,
        )

    with open(OUT_ROOT / "jobs.json", "w") as handle:
        json.dump(jobs, handle, indent=2)
    print(f"[done] prepared {len(jobs)} jobs -> {OUT_ROOT / 'jobs.json'}", flush=True)


if __name__ == "__main__":
    main()
