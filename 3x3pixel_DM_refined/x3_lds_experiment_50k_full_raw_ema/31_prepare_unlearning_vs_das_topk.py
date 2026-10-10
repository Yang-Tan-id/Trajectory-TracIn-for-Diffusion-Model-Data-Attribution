"""Prepare q00-q49 top-1000 removal jobs for unlearning alpha=.25 vs DAS."""

import json

import numpy as np

from exp_config import ATTR_DIR, N_TRAIN, QUERY_DIR, ROOT, TRAIN_SEED


TOPK = 1000
QUERY_IDS = tuple(range(50))
OUT_ROOT = ROOT / "topk_removal_unlearning_alpha0p25_vs_das_q00_q49"
UNLEARNING_METHOD = (
    "trajectory_unlearning_normalized_sgd_4step_50ckpt_1000t_4event_alpha_0p25"
)
METHODS = (
    {
        "tag": "unlearning_normalized_alpha_0p25",
        "method": UNLEARNING_METHOD,
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
    lambda_tag = str(float(spec["lambda"])).replace(".", "p")
    return root / f"lambda_{lambda_tag}" / "scores.npy"


def main():
    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    queries = {int(query["query_id"]): query for query in manifest}
    selected = [queries[query_id] for query_id in QUERY_IDS]
    if any(query["family"] != "prompted" for query in selected):
        raise ValueError("q00-q49 were expected to be prompted queries")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = []
    removed_sets = {}
    for spec in METHODS:
        for query in selected:
            query_id = int(query["query_id"])
            path = score_path(spec, query_id)
            if not path.is_file():
                raise FileNotFoundError(f"missing attribution scores: {path}")
            scores = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
            if scores.shape != (N_TRAIN,):
                raise ValueError(f"{path} shape={scores.shape}, expected={(N_TRAIN,)}")
            if not np.isfinite(scores).all():
                raise ValueError(f"{path} contains non-finite scores")

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
                "ranking": "descending_saved_score_without_lds_sign",
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
            removed_sets[(spec["tag"], query_id)] = set(removed.tolist())
            print(
                f"[prepared] {spec['tag']} q{query_id:02d} removed={TOPK} "
                f"score=[{scores[removed].min():.6g}, {scores[removed].max():.6g}]",
                flush=True,
            )

    left_tag, right_tag = METHODS[0]["tag"], METHODS[1]["tag"]
    overlaps = []
    for query_id in QUERY_IDS:
        count = len(
            removed_sets[(left_tag, query_id)]
            & removed_sets[(right_tag, query_id)]
        )
        overlaps.append(
            {"query_id": query_id, "overlap": count, "fraction": count / TOPK}
        )
    with open(OUT_ROOT / "jobs.json", "w") as handle:
        json.dump(jobs, handle, indent=2)
    with open(OUT_ROOT / "method_overlap.json", "w") as handle:
        json.dump(overlaps, handle, indent=2)
    print(f"[done] prepared {len(jobs)} jobs -> {OUT_ROOT / 'jobs.json'}", flush=True)


if __name__ == "__main__":
    main()
