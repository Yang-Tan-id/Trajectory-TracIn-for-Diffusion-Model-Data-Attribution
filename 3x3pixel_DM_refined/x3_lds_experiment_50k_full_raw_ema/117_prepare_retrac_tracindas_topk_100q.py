"""Prepare 900 top-k removal jobs for ReTrac, end TracIn, and TracIn-DAS."""

import json

import numpy as np

from diffusion_retrac_config import RETRAC_METHOD, RETRAC_TRACIN_METHOD
from exp_config import ATTR_DIR, N_TRAIN, QUERY_DIR, ROOT, TRAIN_SEED
from tracin_das_config import tracin_das_methods


OUT_ROOT = ROOT / "topk_removal_retrac_endtracin_tracindas_100q"
REMOVAL_FRACTIONS = (0.02, 0.05, 0.10)
METHODS = (
    {
        "tag": "diffusion_retrac_normalized",
        "method": RETRAC_METHOD,
        "description": "Diffusion-ReTrac with full-gradient L2 normalization",
    },
    {
        "tag": "end_tracin_unnormalized",
        "method": RETRAC_TRACIN_METHOD,
        "description": "same endpoint diffusion replay as ReTrac, without normalization",
    },
    {
        "tag": "tracin_das_aligned_timestamp_square",
        "method": tracin_das_methods(
            "checkpoint", "projected4096", "aligned"
        )["timestamp_sum_squared"],
        "description": (
            "checkpoint-noise projected4096 aligned TracIn-DAS, "
            "timestamp-sum-squared contraction"
        ),
    },
)


def fraction_tag(value):
    return f"remove_{int(round(100 * float(value))):02d}pct"


def main():
    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    by_id = {int(item["query_id"]): item for item in manifest}
    if sorted(by_id) != list(range(100)):
        raise ValueError("expected q00-q99 in query manifest")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = []
    removed_sets = {}
    for fraction in REMOVAL_FRACTIONS:
        topk = int(round(N_TRAIN * fraction))
        fraction_name = fraction_tag(fraction)
        for spec in METHODS:
            for query_id in range(100):
                query = by_id[query_id]
                score_path = ATTR_DIR / spec["method"] / f"q{query_id:02d}" / "scores.npy"
                if not score_path.is_file():
                    raise FileNotFoundError(
                        f"missing {spec['tag']} q{query_id:02d}: {score_path}"
                    )
                scores = np.asarray(np.load(score_path), dtype=np.float64).reshape(-1)
                if scores.shape != (N_TRAIN,):
                    raise ValueError(
                        f"{score_path} shape={scores.shape}, expected={(N_TRAIN,)}"
                    )
                if not np.isfinite(scores).all():
                    raise ValueError(f"non-finite scores in {score_path}")
                removed = np.argsort(-scores, kind="stable")[:topk].astype(np.int64)
                keep_mask = np.ones(N_TRAIN, dtype=bool)
                keep_mask[removed] = False
                kept = np.flatnonzero(keep_mask).astype(np.int64)
                job_dir = (
                    OUT_ROOT / spec["tag"] / fraction_name / f"q{query_id:02d}"
                )
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
                    "method_description": spec["description"],
                    "score_param_source": "raw",
                    "eval_param_source": "ema",
                    "lambda": None,
                    "removal_fraction": float(fraction),
                    "removal_fraction_tag": fraction_name,
                    "topk": topk,
                    "ranking": "descending_saved_score_without_lds_sign",
                    "score_path": str(score_path),
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
                removed_sets[(fraction_name, spec["tag"], query_id)] = set(
                    removed.tolist()
                )
                print(
                    f"[prepared] {fraction_name} {spec['tag']} q{query_id:02d} "
                    f"{query['family']} removed={topk} kept={len(kept)} "
                    f"score=[{scores[removed].min():.6g},{scores[removed].max():.6g}]",
                    flush=True,
                )

    overlaps = []
    for fraction in REMOVAL_FRACTIONS:
        fraction_name = fraction_tag(fraction)
        topk = int(round(N_TRAIN * fraction))
        for query_id in range(100):
            for left_index in range(len(METHODS)):
                for right_index in range(left_index + 1, len(METHODS)):
                    left = METHODS[left_index]["tag"]
                    right = METHODS[right_index]["tag"]
                    count = len(
                        removed_sets[(fraction_name, left, query_id)]
                        & removed_sets[(fraction_name, right, query_id)]
                    )
                    overlaps.append(
                        {
                            "query_id": query_id,
                            "family": by_id[query_id]["family"],
                            "removal_fraction": float(fraction),
                            "topk": topk,
                            "left_method": left,
                            "right_method": right,
                            "overlap_count": count,
                            "overlap_fraction": count / float(topk),
                        }
                    )
    with open(OUT_ROOT / "jobs.json", "w") as handle:
        json.dump(jobs, handle, indent=2)
    with open(OUT_ROOT / "method_overlap.json", "w") as handle:
        json.dump(overlaps, handle, indent=2)
    with open(OUT_ROOT / "contract.json", "w") as handle:
        json.dump(
            {
                "query_ids": list(range(100)),
                "removal_fractions": list(REMOVAL_FRACTIONS),
                "topk_counts": [int(round(N_TRAIN * f)) for f in REMOVAL_FRACTIONS],
                "methods": list(METHODS),
                "job_count": len(jobs),
                "ranking": "descending_saved_score_without_lds_sign",
                "evaluation_parameter_source": "ema",
            },
            handle,
            indent=2,
        )
    print(f"[done] prepared {len(jobs)} jobs -> {OUT_ROOT / 'jobs.json'}", flush=True)


if __name__ == "__main__":
    main()
