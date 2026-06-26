from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

try:
    from .config_loader import load_config, require_attr
except ImportError:
    import sys

    refine_root = Path(__file__).resolve().parents[1]
    if str(refine_root) not in sys.path:
        sys.path.insert(0, str(refine_root))
    from common.config_loader import load_config, require_attr


def _split_paths(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _range_suffix(part: str) -> str:
    start_end = part.strip().replace(":", "-").split("-")
    if len(start_end) != 2:
        raise ValueError("ATTRIBUTION_RANGES must look like '1-2500,2501-5000'.")
    return f"range_{int(start_end[0])}_{int(start_end[1])}"


def _result_dirs(cfg, algorithm: str) -> list[Path]:
    explicit = os.environ.get("ATTRIBUTION_RESULT_DIRS")
    if explicit:
        return [Path(x) for x in _split_paths(explicit)]
    root = Path(require_attr(cfg, "ATTRIBUTION_ROOT"))
    base = root / f"{algorithm}_unprompted"
    ranges = os.environ.get("ATTRIBUTION_RANGES") or os.environ.get("SCORE_INDEX_RANGES")
    if ranges:
        parts = [p for p in ranges.replace(",", " ").split() if p]
        return [base.with_name(f"{base.name}_{_range_suffix(part)}") for part in parts]
    return [base]


def _load_scores(result_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(np.load(result_dir / "scores.npy"), dtype=np.float64).reshape(-1)
    indices = np.asarray(np.load(result_dir / "score_indices.npy"), dtype=np.int64).reshape(-1)
    if len(scores) != len(indices):
        raise ValueError(f"Length mismatch in {result_dir}")
    return indices, scores


def _combine(result_dirs: list[Path], duplicate_policy: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    values: dict[int, list[float]] = {}
    sources = []
    for result_dir in result_dirs:
        indices, scores = _load_scores(result_dir)
        sources.append({"result_dir": str(result_dir), "num_scores": int(len(scores))})
        for idx, score in zip(indices.tolist(), scores.tolist()):
            values.setdefault(int(idx), []).append(float(score))
    combined_indices = np.asarray(sorted(values), dtype=np.int64)
    if duplicate_policy == "max":
        combined_scores = np.asarray([max(values[int(i)]) for i in combined_indices], dtype=np.float64)
    elif duplicate_policy == "sum":
        combined_scores = np.asarray([sum(values[int(i)]) for i in combined_indices], dtype=np.float64)
    elif duplicate_policy == "mean":
        combined_scores = np.asarray([sum(values[int(i)]) / len(values[int(i)]) for i in combined_indices], dtype=np.float64)
    else:
        raise ValueError(f"Unknown duplicate policy: {duplicate_policy}")
    return combined_indices, combined_scores, sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Unprompted counterfactual removal-set eval.")
    parser.add_argument("config", type=str)
    parser.add_argument("--algorithm", default=os.environ.get("ALGORITHM", "das"))
    parser.add_argument("--topk", type=int, default=int(os.environ.get("TOPK", "5000")))
    parser.add_argument("--duplicate-policy", choices=["max", "sum", "mean"], default=os.environ.get("DUPLICATE_POLICY", "max"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    result_dirs = _result_dirs(cfg, args.algorithm)
    indices, scores, sources = _combine(result_dirs, args.duplicate_policy)
    if len(scores) == 0:
        raise RuntimeError("No unprompted scores found.")

    topk = min(int(args.topk), len(scores))
    order = np.argsort(-scores)[:topk]
    selected_indices = indices[order]
    selected_scores = scores[order]

    out_root = Path(require_attr(cfg, "EVAL_ROOT")) / "counterfactual_unprompted" / args.algorithm
    out_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "backend": "diffusers_unprompted",
        "metric": "counterfactual_removal_set",
        "algorithm": args.algorithm,
        "sources": sources,
        "duplicate_policy": args.duplicate_policy,
        "num_scores": int(len(scores)),
        "topk": int(topk),
        "mean_selected_score": float(np.mean(selected_scores)) if len(selected_scores) else None,
        "selected": [
            {"rank": rank, "idx": int(idx), "idx_1based": int(idx) + 1, "score": float(score)}
            for rank, (idx, score) in enumerate(zip(selected_indices.tolist(), selected_scores.tolist()), start=1)
        ],
        "note": (
            "This unprompted counterfactual backend creates the removal set from diffusers scores. "
            "Retraining-after-removal for diffusers can be added on top of this file."
        ),
    }
    with open(out_root / "counterfactual_unprompted.json", "w") as f:
        json.dump(payload, f, indent=2)
    np.save(out_root / "removed_attribution_indices.npy", selected_indices.astype(np.int64))
    np.save(out_root / "removed_scores.npy", selected_scores.astype(np.float64))
    print(f"Saved unprompted counterfactual eval to {out_root}")


if __name__ == "__main__":
    main()
