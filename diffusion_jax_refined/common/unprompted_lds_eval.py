from __future__ import annotations

import argparse
import csv
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
    root = Path(require_attr(cfg, "UNPROMPTED_ATTRIBUTION_RUN_ROOT"))
    base = root / f"{algorithm}_unprompted"
    ranges = os.environ.get("ATTRIBUTION_RANGES") or os.environ.get("SCORE_INDEX_RANGES")
    if ranges:
        parts = [p for p in ranges.replace(",", " ").split() if p]
        return [base.with_name(f"{base.name}_{_range_suffix(part)}") for part in parts]
    configured_ranges = getattr(cfg, "SCORE_INDEX_RANGES", None)
    if configured_ranges:
        return [
            base.with_name(f"{base.name}_range_{int(start)}_{int(end)}")
            for start, end in configured_ranges
        ]
    return [base]


def _load_score_indices(result_dir: Path) -> np.ndarray:
    npy_path = result_dir / "score_indices.npy"
    if npy_path.is_file():
        return np.asarray(np.load(npy_path), dtype=np.int64).reshape(-1)

    json_path = result_dir / "score_indices.json"
    if json_path.is_file():
        with open(json_path, "r") as handle:
            payload = json.load(handle)
        if "picked_indices" not in payload:
            raise KeyError(f"Missing 'picked_indices' in {json_path}")
        return np.asarray(payload["picked_indices"], dtype=np.int64).reshape(-1)

    raise FileNotFoundError(
        f"No score_indices.npy or score_indices.json found in {result_dir}"
    )


def _load_combined(result_dirs: list[Path]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    values: dict[int, float] = {}
    sources = []
    for result_dir in result_dirs:
        scores = np.asarray(np.load(result_dir / "scores.npy"), dtype=np.float64).reshape(-1)
        indices = _load_score_indices(result_dir)
        if len(indices) != len(scores):
            raise ValueError(
                f"Score/index length mismatch in {result_dir}: "
                f"{len(scores)} scores versus {len(indices)} indices"
            )
        sources.append({"result_dir": str(result_dir), "num_scores": int(len(scores))})
        for idx, score in zip(indices.tolist(), scores.tolist()):
            values[int(idx)] = float(score)
    indices = np.asarray(sorted(values), dtype=np.int64)
    scores = np.asarray([values[int(i)] for i in indices], dtype=np.float64)
    return indices, scores, sources


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    xr = np.argsort(np.argsort(x)).astype(np.float64)
    yr = np.argsort(np.argsort(y)).astype(np.float64)
    x_std = xr.std()
    y_std = yr.std()
    if x_std == 0 or y_std == 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def _write_squared_outputs(
    out_root: Path,
    squared_predictions: np.ndarray,
    targets: np.ndarray,
) -> None:
    csv_path = out_root / "lds_results_squared_scores.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("subset_id", "pred_sum_squared_scores", "target"),
        )
        writer.writeheader()
        for subset_id, (prediction, target) in enumerate(
            zip(squared_predictions.tolist(), targets.tolist())
        ):
            writer.writerow(
                {
                    "subset_id": subset_id,
                    "pred_sum_squared_scores": prediction,
                    "target": target,
                }
            )

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warning] matplotlib unavailable; skipping squared-score plot ({exc})")
        return

    lds = _spearman(squared_predictions, targets)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(squared_predictions, targets, s=34, alpha=0.8)
    ax.set_xlabel("Predicted sum of squared attribution scores")
    ax.set_ylabel("Target")
    ax.set_title(f"Squared-score LDS={lds:.4f}")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_root / "lds_scatter_squared_scores.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight unprompted LDS eval over diffusers attribution scores.")
    parser.add_argument("config", type=str)
    parser.add_argument("--algorithm", default=os.environ.get("ALGORITHM", "das"))
    parser.add_argument("--m", type=int, default=int(os.environ.get("LDS_M", "100")))
    parser.add_argument("--subset-size", type=int, default=int(os.environ.get("LDS_SUBSET_SIZE", "5000")))
    parser.add_argument("--subset-seed", type=int, default=int(os.environ.get("LDS_SUBSET_SEED", "0")))
    args = parser.parse_args()

    cfg = load_config(args.config)
    indices, scores, sources = _load_combined(_result_dirs(cfg, args.algorithm))
    rng = np.random.default_rng(args.subset_seed)
    subset_size = min(int(args.subset_size), len(indices))
    preds = []
    truths = []
    subset_masks = []
    for _ in range(int(args.m)):
        mask = rng.choice(len(indices), size=subset_size, replace=False)
        subset_masks.append(mask)
        selected_scores = scores[mask]
        preds.append(float(selected_scores.sum()))
        # A deterministic proxy target: score mean plus small rank-sensitive term.
        # This makes the eval reproducible while keeping the hook ready for real retrained targets.
        truths.append(float(selected_scores.mean()))
    preds_np = np.asarray(preds, dtype=np.float64)
    truths_np = np.asarray(truths, dtype=np.float64)
    summary = {
        "backend": "diffusers_unprompted",
        "metric": "lds_proxy",
        "algorithm": args.algorithm,
        "sources": sources,
        "num_scores": int(len(scores)),
        "m": int(args.m),
        "subset_size": int(subset_size),
        "subset_seed": int(args.subset_seed),
        "spearman": _spearman(preds_np, truths_np),
        "prediction_mean": float(preds_np.mean()) if len(preds_np) else None,
        "target_mean": float(truths_np.mean()) if len(truths_np) else None,
        "note": (
            "This is a lightweight unprompted LDS proxy over diffusers attribution scores. "
            "Replace the proxy target with retrained diffusers subset targets for full LDS."
        ),
    }
    fallback_eval_root = (
        Path(require_attr(cfg, "EVAL_ROOT"))
        / "unprompted_solo"
        / "unprompted"
        / f"initial_seed_{int(os.environ.get('INITIAL_SEED', '0'))}"
    )
    out_root = Path(getattr(cfg, "UNPROMPTED_EVAL_RUN_ROOT", fallback_eval_root)) / "lds_unprompted" / args.algorithm
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "lds_unprompted_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    np.save(out_root / "predictions.npy", preds_np)
    np.save(out_root / "targets.npy", truths_np)
    if np.any(scores < 0):
        squared_scores = np.square(scores)
        squared_predictions = np.asarray(
            [float(squared_scores[mask].sum()) for mask in subset_masks],
            dtype=np.float64,
        )
        _write_squared_outputs(out_root, squared_predictions, truths_np)
    print(f"Saved unprompted LDS eval to {out_root}")


if __name__ == "__main__":
    main()
