from __future__ import annotations

"""Evaluate full-dim Traj-TracIn term scores with post-hoc term reweighting.

This script reuses full_dim_term_scores.npz artifacts and cached LDS targets.
It does not recompute gradients. A predicted-noise-change table can be supplied
as a 50x100/49x100 checkpoint-by-snapshot array and will modulate the already
learning-rate-weighted term contributions.
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def split_paths(text: str) -> list[Path]:
    paths: list[Path] = []
    for item in text.split(","):
        item = item.strip()
        if item:
            paths.append(Path(item).expanduser())
    if not paths:
        raise ValueError("No --term-score-npz paths were provided")
    return paths


def parse_rewrites(items: list[str]) -> list[tuple[str, str]]:
    rewrites = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"--subset-dir-rewrite must look like OLD=NEW, got {item!r}")
        old, new = item.split("=", 1)
        if not old:
            raise ValueError(f"Empty OLD prefix in rewrite {item!r}")
        rewrites.append((old.rstrip("/"), new.rstrip("/")))
    return rewrites


def rewrite_path(path_text: str, rewrites: list[tuple[str, str]]) -> str:
    for old, new in rewrites:
        if path_text == old or path_text.startswith(old + "/"):
            return new + path_text[len(old) :]
    return path_text


def rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman_corr(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    true = np.asarray(true, dtype=np.float64).reshape(-1)
    mask = np.isfinite(pred) & np.isfinite(true)
    if int(mask.sum()) < 2:
        return float("nan")
    pred_rank = rankdata_average(pred[mask])
    true_rank = rankdata_average(true[mask])
    pred_std = float(pred_rank.std())
    true_std = float(true_rank.std())
    if pred_std == 0.0 or true_std == 0.0:
        return float("nan")
    return float(np.corrcoef(pred_rank, true_rank)[0, 1])


def plot_scatter(path: Path, pred: np.ndarray, true: np.ndarray, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(5, 4), dpi=160)
    ax.scatter(pred, true, s=16, alpha=0.8)
    ax.set_xlabel("pred_sum_tau")
    ax.set_ylabel("true_f")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def infer_target_function(rows: list[dict[str, str]], default: str) -> str:
    if not rows:
        return default
    source_dir = rows[0].get("source_dir") or ""
    parts = Path(source_dir).parts
    for target in ("noise_trajectory", "projected_trajectory", "simple_loss"):
        if target in parts:
            return target
    return default


def prediction_indices(subset_dir: Path, subset: str) -> np.ndarray:
    filename = "kept_attribution_indices.npy" if subset == "kept" else "excluded_attribution_indices.npy"
    return np.load(subset_dir / filename)


def load_delta_or_weight(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        for key in (
            "change_weight_by_ckpt_snapshot",
            "change_weights_by_ckpt_snapshot",
            "delta_by_ckpt_snapshot",
            "deltas_by_ckpt_snapshot",
            "weights",
            "deltas",
        ):
            if key in payload:
                return np.asarray(payload[key], dtype=np.float64)
    raise ValueError(
        f"{path} does not contain a recognized weight/delta key. "
        "Expected change_weight_by_ckpt_snapshot, delta_by_ckpt_snapshot, weights, or deltas."
    )


def normalize_delta(delta: np.ndarray, mode: str, eps: float, clip: tuple[float, float] | None) -> np.ndarray:
    arr = np.asarray(delta, dtype=np.float64)
    if mode == "as_is":
        weights = arr
    elif mode == "global_linear":
        weights = arr / max(float(np.nanmean(arr)), eps)
    elif mode == "per_timestamp_linear":
        if arr.ndim != 2:
            raise ValueError("per_timestamp_linear requires a 2D checkpoint-by-snapshot delta/weight table")
        denom = np.nanmean(arr, axis=0, keepdims=True)
        weights = arr / np.maximum(denom, eps)
    else:
        raise ValueError(f"Unknown --weight-normalization {mode!r}")
    weights = np.where(np.isfinite(weights), weights, 0.0)
    if clip is not None:
        weights = np.clip(weights, clip[0], clip[1])
    return weights


def expand_ckpt_snapshot_weights(
    table: np.ndarray,
    ckpt_indices: np.ndarray,
    snapshot_positions: np.ndarray,
) -> np.ndarray:
    if table.ndim == 1:
        if len(table) != len(ckpt_indices):
            raise ValueError(f"1D weight table length {len(table)} does not match num terms {len(ckpt_indices)}")
        return table.astype(np.float64)
    if table.ndim != 2:
        raise ValueError(f"Weight table must be 1D or 2D, got shape {table.shape}")

    max_ckpt = int(np.max(ckpt_indices))
    max_snapshot = int(np.max(snapshot_positions))
    if table.shape[1] <= max_snapshot:
        raise ValueError(f"Weight table has {table.shape[1]} snapshots but term metadata needs {max_snapshot + 1}")
    if table.shape[0] <= max_ckpt:
        if table.shape[0] == max_ckpt:
            padded = np.concatenate([table, table[-1:, :]], axis=0)
            table = padded
        else:
            raise ValueError(f"Weight table has {table.shape[0]} checkpoints but term metadata needs {max_ckpt + 1}")
    return table[ckpt_indices.astype(np.int64), snapshot_positions.astype(np.int64)].astype(np.float64)


def score_map_from_term_artifacts(
    paths: list[Path],
    *,
    score_key: str,
    term_weight: np.ndarray | None,
) -> tuple[dict[int, float], dict[str, object]]:
    score_map: dict[int, float] = {}
    term_meta: dict[str, object] | None = None
    sources = []

    for path in paths:
        with np.load(path, allow_pickle=True) as payload:
            if score_key not in payload:
                raise ValueError(f"{path} does not contain {score_key!r}; available keys={payload.files}")
            scores_by_term = np.asarray(payload[score_key], dtype=np.float64)
            if scores_by_term.ndim != 2:
                raise ValueError(f"{path}:{score_key} must be [terms, points], got {scores_by_term.shape}")
            indices = np.asarray(payload["score_indices"], dtype=np.int64).reshape(-1)
            if scores_by_term.shape[1] != len(indices):
                raise ValueError(
                    f"{path}: score columns {scores_by_term.shape[1]} do not match score_indices {len(indices)}"
                )

            ckpt_indices = np.asarray(payload["term_ckpt_indices"], dtype=np.int32).reshape(-1)
            snapshot_positions = np.asarray(payload["term_snapshot_positions"], dtype=np.int32).reshape(-1)
            term_timesteps = np.asarray(payload["term_timesteps"], dtype=np.int32).reshape(-1)
            if scores_by_term.shape[0] != len(ckpt_indices):
                raise ValueError(f"{path}: num terms does not match term_ckpt_indices")

            if term_meta is None:
                term_meta = {
                    "term_ckpt_indices": ckpt_indices,
                    "term_snapshot_positions": snapshot_positions,
                    "term_timesteps": term_timesteps,
                    "num_terms": int(scores_by_term.shape[0]),
                }
            else:
                for key, expected in (
                    ("term_ckpt_indices", ckpt_indices),
                    ("term_snapshot_positions", snapshot_positions),
                    ("term_timesteps", term_timesteps),
                ):
                    if not np.array_equal(np.asarray(term_meta[key]), expected):
                        raise ValueError(f"{path}: {key} differs from the first shard")

            weights = np.ones(scores_by_term.shape[0], dtype=np.float64)
            if term_weight is not None:
                weights = expand_ckpt_snapshot_weights(term_weight, ckpt_indices, snapshot_positions)
            shard_scores = weights @ scores_by_term

            for idx, score in zip(indices, shard_scores):
                idx_int = int(idx)
                if idx_int in score_map:
                    raise ValueError(f"Duplicate score index {idx_int} across full-dim term shards")
                score_map[idx_int] = float(score)
            sources.append({"path": str(path), "num_scores": int(len(indices))})

    if term_meta is None:
        raise ValueError("No term artifacts loaded")
    meta = {
        **term_meta,
        "score_sources": sources,
        "num_scores": int(len(score_map)),
        "score_key": score_key,
    }
    return score_map, meta


def sum_scores(indices: np.ndarray, score_map: dict[int, float], sign: float) -> float:
    total = 0.0
    for idx in np.asarray(indices, dtype=np.int64).reshape(-1):
        total += score_map.get(int(idx), 0.0)
    return float(sign) * total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc LDS eval for full-dim per-term Traj-TracIn scores with optional change weights."
    )
    parser.add_argument("--term-score-npz", required=True, help="Comma-separated full_dim_term_scores.npz shards.")
    parser.add_argument("--target-results", required=True, help="Cached LDS lds_results.csv containing true_f/subset_dir.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--score-key", default="scores_by_term_raw")
    parser.add_argument("--change-weight-npz", default=None, help="NPZ with 50x100/49x100 deltas or weights.")
    parser.add_argument(
        "--weight-normalization",
        choices=["as_is", "global_linear", "per_timestamp_linear"],
        default="per_timestamp_linear",
    )
    parser.add_argument("--clip-weight", default=None, help="Optional MIN,MAX clip after normalization, e.g. 0.25,4.")
    parser.add_argument("--prediction-subset", choices=["kept", "removed"], default="kept")
    parser.add_argument("--prediction-sign", type=float, default=-1.0)
    parser.add_argument("--target-function", default=None)
    parser.add_argument("--trajectory-reduction", default="snapshot_mean")
    parser.add_argument("--algorithm", default="traj_tracin_full_dim_term_weighted")
    parser.add_argument("--mode", choices=["prompted", "unprompted"], default="prompted")
    parser.add_argument("--query", default=None)
    parser.add_argument(
        "--subset-dir-rewrite",
        action="append",
        default=[],
        help="Rewrite subset_dir prefixes in target CSV, e.g. /work2/.../repo=/local/repo. May repeat.",
    )
    args = parser.parse_args()

    started = time.time()
    term_paths = split_paths(args.term_score_npz)
    target_path = Path(args.target_results).expanduser().resolve()
    target_rows = read_rows(target_path)
    if not target_rows:
        raise ValueError(f"No rows in target results: {target_path}")

    clip = None
    if args.clip_weight:
        parts = [float(x.strip()) for x in args.clip_weight.split(",")]
        if len(parts) != 2:
            raise ValueError("--clip-weight must look like MIN,MAX")
        clip = (parts[0], parts[1])

    raw_weight = None
    normalized_weight = None
    if args.change_weight_npz:
        raw_weight = load_delta_or_weight(Path(args.change_weight_npz).expanduser())
        normalized_weight = normalize_delta(raw_weight, args.weight_normalization, eps=1e-12, clip=clip)

    score_map, score_meta = score_map_from_term_artifacts(
        term_paths,
        score_key=args.score_key,
        term_weight=normalized_weight,
    )

    rewrites = parse_rewrites(args.subset_dir_rewrite)
    rows: list[dict[str, object]] = []
    for source_row in target_rows:
        row = dict(source_row)
        subset_dir = Path(rewrite_path(row["subset_dir"], rewrites))
        pred_indices = prediction_indices(subset_dir, args.prediction_subset)
        row["subset_dir"] = str(subset_dir)
        row["prediction_subset"] = args.prediction_subset
        row["prediction_sign"] = args.prediction_sign
        row["pred_sum_tau"] = sum_scores(pred_indices, score_map, args.prediction_sign)
        rows.append(row)

    pred = np.asarray([float(row["pred_sum_tau"]) for row in rows], dtype=np.float64)
    true = np.asarray([float(row["true_f"]) for row in rows], dtype=np.float64)
    lds = spearman_corr(pred, true)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "lds_results.csv", rows)

    if normalized_weight is not None:
        np.savez_compressed(
            out_dir / "applied_term_weights.npz",
            raw_weight=np.asarray(raw_weight, dtype=np.float64),
            normalized_weight=np.asarray(normalized_weight, dtype=np.float64),
        )

    summary = {
        "algorithm": args.algorithm,
        "mode": args.mode,
        "query": args.query,
        "score_key": args.score_key,
        "score_sources": score_meta["score_sources"],
        "num_terms": score_meta["num_terms"],
        "num_scores": score_meta["num_scores"],
        "target_cache": str(target_path),
        "num_models": len(rows),
        "lds_spearman": lds,
        "lds_percent": 100.0 * lds if not math.isnan(lds) else float("nan"),
        "target_function": args.target_function or infer_target_function(target_rows, "cached_target"),
        "trajectory_reduction": args.trajectory_reduction,
        "prediction_subset": args.prediction_subset,
        "prediction_sign": args.prediction_sign,
        "change_weight_npz": args.change_weight_npz,
        "weight_normalization": args.weight_normalization if args.change_weight_npz else "none",
        "clip_weight": args.clip_weight,
        "subset_dir_rewrite": args.subset_dir_rewrite,
        "elapsed_sec": time.time() - started,
    }
    (out_dir / "lds_summary.json").write_text(json.dumps(summary, indent=2))
    plot_scatter(out_dir / "lds_scatter.png", pred, true, f"LDS={lds:.4f} ({100.0 * lds:.2f}%)")
    print(f"[saved] {out_dir} | LDS={lds:.6f} ({100.0 * lds:.3f}%)")


if __name__ == "__main__":
    main()
