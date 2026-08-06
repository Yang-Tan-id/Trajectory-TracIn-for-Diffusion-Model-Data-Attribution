from __future__ import annotations

"""Analyze LDS behavior of full-dim Traj-TracIn per-term score artifacts.

The script is intentionally CPU-only. It loads full_dim_term_scores.npz shards,
builds subset-level LDS predictions from cached target lds_results.csv, and
reports whether a query's negative LDS comes from a global sign flip or from
specific checkpoint/timestamp regions.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def split_paths(text: str) -> list[Path]:
    paths = [Path(item.strip()).expanduser() for item in text.split(",") if item.strip()]
    if not paths:
        raise ValueError("No paths provided")
    return paths


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
    if float(pred_rank.std()) == 0.0 or float(true_rank.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(pred_rank, true_rank)[0, 1])


def load_term_matrix(paths: list[Path], score_key: str) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    score_blocks = []
    index_blocks = []
    meta: dict[str, np.ndarray] | None = None
    for path in paths:
        with np.load(path, allow_pickle=True) as payload:
            if score_key not in payload:
                raise ValueError(f"{path} missing {score_key}; keys={payload.files}")
            scores = np.asarray(payload[score_key], dtype=np.float32)
            indices = np.asarray(payload["score_indices"], dtype=np.int64).reshape(-1)
            this_meta = {
                "term_ckpt_indices": np.asarray(payload["term_ckpt_indices"], dtype=np.int32).reshape(-1),
                "term_timesteps": np.asarray(payload["term_timesteps"], dtype=np.int32).reshape(-1),
                "term_snapshot_positions": np.asarray(payload["term_snapshot_positions"], dtype=np.int32).reshape(-1),
            }
            if scores.ndim != 2:
                raise ValueError(f"{path}:{score_key} must be [terms, points], got {scores.shape}")
            if scores.shape[1] != len(indices):
                raise ValueError(f"{path}: score columns do not match score_indices")
            if meta is None:
                meta = this_meta
            else:
                for key, expected in this_meta.items():
                    if not np.array_equal(meta[key], expected):
                        raise ValueError(f"{path}: {key} differs from first shard")
            score_blocks.append(scores)
            index_blocks.append(indices)

    assert meta is not None
    all_indices = np.concatenate(index_blocks, axis=0)
    order = np.argsort(all_indices)
    all_indices = all_indices[order]
    matrix = np.concatenate(score_blocks, axis=1)[:, order].astype(np.float64)
    if len(np.unique(all_indices)) != len(all_indices):
        raise ValueError("Duplicate score_indices across shards")
    return matrix, all_indices, meta


def subset_indicator_matrix(
    target_rows: list[dict[str, str]],
    score_indices: np.ndarray,
    *,
    subset: str,
) -> tuple[np.ndarray, np.ndarray]:
    pos = {int(idx): i for i, idx in enumerate(np.asarray(score_indices, dtype=np.int64))}
    indicators = np.zeros((len(target_rows), len(score_indices)), dtype=np.float64)
    missing_counts = np.zeros((len(target_rows),), dtype=np.int64)
    filename = "kept_attribution_indices.npy" if subset == "kept" else "excluded_attribution_indices.npy"
    for row_i, row in enumerate(target_rows):
        arr = np.load(Path(row["subset_dir"]) / filename)
        missing = 0
        for idx in np.asarray(arr, dtype=np.int64).reshape(-1):
            col = pos.get(int(idx))
            if col is None:
                missing += 1
            else:
                indicators[row_i, col] = 1.0
        missing_counts[row_i] = missing
    return indicators, missing_counts


def pred_for_terms(
    matrix: np.ndarray,
    indicators: np.ndarray,
    term_mask: np.ndarray,
    *,
    sign: float,
) -> np.ndarray:
    if int(np.sum(term_mask)) == 0:
        return np.full((indicators.shape[0],), np.nan, dtype=np.float64)
    per_point = np.sum(matrix[term_mask, :], axis=0)
    return float(sign) * (indicators @ per_point)


def summarize_prediction(name: str, pred: np.ndarray, true: np.ndarray, num_terms: int) -> dict[str, object]:
    lds = spearman_corr(pred, true)
    return {
        "name": name,
        "num_terms": int(num_terms),
        "lds_spearman": lds,
        "lds_percent": 100.0 * lds if not math.isnan(lds) else float("nan"),
        "pred_mean": float(np.nanmean(pred)),
        "pred_std": float(np.nanstd(pred)),
        "pred_min": float(np.nanmin(pred)),
        "pred_max": float(np.nanmax(pred)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze full-dim per-term Traj-TracIn LDS behavior.")
    parser.add_argument("--term-score-npz", required=True, help="Comma-separated full_dim_term_scores.npz shards.")
    parser.add_argument("--target-results", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--score-key", default="scores_by_term_raw")
    parser.add_argument("--prediction-subset", choices=["kept", "removed"], default="kept")
    parser.add_argument("--prediction-sign", type=float, default=-1.0)
    parser.add_argument("--query", default=None)
    parser.add_argument("--target-function", default=None)
    args = parser.parse_args()

    paths = split_paths(args.term_score_npz)
    target_rows = read_rows(Path(args.target_results).expanduser())
    if not target_rows:
        raise ValueError(f"No rows in {args.target_results}")

    matrix, score_indices, meta = load_term_matrix(paths, args.score_key)
    true = np.asarray([float(row["true_f"]) for row in target_rows], dtype=np.float64)
    indicators, missing_counts = subset_indicator_matrix(
        target_rows,
        score_indices,
        subset=args.prediction_subset,
    )
    if np.any(missing_counts):
        print(f"[warning] some subset indices were missing from score shards: max_missing={int(np.max(missing_counts))}")

    ckpts = np.asarray(meta["term_ckpt_indices"], dtype=np.int32)
    snapshots = np.asarray(meta["term_snapshot_positions"], dtype=np.int32)
    timesteps = np.asarray(meta["term_timesteps"], dtype=np.int32)

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_mask = np.ones(matrix.shape[0], dtype=bool)
    summary_rows = []
    pred_minus = pred_for_terms(matrix, indicators, all_mask, sign=-1.0)
    pred_plus = pred_for_terms(matrix, indicators, all_mask, sign=1.0)
    summary_rows.append(summarize_prediction("all_terms_sign_m1", pred_minus, true, matrix.shape[0]))
    summary_rows.append(summarize_prediction("all_terms_sign_p1", pred_plus, true, matrix.shape[0]))
    pred_configured = pred_for_terms(matrix, indicators, all_mask, sign=args.prediction_sign)
    summary_rows.append(
        summarize_prediction(f"all_terms_sign_{args.prediction_sign:g}", pred_configured, true, matrix.shape[0])
    )
    write_csv(out_dir / "summary.csv", summary_rows)

    ckpt_rows = []
    for ckpt in sorted(np.unique(ckpts).tolist()):
        mask = ckpts == int(ckpt)
        pred = pred_for_terms(matrix, indicators, mask, sign=args.prediction_sign)
        row = summarize_prediction(f"ckpt_{int(ckpt):04d}", pred, true, int(np.sum(mask)))
        row["ckpt_index"] = int(ckpt)
        ckpt_rows.append(row)
    write_csv(out_dir / "checkpoint_lds.csv", ckpt_rows)

    snapshot_rows = []
    unique_snapshots = sorted(np.unique(snapshots).tolist())
    for compact_i, snapshot in enumerate(unique_snapshots):
        mask = snapshots == int(snapshot)
        pred = pred_for_terms(matrix, indicators, mask, sign=args.prediction_sign)
        row = summarize_prediction(f"snapshot_{compact_i:04d}", pred, true, int(np.sum(mask)))
        row["snapshot_column"] = int(compact_i)
        row["snapshot_position"] = int(snapshot)
        row["timestep"] = int(timesteps[np.where(mask)[0][0]])
        snapshot_rows.append(row)
    write_csv(out_dir / "snapshot_lds.csv", snapshot_rows)

    heat_rows = []
    for ckpt in sorted(np.unique(ckpts).tolist()):
        for compact_i, snapshot in enumerate(unique_snapshots):
            mask = (ckpts == int(ckpt)) & (snapshots == int(snapshot))
            pred = pred_for_terms(matrix, indicators, mask, sign=args.prediction_sign)
            row = summarize_prediction(
                f"ckpt_{int(ckpt):04d}_snapshot_{compact_i:04d}",
                pred,
                true,
                int(np.sum(mask)),
            )
            row["ckpt_index"] = int(ckpt)
            row["snapshot_column"] = int(compact_i)
            row["snapshot_position"] = int(snapshot)
            row["timestep"] = int(timesteps[np.where(mask)[0][0]]) if np.any(mask) else -1
            heat_rows.append(row)
    write_csv(out_dir / "checkpoint_snapshot_lds.csv", heat_rows)

    summary = {
        "query": args.query,
        "target_function": args.target_function,
        "target_results": str(Path(args.target_results).expanduser()),
        "term_score_npz": [str(p) for p in paths],
        "score_key": args.score_key,
        "num_terms": int(matrix.shape[0]),
        "num_scores": int(matrix.shape[1]),
        "num_models": int(len(target_rows)),
        "prediction_subset": args.prediction_subset,
        "prediction_sign": args.prediction_sign,
        "true_f_min": float(np.min(true)),
        "true_f_mean": float(np.mean(true)),
        "true_f_max": float(np.max(true)),
        "summary": summary_rows,
        "best_checkpoints": sorted(ckpt_rows, key=lambda r: float(r["lds_spearman"]), reverse=True)[:5],
        "worst_checkpoints": sorted(ckpt_rows, key=lambda r: float(r["lds_spearman"]))[:5],
        "best_snapshots": sorted(snapshot_rows, key=lambda r: float(r["lds_spearman"]), reverse=True)[:5],
        "worst_snapshots": sorted(snapshot_rows, key=lambda r: float(r["lds_spearman"]))[:5],
        "best_terms": sorted(heat_rows, key=lambda r: float(r["lds_spearman"]), reverse=True)[:10],
        "worst_terms": sorted(heat_rows, key=lambda r: float(r["lds_spearman"]))[:10],
    }
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"[saved] {out_dir}")
    for row in summary_rows:
        print(f"{row['name']:>20s}  LDS={float(row['lds_percent']):8.3f}%")
    print("[best checkpoints]", ", ".join(f"{r['ckpt_index']}:{float(r['lds_percent']):.2f}%" for r in summary["best_checkpoints"]))
    print("[worst checkpoints]", ", ".join(f"{r['ckpt_index']}:{float(r['lds_percent']):.2f}%" for r in summary["worst_checkpoints"]))
    print("[best snapshots]", ", ".join(f"{r['snapshot_column']}:{float(r['lds_percent']):.2f}%" for r in summary["best_snapshots"]))
    print("[worst snapshots]", ", ".join(f"{r['snapshot_column']}:{float(r['lds_percent']):.2f}%" for r in summary["worst_snapshots"]))


if __name__ == "__main__":
    main()
