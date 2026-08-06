from __future__ import annotations

"""Diagnose mismatch between Traj-TracIn subset predictions and LDS true_f."""

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


def corr(x: np.ndarray, y: np.ndarray, *, spearman: bool = False) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    xx = rankdata_average(x[mask]) if spearman else x[mask]
    yy = rankdata_average(y[mask]) if spearman else y[mask]
    if float(xx.std()) == 0.0 or float(yy.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


def load_score_vector(paths: list[Path], score_key: str) -> dict[int, float]:
    score_map: dict[int, float] = {}
    for path in paths:
        with np.load(path, allow_pickle=True) as payload:
            if score_key not in payload:
                raise ValueError(f"{path} missing {score_key}; keys={payload.files}")
            scores_by_term = np.asarray(payload[score_key], dtype=np.float64)
            indices = np.asarray(payload["score_indices"], dtype=np.int64).reshape(-1)
            if scores_by_term.ndim != 2:
                raise ValueError(f"{path}:{score_key} must be [terms, points], got {scores_by_term.shape}")
            if scores_by_term.shape[1] != len(indices):
                raise ValueError(f"{path}: score columns do not match score_indices")
            scores = np.sum(scores_by_term, axis=0)
            for idx, score in zip(indices, scores):
                idx_int = int(idx)
                if idx_int in score_map:
                    raise ValueError(f"Duplicate score index {idx_int}")
                score_map[idx_int] = float(score)
    return score_map


def prediction_indices(subset_dir: Path, subset: str) -> np.ndarray:
    filename = "kept_attribution_indices.npy" if subset == "kept" else "excluded_attribution_indices.npy"
    return np.load(subset_dir / filename)


def sum_scores(indices: np.ndarray, score_map: dict[int, float], sign: float) -> float:
    total = 0.0
    for idx in np.asarray(indices, dtype=np.int64).reshape(-1):
        total += score_map.get(int(idx), 0.0)
    return float(sign) * total


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    std = float(np.std(values))
    if std == 0.0:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / std


def write_scatter_svg(path: Path, pred: np.ndarray, true: np.ndarray, title: str) -> None:
    w, h = 720, 420
    left, right, top, bottom = 70, 24, 34, 58
    xmin, xmax = float(np.min(pred)), float(np.max(pred))
    ymin, ymax = float(np.min(true)), float(np.max(true))
    if xmin == xmax:
        xmax = xmin + 1.0
    if ymin == ymax:
        ymax = ymin + 1.0
    order = np.argsort(true)
    colors = ["#2563eb", "#0891b2", "#16a34a", "#ea580c"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="16" y="22" font-family="Arial" font-size="15" font-weight="700">{title}</text>',
        f'<line x1="{left}" y1="{h-bottom}" x2="{w-right}" y2="{h-bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{h-bottom}" stroke="#333"/>',
        f'<text x="{left}" y="{h-18}" font-family="Arial" font-size="11" fill="#555">predicted sum of attribution scores</text>',
        f'<text x="12" y="{top+12}" font-family="Arial" font-size="11" fill="#555">true_f</text>',
    ]
    for rank, idx in enumerate(order):
        quartile = min(3, int(4 * rank / max(1, len(order))))
        x = left + (w - left - right) * (float(pred[idx]) - xmin) / (xmax - xmin)
        y = top + (h - top - bottom) * (1.0 - (float(true[idx]) - ymin) / (ymax - ymin))
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colors[quartile]}" opacity="0.78">'
            f'<title>subset={idx}, true_f={float(true[idx]):.6g}, pred={float(pred[idx]):.6g}, quartile=Q{quartile + 1}</title>'
            '</circle>'
        )
    for q, color in enumerate(colors):
        x = left + q * 96
        parts.append(f'<circle cx="{x}" cy="390" r="4" fill="{color}"/>')
        parts.append(f'<text x="{x+8}" y="394" font-family="Arial" font-size="11" fill="#555">true_f Q{q + 1}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose full-dim score vs LDS true_f mismatch.")
    parser.add_argument("--term-score-npz", required=True, help="Comma-separated full_dim_term_scores.npz shards.")
    parser.add_argument("--target-results", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--score-key", default="scores_by_term_raw")
    parser.add_argument("--prediction-subset", choices=["kept", "removed"], default="kept")
    parser.add_argument("--prediction-sign", type=float, default=-1.0)
    parser.add_argument("--name", default="traj_tracin")
    args = parser.parse_args()

    score_map = load_score_vector(split_paths(args.term_score_npz), args.score_key)
    rows = read_rows(Path(args.target_results).expanduser())
    pred_values = []
    true_values = []
    out_rows = []
    for i, row in enumerate(rows):
        true_f = float(row["true_f"])
        pred = sum_scores(prediction_indices(Path(row["subset_dir"]), args.prediction_subset), score_map, args.prediction_sign)
        pred_values.append(pred)
        true_values.append(true_f)
        out_rows.append(
            {
                "subset_row": i,
                "true_f": true_f,
                "pred_sum_tau": pred,
                "subset_dir": row["subset_dir"],
            }
        )

    pred_np = np.asarray(pred_values, dtype=np.float64)
    true_np = np.asarray(true_values, dtype=np.float64)
    pred_z = zscore(pred_np)
    true_z = zscore(true_np)
    mismatch = pred_z - true_z
    true_order = np.argsort(true_np)
    for rank, idx in enumerate(true_order):
        out_rows[int(idx)]["true_rank"] = rank + 1
        out_rows[int(idx)]["true_quartile"] = min(4, int(4 * rank / max(1, len(true_order))) + 1)
    pred_order = np.argsort(pred_np)
    pred_rank = np.empty(len(pred_order), dtype=np.int64)
    pred_rank[pred_order] = np.arange(1, len(pred_order) + 1)
    for i in range(len(out_rows)):
        out_rows[i]["pred_rank"] = int(pred_rank[i])
        out_rows[i]["rank_error"] = int(pred_rank[i]) - int(out_rows[i]["true_rank"])
        out_rows[i]["pred_z"] = float(pred_z[i])
        out_rows[i]["true_z"] = float(true_z[i])
        out_rows[i]["z_mismatch"] = float(mismatch[i])

    quartile_rows = []
    for q in range(1, 5):
        mask = np.asarray([int(row["true_quartile"]) == q for row in out_rows], dtype=bool)
        quartile_rows.append(
            {
                "true_f_quartile": q,
                "n": int(np.sum(mask)),
                "true_f_mean": float(np.mean(true_np[mask])),
                "true_f_std": float(np.std(true_np[mask])),
                "pred_mean": float(np.mean(pred_np[mask])),
                "pred_std": float(np.std(pred_np[mask])),
                "pred_z_mean": float(np.mean(pred_z[mask])),
                "mean_abs_rank_error": float(np.mean(np.abs([float(row["rank_error"]) for row in out_rows if int(row["true_quartile"]) == q]))),
            }
        )

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "subset_pred_true_mismatch.csv", out_rows)
    write_csv(out_dir / "true_f_quartile_summary.csv", quartile_rows)
    write_scatter_svg(out_dir / "pred_true_scatter.svg", pred_np, true_np, args.name)

    summary = {
        "name": args.name,
        "target_results": str(Path(args.target_results).expanduser()),
        "num_models": int(len(rows)),
        "num_scored_points": int(len(score_map)),
        "prediction_subset": args.prediction_subset,
        "prediction_sign": args.prediction_sign,
        "true_f_min": float(np.min(true_np)),
        "true_f_mean": float(np.mean(true_np)),
        "true_f_max": float(np.max(true_np)),
        "true_f_range": float(np.max(true_np) - np.min(true_np)),
        "true_f_std": float(np.std(true_np)),
        "pred_min": float(np.min(pred_np)),
        "pred_mean": float(np.mean(pred_np)),
        "pred_max": float(np.max(pred_np)),
        "pred_range": float(np.max(pred_np) - np.min(pred_np)),
        "pred_std": float(np.std(pred_np)),
        "pred_std_over_true_f_std": float(np.std(pred_np) / max(float(np.std(true_np)), 1e-30)),
        "spearman": corr(pred_np, true_np, spearman=True),
        "pearson": corr(pred_np, true_np, spearman=False),
        "mean_abs_rank_error": float(np.mean(np.abs([float(row["rank_error"]) for row in out_rows]))),
        "worst_rank_mismatches": sorted(out_rows, key=lambda row: abs(float(row["rank_error"])), reverse=True)[:10],
        "quartiles": quartile_rows,
    }
    (out_dir / "mismatch_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[saved] {out_dir}")
    print(
        f"LDS={100.0 * summary['spearman']:.3f}% | "
        f"true_f std={summary['true_f_std']:.6g} range={summary['true_f_range']:.6g} | "
        f"pred std={summary['pred_std']:.6g} range={summary['pred_range']:.6g}"
    )
    print(f"pred_std/true_f_std={summary['pred_std_over_true_f_std']:.6g}")


if __name__ == "__main__":
    main()
