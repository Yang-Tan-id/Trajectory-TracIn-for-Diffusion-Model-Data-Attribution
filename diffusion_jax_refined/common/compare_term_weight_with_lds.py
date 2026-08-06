from __future__ import annotations

"""Compare checkpoint/timestamp LDS ablations with predicted-noise-change weights."""

import argparse
import csv
import json
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


def load_weight(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        if key not in payload:
            raise ValueError(f"{path} missing key {key!r}; available keys={payload.files}")
        return np.asarray(payload[key], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare term LDS tables with predicted-noise-change weights.")
    parser.add_argument("--analysis-dir", required=True, help="Directory containing checkpoint_lds.csv/snapshot_lds.csv.")
    parser.add_argument("--weight-npz", required=True)
    parser.add_argument("--weight-key", default="change_weight_by_ckpt_snapshot")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    weights = load_weight(Path(args.weight_npz).expanduser(), args.weight_key)

    checkpoint_rows = read_rows(analysis_dir / "checkpoint_lds.csv")
    snapshot_rows = read_rows(analysis_dir / "snapshot_lds.csv")
    heat_rows = read_rows(analysis_dir / "checkpoint_snapshot_lds.csv")

    ckpt_out = []
    for row in checkpoint_rows:
        ckpt = int(row["ckpt_index"])
        if ckpt >= weights.shape[0]:
            continue
        values = weights[ckpt, :]
        new_row = dict(row)
        new_row["weight_mean"] = float(np.nanmean(values))
        new_row["weight_std"] = float(np.nanstd(values))
        new_row["weight_min"] = float(np.nanmin(values))
        new_row["weight_max"] = float(np.nanmax(values))
        ckpt_out.append(new_row)
    write_csv(out_dir / "checkpoint_lds_with_weight.csv", ckpt_out)

    snapshot_out = []
    for row in snapshot_rows:
        col = int(row["snapshot_column"])
        if col >= weights.shape[1]:
            continue
        values = weights[:, col]
        new_row = dict(row)
        new_row["weight_mean"] = float(np.nanmean(values))
        new_row["weight_std"] = float(np.nanstd(values))
        new_row["weight_min"] = float(np.nanmin(values))
        new_row["weight_max"] = float(np.nanmax(values))
        snapshot_out.append(new_row)
    write_csv(out_dir / "snapshot_lds_with_weight.csv", snapshot_out)

    heat_out = []
    for row in heat_rows:
        ckpt = int(row["ckpt_index"])
        col = int(row["snapshot_column"])
        if ckpt >= weights.shape[0] or col >= weights.shape[1]:
            continue
        new_row = dict(row)
        new_row["weight"] = float(weights[ckpt, col])
        heat_out.append(new_row)
    write_csv(out_dir / "checkpoint_snapshot_lds_with_weight.csv", heat_out)

    ckpt_lds = np.asarray([float(r["lds_percent"]) for r in ckpt_out], dtype=np.float64)
    ckpt_weight = np.asarray([float(r["weight_mean"]) for r in ckpt_out], dtype=np.float64)
    snapshot_lds = np.asarray([float(r["lds_percent"]) for r in snapshot_out], dtype=np.float64)
    snapshot_weight = np.asarray([float(r["weight_mean"]) for r in snapshot_out], dtype=np.float64)
    heat_lds = np.asarray([float(r["lds_percent"]) for r in heat_out], dtype=np.float64)
    heat_weight = np.asarray([float(r["weight"]) for r in heat_out], dtype=np.float64)

    summary = {
        "weight_npz": str(Path(args.weight_npz).expanduser()),
        "weight_key": args.weight_key,
        "weight_shape": list(weights.shape),
        "checkpoint_pearson_lds_weight": corr(ckpt_lds, ckpt_weight, spearman=False),
        "checkpoint_spearman_lds_weight": corr(ckpt_lds, ckpt_weight, spearman=True),
        "snapshot_pearson_lds_weight": corr(snapshot_lds, snapshot_weight, spearman=False),
        "snapshot_spearman_lds_weight": corr(snapshot_lds, snapshot_weight, spearman=True),
        "term_pearson_lds_weight": corr(heat_lds, heat_weight, spearman=False),
        "term_spearman_lds_weight": corr(heat_lds, heat_weight, spearman=True),
        "mean_weight_positive_checkpoint_lds": float(np.nanmean(ckpt_weight[ckpt_lds > 0])) if np.any(ckpt_lds > 0) else float("nan"),
        "mean_weight_negative_checkpoint_lds": float(np.nanmean(ckpt_weight[ckpt_lds < 0])) if np.any(ckpt_lds < 0) else float("nan"),
        "mean_weight_positive_term_lds": float(np.nanmean(heat_weight[heat_lds > 0])) if np.any(heat_lds > 0) else float("nan"),
        "mean_weight_negative_term_lds": float(np.nanmean(heat_weight[heat_lds < 0])) if np.any(heat_lds < 0) else float("nan"),
        "top_weight_checkpoints": sorted(ckpt_out, key=lambda r: float(r["weight_mean"]), reverse=True)[:10],
        "top_weight_terms": sorted(heat_out, key=lambda r: float(r["weight"]), reverse=True)[:20],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "weight_lds_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"[saved] {out_dir}")
    for key in (
        "checkpoint_spearman_lds_weight",
        "snapshot_spearman_lds_weight",
        "term_spearman_lds_weight",
        "mean_weight_positive_checkpoint_lds",
        "mean_weight_negative_checkpoint_lds",
        "mean_weight_positive_term_lds",
        "mean_weight_negative_term_lds",
    ):
        print(f"{key}: {summary[key]}")


if __name__ == "__main__":
    main()
