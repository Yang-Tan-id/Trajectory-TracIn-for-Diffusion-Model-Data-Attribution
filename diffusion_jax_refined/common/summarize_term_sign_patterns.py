from __future__ import annotations

"""Summarize checkpoint/timestamp LDS sign patterns from full-dim term analysis."""

import argparse
import csv
import json
from pathlib import Path


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


def f(row: dict[str, str], key: str) -> float:
    text = row.get(key, "")
    return float(text) if text and text.lower() != "nan" else float("nan")


def sign_label(value: float, eps: float) -> str:
    if value > eps:
        return "positive"
    if value < -eps:
        return "negative"
    return "near_zero"


def mean(values: list[float]) -> float:
    vals = [v for v in values if v == v]
    return sum(vals) / len(vals) if vals else float("nan")


def contiguous_runs(items: list[int]) -> list[dict[str, int]]:
    if not items:
        return []
    vals = sorted(set(items))
    runs = []
    start = prev = vals[0]
    for val in vals[1:]:
        if val == prev + 1:
            prev = val
            continue
        runs.append({"start": start, "end": prev, "length": prev - start + 1})
        start = prev = val
    runs.append({"start": start, "end": prev, "length": prev - start + 1})
    return runs


def color_for(value: float, vlim: float) -> str:
    if value != value:
        return "#f3f4f6"
    x = max(-1.0, min(1.0, value / max(vlim, 1e-12)))
    if x >= 0:
        r = int(255 - 125 * x)
        g = int(255 - 70 * x)
        b = int(255 - 120 * x)
    else:
        x = -x
        r = int(255 - 65 * x)
        g = int(255 - 135 * x)
        b = int(255 - 165 * x)
    return f"#{r:02x}{g:02x}{b:02x}"


def write_svg_heatmap(path: Path, matrix: dict[tuple[int, int], float], ckpts: list[int], snapshots: list[int]) -> None:
    cell = 8
    left = 70
    top = 48
    width = left + len(snapshots) * cell + 20
    height = top + len(ckpts) * cell + 40
    vals = [abs(v) for v in matrix.values() if v == v]
    vlim = max(vals) if vals else 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="12" y="24" font-family="Arial" font-size="15" font-weight="700">Term LDS% heatmap</text>',
        '<text x="12" y="40" font-family="Arial" font-size="10" fill="#555">red=negative, green=positive</text>',
    ]
    for i, ckpt in enumerate(ckpts):
        y = top + i * cell
        if i % 5 == 0:
            parts.append(f'<text x="8" y="{y + cell - 1}" font-family="Arial" font-size="8" fill="#555">ckpt {ckpt}</text>')
    for j, snap in enumerate(snapshots):
        x = left + j * cell
        if j % 10 == 0:
            parts.append(
                f'<text x="{x}" y="{top - 6}" font-family="Arial" font-size="8" fill="#555" transform="rotate(-45 {x},{top - 6})">s{snap}</text>'
            )
    for i, ckpt in enumerate(ckpts):
        for j, snap in enumerate(snapshots):
            val = matrix.get((ckpt, snap), float("nan"))
            parts.append(
                f'<rect x="{left + j * cell}" y="{top + i * cell}" width="{cell}" height="{cell}" '
                f'fill="{color_for(val, vlim)}"><title>ckpt={ckpt}, snapshot={snap}, LDS%={val:.3f}</title></rect>'
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize positive/negative checkpoint and timestamp LDS patterns.")
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--eps", type=float, default=0.0, help="Absolute LDS percent threshold for near-zero.")
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else analysis_dir / "sign_patterns"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_rows = read_rows(analysis_dir / "checkpoint_lds.csv")
    snapshot_rows = read_rows(analysis_dir / "snapshot_lds.csv")
    term_rows = read_rows(analysis_dir / "checkpoint_snapshot_lds.csv")

    ckpt_sign = {int(r["ckpt_index"]): sign_label(f(r, "lds_percent"), args.eps) for r in ckpt_rows}
    snap_sign = {int(r["snapshot_column"]): sign_label(f(r, "lds_percent"), args.eps) for r in snapshot_rows}
    ckpt_lds = {int(r["ckpt_index"]): f(r, "lds_percent") for r in ckpt_rows}
    snap_lds = {int(r["snapshot_column"]): f(r, "lds_percent") for r in snapshot_rows}

    quadrant: dict[tuple[str, str], list[float]] = {}
    annotated = []
    matrix = {}
    for row in term_rows:
        ckpt = int(row["ckpt_index"])
        snap = int(row["snapshot_column"])
        val = f(row, "lds_percent")
        matrix[(ckpt, snap)] = val
        key = (ckpt_sign.get(ckpt, "unknown"), snap_sign.get(snap, "unknown"))
        quadrant.setdefault(key, []).append(val)
        annotated.append(
            {
                **row,
                "checkpoint_sign": ckpt_sign.get(ckpt, "unknown"),
                "snapshot_sign": snap_sign.get(snap, "unknown"),
                "checkpoint_lds_percent": ckpt_lds.get(ckpt, float("nan")),
                "snapshot_lds_percent": snap_lds.get(snap, float("nan")),
            }
        )
    write_csv(out_dir / "checkpoint_snapshot_signs.csv", annotated)

    quadrant_rows = []
    for (c_sign, s_sign), vals in sorted(quadrant.items()):
        quadrant_rows.append(
            {
                "checkpoint_sign": c_sign,
                "snapshot_sign": s_sign,
                "num_terms": len(vals),
                "mean_term_lds_percent": mean(vals),
                "positive_term_fraction": mean([1.0 if v > args.eps else 0.0 for v in vals if v == v]),
                "negative_term_fraction": mean([1.0 if v < -args.eps else 0.0 for v in vals if v == v]),
            }
        )
    write_csv(out_dir / "quadrant_summary.csv", quadrant_rows)

    positive_snapshots = sorted([idx for idx, label in snap_sign.items() if label == "positive"])
    negative_snapshots = sorted([idx for idx, label in snap_sign.items() if label == "negative"])
    positive_ckpts = sorted([idx for idx, label in ckpt_sign.items() if label == "positive"])
    negative_ckpts = sorted([idx for idx, label in ckpt_sign.items() if label == "negative"])

    ckpt_cross_rows = []
    for ckpt in sorted(ckpt_sign):
        pos_vals = [matrix[(ckpt, snap)] for snap in positive_snapshots if (ckpt, snap) in matrix]
        neg_vals = [matrix[(ckpt, snap)] for snap in negative_snapshots if (ckpt, snap) in matrix]
        ckpt_cross_rows.append(
            {
                "ckpt_index": ckpt,
                "checkpoint_sign": ckpt_sign[ckpt],
                "checkpoint_lds_percent": ckpt_lds[ckpt],
                "mean_on_positive_snapshots": mean(pos_vals),
                "mean_on_negative_snapshots": mean(neg_vals),
                "positive_fraction_on_positive_snapshots": mean([1.0 if v > args.eps else 0.0 for v in pos_vals if v == v]),
                "positive_fraction_on_negative_snapshots": mean([1.0 if v > args.eps else 0.0 for v in neg_vals if v == v]),
            }
        )
    write_csv(out_dir / "checkpoint_cross_snapshot_sign_summary.csv", ckpt_cross_rows)

    snapshot_cross_rows = []
    for snap in sorted(snap_sign):
        pos_vals = [matrix[(ckpt, snap)] for ckpt in positive_ckpts if (ckpt, snap) in matrix]
        neg_vals = [matrix[(ckpt, snap)] for ckpt in negative_ckpts if (ckpt, snap) in matrix]
        snapshot_cross_rows.append(
            {
                "snapshot_column": snap,
                "snapshot_sign": snap_sign[snap],
                "snapshot_lds_percent": snap_lds[snap],
                "mean_on_positive_checkpoints": mean(pos_vals),
                "mean_on_negative_checkpoints": mean(neg_vals),
                "positive_fraction_on_positive_checkpoints": mean([1.0 if v > args.eps else 0.0 for v in pos_vals if v == v]),
                "positive_fraction_on_negative_checkpoints": mean([1.0 if v > args.eps else 0.0 for v in neg_vals if v == v]),
            }
        )
    write_csv(out_dir / "snapshot_cross_checkpoint_sign_summary.csv", snapshot_cross_rows)

    write_svg_heatmap(out_dir / "checkpoint_snapshot_lds_heatmap.svg", matrix, sorted(ckpt_sign), sorted(snap_sign))

    summary = {
        "analysis_dir": str(analysis_dir),
        "positive_checkpoint_count": len(positive_ckpts),
        "negative_checkpoint_count": len(negative_ckpts),
        "positive_snapshot_count": len(positive_snapshots),
        "negative_snapshot_count": len(negative_snapshots),
        "positive_checkpoint_runs": contiguous_runs(positive_ckpts),
        "negative_checkpoint_runs": contiguous_runs(negative_ckpts),
        "positive_snapshot_runs": contiguous_runs(positive_snapshots),
        "negative_snapshot_runs": contiguous_runs(negative_snapshots),
        "quadrants": quadrant_rows,
    }
    (out_dir / "sign_pattern_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"[saved] {out_dir}")
    print(
        f"checkpoints: positive={len(positive_ckpts)} negative={len(negative_ckpts)} | "
        f"snapshots: positive={len(positive_snapshots)} negative={len(negative_snapshots)}"
    )
    for row in quadrant_rows:
        print(
            f"{row['checkpoint_sign']:>9s} ckpt x {row['snapshot_sign']:>9s} snapshot: "
            f"n={row['num_terms']:4d} mean_term_LDS={float(row['mean_term_lds_percent']):8.3f}% "
            f"pos_frac={float(row['positive_term_fraction']):.3f}"
        )


if __name__ == "__main__":
    main()
