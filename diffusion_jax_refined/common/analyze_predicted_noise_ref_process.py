from __future__ import annotations

"""Summarize predicted-noise distance-to-reference over checkpoints/timestamps."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


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


def color_for_log(value: float, vmin: float, vmax: float) -> str:
    if value != value:
        return "#f3f4f6"
    x = (value - vmin) / max(vmax - vmin, 1e-12)
    x = max(0.0, min(1.0, x))
    # Low distance is green/white, high distance is red.
    r = int(235 + 20 * x)
    g = int(250 - 145 * x)
    b = int(235 - 165 * x)
    return f"#{r:02x}{g:02x}{b:02x}"


def color_for_signed(value: float, vmax_abs: float) -> str:
    if value != value:
        return "#f3f4f6"
    x = max(-1.0, min(1.0, value / max(vmax_abs, 1e-12)))
    if x >= 0:
        r = int(245 - 95 * x)
        g = int(245 - 75 * x)
        b = int(245 - 135 * x)
    else:
        x = -x
        r = int(245 - 45 * x)
        g = int(245 - 130 * x)
        b = int(245 - 130 * x)
    return f"#{r:02x}{g:02x}{b:02x}"


def write_heatmap_svg(path: Path, values: np.ndarray, timesteps: np.ndarray) -> None:
    ckpts, snaps = values.shape
    cell = 8
    left = 72
    top = 52
    width = left + snaps * cell + 24
    height = top + ckpts * cell + 44
    logged = np.log10(np.maximum(values, 1e-30))
    vmin = float(np.nanmin(logged))
    vmax = float(np.nanmax(logged))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="12" y="24" font-family="Arial" font-size="15" font-weight="700">log10 MSE to reference eps</text>',
        '<text x="12" y="42" font-family="Arial" font-size="10" fill="#555">green=closer, red=farther</text>',
    ]
    for c in range(ckpts):
        y = top + c * cell
        if c % 5 == 0:
            parts.append(f'<text x="8" y="{y + cell - 1}" font-family="Arial" font-size="8" fill="#555">ckpt {c}</text>')
    for s in range(snaps):
        x = left + s * cell
        if s % 10 == 0:
            label = f"s{s}/t{int(timesteps[s])}" if s < len(timesteps) else f"s{s}"
            parts.append(
                f'<text x="{x}" y="{top - 6}" font-family="Arial" font-size="8" fill="#555" '
                f'transform="rotate(-45 {x},{top - 6})">{label}</text>'
            )
    for c in range(ckpts):
        for s in range(snaps):
            logv = float(logged[c, s])
            raw = float(values[c, s])
            parts.append(
                f'<rect x="{left + s * cell}" y="{top + c * cell}" width="{cell}" height="{cell}" '
                f'fill="{color_for_log(logv, vmin, vmax)}">'
                f'<title>ckpt={c}, snapshot={s}, timestep={int(timesteps[s]) if s < len(timesteps) else -1}, mse={raw:.6g}</title>'
                '</rect>'
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def write_signed_heatmap_svg(path: Path, values: np.ndarray, timesteps: np.ndarray, *, title: str) -> None:
    rows, snaps = values.shape
    cell = 8
    left = 82
    top = 52
    width = left + snaps * cell + 24
    height = top + rows * cell + 44
    vmax_abs = float(np.nanmax(np.abs(values)))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="12" y="24" font-family="Arial" font-size="15" font-weight="700">{title}</text>',
        '<text x="12" y="42" font-family="Arial" font-size="10" fill="#555">green=positive/toward reference, red=negative/away</text>',
    ]
    for r in range(rows):
        y = top + r * cell
        if r % 5 == 0:
            parts.append(f'<text x="8" y="{y + cell - 1}" font-family="Arial" font-size="8" fill="#555">trans {r}</text>')
    for s in range(snaps):
        x = left + s * cell
        if s % 10 == 0:
            label = f"s{s}/t{int(timesteps[s])}" if s < len(timesteps) else f"s{s}"
            parts.append(
                f'<text x="{x}" y="{top - 6}" font-family="Arial" font-size="8" fill="#555" '
                f'transform="rotate(-45 {x},{top - 6})">{label}</text>'
            )
    for r in range(rows):
        for s in range(snaps):
            raw = float(values[r, s])
            parts.append(
                f'<rect x="{left + s * cell}" y="{top + r * cell}" width="{cell}" height="{cell}" '
                f'fill="{color_for_signed(raw, vmax_abs)}">'
                f'<title>transition={r}, snapshot={s}, timestep={int(timesteps[s]) if s < len(timesteps) else -1}, value={raw:.6g}</title>'
                '</rect>'
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def write_mean_curve_svg(path: Path, checkpoint_mean: np.ndarray) -> None:
    w, h = 760, 320
    pad_l, pad_r, pad_t, pad_b = 54, 18, 24, 42
    xs = np.arange(len(checkpoint_mean), dtype=np.float64)
    ys = np.log10(np.maximum(checkpoint_mean, 1e-30))
    ymin, ymax = float(np.min(ys)), float(np.max(ys))
    if ymin == ymax:
        ymax = ymin + 1.0
    pts = []
    for x, y in zip(xs, ys):
        px = pad_l + (w - pad_l - pad_r) * x / max(len(xs) - 1, 1)
        py = pad_t + (h - pad_t - pad_b) * (1.0 - (y - ymin) / (ymax - ymin))
        pts.append(f"{px:.1f},{py:.1f}")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="14" y="20" font-family="Arial" font-size="15" font-weight="700">Mean eps-to-reference MSE over checkpoints</text>',
        f'<line x1="{pad_l}" y1="{h-pad_b}" x2="{w-pad_r}" y2="{h-pad_b}" stroke="#333"/>',
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{h-pad_b}" stroke="#333"/>',
        f'<polyline points="{" ".join(pts)}" fill="none" stroke="#2563eb" stroke-width="2"/>',
    ]
    for i in range(0, len(xs), 5):
        px = pad_l + (w - pad_l - pad_r) * i / max(len(xs) - 1, 1)
        parts.append(f'<text x="{px-8:.1f}" y="{h-18}" font-family="Arial" font-size="9" fill="#555">{i}</text>')
    parts.extend(
        [
            f'<text x="8" y="{pad_t+5}" font-family="Arial" font-size="9" fill="#555">{ymax:.2f}</text>',
            f'<text x="8" y="{h-pad_b}" font-family="Arial" font-size="9" fill="#555">{ymin:.2f}</text>',
            '<text x="330" y="310" font-family="Arial" font-size="10" fill="#555">checkpoint index</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze eps-to-reference convergence from weights.npz.")
    parser.add_argument("--weights-npz", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    with np.load(Path(args.weights_npz).expanduser(), allow_pickle=False) as payload:
        if "eps_ref_mse_by_ckpt_snapshot" not in payload:
            raise ValueError(
                f"{args.weights_npz} does not contain eps_ref_mse_by_ckpt_snapshot. "
                "Regenerate it with the updated predicted_noise_change_weights.py."
            )
        ref_mse = np.asarray(payload["eps_ref_mse_by_ckpt_snapshot"], dtype=np.float64)
        timesteps = np.asarray(payload["timesteps"], dtype=np.int32)
        positions = np.asarray(payload["snapshot_positions"], dtype=np.int32)
        cosine_transition = (
            np.asarray(payload["eps_to_ref_cosine_by_transition"], dtype=np.float64)
            if "eps_to_ref_cosine_by_transition" in payload
            else None
        )
        progress_transition = (
            np.asarray(payload["eps_to_ref_progress_by_transition"], dtype=np.float64)
            if "eps_to_ref_progress_by_transition" in payload
            else None
        )

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_mean = np.mean(ref_mse, axis=1)
    checkpoint_rows = []
    for c, row in enumerate(ref_mse):
        checkpoint_rows.append(
            {
                "ckpt_index": c,
                "epoch": 4 * (c + 1),
                "mean_ref_mse": float(np.mean(row)),
                "median_ref_mse": float(np.median(row)),
                "min_ref_mse": float(np.min(row)),
                "max_ref_mse": float(np.max(row)),
            }
        )
    write_csv(out_dir / "checkpoint_ref_mse.csv", checkpoint_rows)

    snapshot_rows = []
    ckpt_index = np.arange(ref_mse.shape[0], dtype=np.float64)
    for s in range(ref_mse.shape[1]):
        y = ref_mse[:, s]
        diffs = np.diff(y)
        snapshot_rows.append(
            {
                "snapshot_column": s,
                "snapshot_position": int(positions[s]) if s < len(positions) else s,
                "timestep": int(timesteps[s]) if s < len(timesteps) else -1,
                "start_ref_mse": float(y[0]),
                "end_ref_mse": float(y[-1]),
                "min_ref_mse": float(np.min(y)),
                "max_ref_mse": float(np.max(y)),
                "end_over_start": float(y[-1] / max(y[0], 1e-30)),
                "absolute_change": float(y[-1] - y[0]),
                "decrease_fraction": float(np.mean(diffs <= 0.0)),
                "spearman_ckpt_vs_ref_mse": corr(ckpt_index, y, spearman=True),
                "pearson_ckpt_vs_log_ref_mse": corr(ckpt_index, np.log10(np.maximum(y, 1e-30)), spearman=False),
            }
        )
    write_csv(out_dir / "snapshot_ref_mse_trends.csv", snapshot_rows)

    write_heatmap_svg(out_dir / "eps_ref_mse_heatmap.svg", ref_mse, timesteps)
    write_mean_curve_svg(out_dir / "checkpoint_mean_ref_mse.svg", ckpt_mean)

    has_direction_metrics = cosine_transition is not None and progress_transition is not None
    transition_direction_rows = []
    snapshot_direction_rows = []
    if has_direction_metrics:
        assert cosine_transition is not None
        assert progress_transition is not None
        for c in range(cosine_transition.shape[0]):
            cos_row = cosine_transition[c]
            progress_row = progress_transition[c]
            transition_direction_rows.append(
                {
                    "transition_index": c,
                    "from_ckpt_index": c,
                    "to_ckpt_index": c + 1,
                    "from_epoch": 4 * (c + 1),
                    "to_epoch": 4 * (c + 2),
                    "cosine_mean": float(np.mean(cos_row)),
                    "cosine_median": float(np.median(cos_row)),
                    "cosine_positive_fraction": float(np.mean(cos_row > 0.0)),
                    "progress_mean": float(np.mean(progress_row)),
                    "progress_median": float(np.median(progress_row)),
                    "progress_positive_fraction": float(np.mean(progress_row > 0.0)),
                }
            )
        for s in range(cosine_transition.shape[1]):
            cos_col = cosine_transition[:, s]
            progress_col = progress_transition[:, s]
            snapshot_direction_rows.append(
                {
                    "snapshot_column": s,
                    "snapshot_position": int(positions[s]) if s < len(positions) else s,
                    "timestep": int(timesteps[s]) if s < len(timesteps) else -1,
                    "cosine_mean": float(np.mean(cos_col)),
                    "cosine_median": float(np.median(cos_col)),
                    "cosine_positive_fraction": float(np.mean(cos_col > 0.0)),
                    "progress_mean": float(np.mean(progress_col)),
                    "progress_median": float(np.median(progress_col)),
                    "progress_positive_fraction": float(np.mean(progress_col > 0.0)),
                }
            )
        write_csv(out_dir / "transition_direction_summary.csv", transition_direction_rows)
        write_csv(out_dir / "snapshot_direction_summary.csv", snapshot_direction_rows)
        write_signed_heatmap_svg(
            out_dir / "eps_to_ref_cosine_heatmap.svg",
            cosine_transition,
            timesteps,
            title="Cosine of checkpoint eps update toward reference eps",
        )
        write_signed_heatmap_svg(
            out_dir / "eps_to_ref_progress_heatmap.svg",
            progress_transition,
            timesteps,
            title="Ref-MSE progress by checkpoint transition",
        )

    summary = {
        "weights_npz": str(Path(args.weights_npz).expanduser()),
        "shape": list(ref_mse.shape),
        "checkpoint_mean_start": float(ckpt_mean[0]),
        "checkpoint_mean_end": float(ckpt_mean[-1]),
        "checkpoint_mean_end_over_start": float(ckpt_mean[-1] / max(ckpt_mean[0], 1e-30)),
        "checkpoint_mean_decrease_fraction": float(np.mean(np.diff(ckpt_mean) <= 0.0)),
        "checkpoint_mean_spearman": corr(np.arange(len(ckpt_mean)), ckpt_mean, spearman=True),
        "num_snapshots_end_lower_than_start": int(sum(float(r["end_ref_mse"]) < float(r["start_ref_mse"]) for r in snapshot_rows)),
        "num_snapshots_mostly_decreasing": int(sum(float(r["decrease_fraction"]) >= 0.75 for r in snapshot_rows)),
        "worst_nonconverging_snapshots": sorted(snapshot_rows, key=lambda r: float(r["end_over_start"]), reverse=True)[:10],
        "best_converging_snapshots": sorted(snapshot_rows, key=lambda r: float(r["end_over_start"]))[:10],
        "has_direction_metrics": bool(has_direction_metrics),
    }
    if has_direction_metrics:
        assert cosine_transition is not None
        assert progress_transition is not None
        summary.update(
            {
                "cosine_shape": list(cosine_transition.shape),
                "cosine_mean": float(np.mean(cosine_transition)),
                "cosine_median": float(np.median(cosine_transition)),
                "cosine_positive_fraction": float(np.mean(cosine_transition > 0.0)),
                "progress_mean": float(np.mean(progress_transition)),
                "progress_median": float(np.median(progress_transition)),
                "progress_positive_fraction": float(np.mean(progress_transition > 0.0)),
                "num_snapshots_cosine_positive_fraction_ge_0p75": int(
                    sum(float(r["cosine_positive_fraction"]) >= 0.75 for r in snapshot_direction_rows)
                ),
                "num_snapshots_progress_positive_fraction_ge_0p75": int(
                    sum(float(r["progress_positive_fraction"]) >= 0.75 for r in snapshot_direction_rows)
                ),
                "worst_direction_snapshots": sorted(
                    snapshot_direction_rows,
                    key=lambda r: float(r["cosine_positive_fraction"]),
                )[:10],
                "best_direction_snapshots": sorted(
                    snapshot_direction_rows,
                    key=lambda r: float(r["cosine_positive_fraction"]),
                    reverse=True,
                )[:10],
            }
        )
    (out_dir / "ref_process_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[saved] {out_dir}")
    print(
        f"mean ref MSE start={summary['checkpoint_mean_start']:.6g} "
        f"end={summary['checkpoint_mean_end']:.6g} "
        f"end/start={summary['checkpoint_mean_end_over_start']:.6g} "
        f"decrease_fraction={summary['checkpoint_mean_decrease_fraction']:.3f}"
    )
    print(
        f"snapshots end<start={summary['num_snapshots_end_lower_than_start']}/{ref_mse.shape[1]} | "
        f"mostly_decreasing={summary['num_snapshots_mostly_decreasing']}/{ref_mse.shape[1]}"
    )
    if has_direction_metrics:
        print(
            f"direction cos_mean={summary['cosine_mean']:.4f} "
            f"cos_pos_frac={summary['cosine_positive_fraction']:.3f} | "
            f"progress_pos_frac={summary['progress_positive_fraction']:.3f}"
        )


if __name__ == "__main__":
    main()
