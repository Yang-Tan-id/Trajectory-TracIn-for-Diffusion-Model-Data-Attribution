from __future__ import annotations

"""Export LDS scatter SVGs with red/green LDS-sign coloring."""

import argparse
import csv
import math
import re
import shutil
from pathlib import Path

from aggregate_lds_by_seed import _mean, _spearman, _std


def _escape(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _seed_from_model_name(name: str) -> int | None:
    match = re.search(r"_seed_(\d+)$", name)
    return int(match.group(1)) if match else None


def _query_label(query_dir: Path) -> str:
    name = query_dir.name
    if name.startswith("query_"):
        return name
    if name.endswith("8") and name[:-1].startswith("query_"):
        return name
    return name


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _bounds(values: list[float]) -> tuple[float, float]:
    lo = min(values)
    hi = max(values)
    if lo == hi:
        pad = abs(lo) * 0.05 + 1.0
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.05
    return lo - pad, hi + pad


def _lds_color(lds_value: float) -> str:
    if math.isnan(lds_value):
        return "#7f7f7f"
    return "#2ca02c" if lds_value >= 0.0 else "#d62728"


def _fmt(value: float, width: int = 9) -> str:
    if math.isnan(float(value)):
        return f"{'nan':>{width}}"
    return f"{float(value):>{width}.3g}"


def _write_scatter_grid(path: Path, groups: list[dict], *, title: str) -> None:
    if not groups:
        return
    cols = 4 if len(groups) > 1 else 1
    rows = math.ceil(len(groups) / cols)
    panel_w = 300
    panel_h = 240
    margin_l = 52
    margin_t = 42
    margin_r = 18
    margin_b = 44
    width = cols * panel_w
    height = rows * panel_h + 76
    all_pred = [x for group in groups for x in group["pred"]]
    all_true = [y for group in groups for y in group["true"]]
    x0, x1 = _bounds(all_pred)
    y0, y1 = _bounds(all_true)

    def sx(x: float, col: int) -> float:
        left = col * panel_w + margin_l
        right = (col + 1) * panel_w - margin_r
        return left + (x - x0) / (x1 - x0) * (right - left)

    def sy(y: float, row: int) -> float:
        top = 54 + row * panel_h + margin_t
        bottom = 54 + (row + 1) * panel_h - margin_b
        return bottom - (y - y0) / (y1 - y0) * (bottom - top)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" font-family="Arial" font-size="20">{_escape(title)}</text>',
        f'<circle cx="{width - 174}" cy="28" r="4" fill="#2ca02c" fill-opacity="0.72"/>',
        f'<text x="{width - 164}" y="32" font-family="Arial" font-size="12">LDS >= 0</text>',
        f'<circle cx="{width - 86}" cy="28" r="4" fill="#d62728" fill-opacity="0.72"/>',
        f'<text x="{width - 76}" y="32" font-family="Arial" font-size="12">LDS &lt; 0</text>',
    ]
    for i, group in enumerate(groups):
        row = i // cols
        col = i % cols
        left = col * panel_w + margin_l
        right = (col + 1) * panel_w - margin_r
        top = 54 + row * panel_h + margin_t
        bottom = 54 + (row + 1) * panel_h - margin_b
        parts.append(f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" fill="#fbfbfb" stroke="#cccccc"/>')
        for frac in (0.25, 0.5, 0.75):
            gx = left + frac * (right - left)
            gy = top + frac * (bottom - top)
            parts.append(f'<line x1="{gx}" x2="{gx}" y1="{top}" y2="{bottom}" stroke="#e6e6e6"/>')
            parts.append(f'<line x1="{left}" x2="{right}" y1="{gy}" y2="{gy}" stroke="#e6e6e6"/>')
        point_color = _lds_color(float(group["lds"]))
        for x, y in zip(group["pred"], group["true"]):
            parts.append(
                f'<circle cx="{sx(x, col):.2f}" cy="{sy(y, row):.2f}" r="3.2" fill="{point_color}" fill-opacity="0.72"/>'
            )
        lds_pct = group["lds"] * 100.0 if not math.isnan(group["lds"]) else float("nan")
        label = f"seed {group['seed']} | LDS={lds_pct:.2f}% | n={len(group['pred'])}"
        parts.append(
            f'<text x="{(left + right) / 2:.1f}" y="{top - 13}" text-anchor="middle" font-family="Arial" font-size="13">{_escape(label)}</text>'
        )
        parts.append(f'<text x="{left}" y="{bottom + 18}" font-family="Arial" font-size="10">{x0:.3g}</text>')
        parts.append(f'<text x="{right}" y="{bottom + 18}" text-anchor="end" font-family="Arial" font-size="10">{x1:.3g}</text>')
        parts.append(f'<text x="{left - 8}" y="{bottom}" text-anchor="end" font-family="Arial" font-size="10">{y0:.3g}</text>')
        parts.append(f'<text x="{left - 8}" y="{top + 4}" text-anchor="end" font-family="Arial" font-size="10">{y1:.3g}</text>')
    parts.append(f'<text x="{width / 2:.1f}" y="{height - 22}" text-anchor="middle" font-family="Arial" font-size="14">Predicted sum of attribution scores</text>')
    parts.append(
        f'<text x="16" y="{height / 2:.1f}" text-anchor="middle" font-family="Arial" font-size="14" transform="rotate(-90 16 {height / 2:.1f})">True counterfactual f</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def _aggregate_group(group_dir: Path) -> tuple[list[dict], dict] | None:
    seed_groups = []
    for seed_dir in sorted(group_dir.glob("m_*_k_*_seed_*"), key=lambda p: (_seed_from_model_name(p.name) or 0, p.name)):
        csv_path = seed_dir / "lds_results.csv"
        if not csv_path.is_file():
            continue
        rows = _read_rows(csv_path)
        if not rows:
            continue
        pred = [float(row["pred_sum_tau"]) for row in rows]
        true = [float(row["true_f"]) for row in rows]
        seed = _seed_from_model_name(seed_dir.name)
        seed_groups.append(
            {
                "seed": seed if seed is not None else seed_dir.name,
                "pred": pred,
                "true": true,
                "lds": _spearman(pred, true),
                "pred_mean": _mean(pred),
                "pred_std": _std(pred),
                "true_mean": _mean(true),
                "true_std": _std(true),
                "source_dir": str(seed_dir),
            }
        )
    if not seed_groups:
        return None
    lds_values = [group["lds"] for group in seed_groups]
    summary = {
        "num_groups": len(seed_groups),
        "num_points": sum(len(group["pred"]) for group in seed_groups),
        "lds_percent_mean": 100.0 * _mean(lds_values),
        "lds_percent_std": 100.0 * _std(lds_values),
    }
    return seed_groups, summary


def _all_points_group(seed_groups: list[dict]) -> list[dict]:
    pred = [x for group in seed_groups for x in group["pred"]]
    true = [y for group in seed_groups for y in group["true"]]
    return [{"seed": "all", "pred": pred, "true": true, "lds": _spearman(pred, true)}]


def _summary_block(*, query: str, algorithm: str, target: str, seed_groups: list[dict], summary: dict) -> str:
    lines = [
        "=" * 90,
        f"{query} {algorithm} {target}",
        "seed | LDS% | n | pred_mean +/- pred_std | true_mean +/- true_std",
        "-" * 90,
    ]
    for group in seed_groups:
        lds_pct = 100.0 * float(group["lds"]) if not math.isnan(float(group["lds"])) else float("nan")
        lines.append(
            f"{str(group['seed']):>4} | "
            f"{lds_pct:>7.2f} | "
            f"{len(group['pred']):>2} | "
            f"{_fmt(group['pred_mean'], 11)} +/- {_fmt(group['pred_std'], 9)} | "
            f"{_fmt(group['true_mean'], 9)} +/- {_fmt(group['true_std'], 9)}"
        )
    lines.extend(
        [
            "-" * 90,
            (
                "per-seed LDS mean +/- std: "
                f"{summary['lds_percent_mean']:.2f}% +/- {summary['lds_percent_std']:.2f}% "
                f"(groups={summary['num_groups']})"
            ),
        ]
    )
    return "\n".join(lines)


def export_group(group_dir: Path, out_dir: Path, *, query: str, algorithm: str, target: str) -> str | None:
    aggregated = _aggregate_group(group_dir)
    if aggregated is None:
        return None
    seed_groups, summary = aggregated
    prefix = f"{query}__{algorithm}__{target}"
    per_seed_svg = out_dir / f"{prefix}__per_seed_scatter_grid.svg"
    all_points_svg = out_dir / f"{prefix}__all_points_scatter.svg"
    _write_scatter_grid(per_seed_svg, seed_groups, title=f"{query} | {algorithm} | {target}")
    _write_scatter_grid(all_points_svg, _all_points_group(seed_groups), title=f"{query} | {algorithm} | {target} | all points")
    summary["query"] = query
    summary["algorithm"] = algorithm
    summary["target_function"] = target
    summary["source_group_dir"] = str(group_dir)
    return _summary_block(query=query, algorithm=algorithm, target=target, seed_groups=seed_groups, summary=summary)


def discover_groups(eval_root: Path, *, target: str, prediction_dir: str, query_suffix: str | None) -> list[tuple[Path, str, str]]:
    groups = []
    for query_dir in sorted(eval_root.glob("query_*")):
        if query_suffix and not query_dir.name.endswith(query_suffix):
            continue
        lds_root = query_dir / "initial_seed_42" / "lds"
        if not lds_root.is_dir():
            continue
        for alg_dir in sorted(path for path in lds_root.iterdir() if path.is_dir()):
            group_dir = alg_dir / target / prediction_dir
            if group_dir.is_dir():
                groups.append((group_dir, _query_label(query_dir), alg_dir.name))
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--target-function", default="simple_loss")
    parser.add_argument("--prediction-dir", default="pred_kept_sign_m1")
    parser.add_argument("--query-suffix", default=None, help="Only export query dirs ending with this suffix, e.g. 8.")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    for group_dir, query, algorithm in discover_groups(
        Path(args.eval_root).expanduser().resolve(),
        target=args.target_function,
        prediction_dir=args.prediction_dir,
        query_suffix=args.query_suffix,
    ):
        line = export_group(group_dir, out_dir, query=query, algorithm=algorithm, target=args.target_function)
        if line:
            lines.append(line)

    if not lines:
        raise SystemExit("No matching LDS result groups found.")
    (out_dir / "SUMMARY.txt").write_text("\n".join(lines) + "\n")
    print(f"Exported {len(lines)} groups to {out_dir}")


if __name__ == "__main__":
    main()
