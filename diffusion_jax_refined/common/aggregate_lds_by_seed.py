from __future__ import annotations

"""Aggregate per-seed LDS eval folders into summary tables and scatter grids."""

import argparse
import csv
import json
import math
import re
from pathlib import Path


def _rank_average_ties(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1)
        for idx in order[i:j]:
            ranks[idx] = rank
        i = j
    return ranks


def _spearman(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("spearman inputs must have the same length")
    if len(a) < 2:
        return float("nan")
    ra = _rank_average_ties([float(x) for x in a])
    rb = _rank_average_ties([float(x) for x in b])
    ma = sum(ra) / len(ra)
    mb = sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    if da == 0 or db == 0:
        return float("nan")
    return num / (da * db)


def _mean(values: list[float]) -> float:
    clean = [float(x) for x in values if not math.isnan(float(x))]
    return sum(clean) / len(clean) if clean else float("nan")


def _std(values: list[float]) -> float:
    clean = [float(x) for x in values if not math.isnan(float(x))]
    if not clean:
        return float("nan")
    mu = sum(clean) / len(clean)
    return math.sqrt(sum((x - mu) ** 2 for x in clean) / len(clean))


def _seed_from_model_name(name: str) -> int | None:
    match = re.search(r"(?:_subset)?_seed_(\d+)$", name)
    return int(match.group(1)) if match else None


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def _data_bounds(series: list[float]) -> tuple[float, float]:
    lo = min(series)
    hi = max(series)
    if lo == hi:
        pad = abs(lo) * 0.05 + 1.0
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.05
    return lo - pad, hi + pad


def _svg_scatter_grid(
    path: Path,
    seed_points: list[dict],
    *,
    title: str,
    xlabel: str = "Predicted sum of attribution scores",
    ylabel: str = "True counterfactual f",
) -> None:
    if not seed_points:
        return
    cols = 4
    rows = math.ceil(len(seed_points) / cols)
    panel_w = 300
    panel_h = 240
    margin_l = 52
    margin_t = 42
    margin_r = 18
    margin_b = 44
    width = cols * panel_w
    height = rows * panel_h + 54
    all_pred = [x for item in seed_points for x in item["pred"]]
    all_true = [y for item in seed_points for y in item["true"]]
    x0, x1 = _data_bounds(all_pred)
    y0, y1 = _data_bounds(all_true)

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
    ]
    for i, item in enumerate(seed_points):
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
        for x, y in zip(item["pred"], item["true"]):
            parts.append(
                f'<circle cx="{sx(x, col):.2f}" cy="{sy(y, row):.2f}" r="3.2" fill="#1f77b4" fill-opacity="0.72"/>'
            )
        lds_pct = item["lds"] * 100.0 if not math.isnan(item["lds"]) else float("nan")
        label = f"seed {item['seed']} | LDS={lds_pct:.2f}% | n={len(item['pred'])}"
        parts.append(
            f'<text x="{(left + right) / 2:.1f}" y="{top - 13}" text-anchor="middle" font-family="Arial" font-size="13">{_escape(label)}</text>'
        )
        parts.append(f'<text x="{left}" y="{bottom + 18}" font-family="Arial" font-size="10">{x0:.3g}</text>')
        parts.append(f'<text x="{right}" y="{bottom + 18}" text-anchor="end" font-family="Arial" font-size="10">{x1:.3g}</text>')
        parts.append(
            f'<text x="{left - 8}" y="{bottom}" text-anchor="end" font-family="Arial" font-size="10">{y0:.3g}</text>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{top + 4}" text-anchor="end" font-family="Arial" font-size="10">{y1:.3g}</text>'
        )
    parts.append(f'<text x="{width / 2:.1f}" y="{height - 10}" text-anchor="middle" font-family="Arial" font-size="14">{_escape(xlabel)}</text>')
    parts.append(
        f'<text x="16" y="{height / 2:.1f}" text-anchor="middle" font-family="Arial" font-size="14" transform="rotate(-90 16 {height / 2:.1f})">{_escape(ylabel)}</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def _svg_all_points(path: Path, seed_points: list[dict], *, title: str) -> None:
    merged = {
        "seed": "all",
        "pred": [x for item in seed_points for x in item["pred"]],
        "true": [y for item in seed_points for y in item["true"]],
        "lds": _spearman(
            [x for item in seed_points for x in item["pred"]],
            [y for item in seed_points for y in item["true"]],
        ),
    }
    _svg_scatter_grid(path, [merged], title=title)


def _escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def aggregate_group(group_dir: Path, *, model_glob: str, output_name: str) -> Path | None:
    seed_points = []
    all_rows = []
    summary_rows = []
    for seed_dir in sorted(group_dir.glob(model_glob), key=lambda p: (_seed_from_model_name(p.name) or 0, p.name)):
        csv_path = seed_dir / "lds_results.csv"
        if not csv_path.is_file():
            continue
        rows = _read_rows(csv_path)
        if not rows:
            continue
        pred = [float(row["pred_sum_tau"]) for row in rows]
        true = [float(row["true_f"]) for row in rows]
        seed = _seed_from_model_name(seed_dir.name)
        lds = _spearman(pred, true)
        seed_points.append({"seed": seed if seed is not None else seed_dir.name, "pred": pred, "true": true, "lds": lds})
        summary_rows.append(
            {
                "seed": seed if seed is not None else seed_dir.name,
                "n": len(rows),
                "lds_spearman": lds,
                "lds_percent": 100.0 * lds if not math.isnan(lds) else float("nan"),
                "pred_mean": _mean(pred),
                "pred_std": _std(pred),
                "true_mean": _mean(true),
                "true_std": _std(true),
                "source_dir": str(seed_dir),
            }
        )
        for row in rows:
            out = dict(row)
            out["lds_seed"] = seed if seed is not None else seed_dir.name
            out["source_dir"] = str(seed_dir)
            all_rows.append(out)
    if not seed_points:
        return None

    seeds = [int(item["seed"]) for item in seed_points if isinstance(item["seed"], int)]
    if seeds:
        default_name = f"aggregate_seeds_{min(seeds)}_{max(seeds)}"
    else:
        default_name = "aggregate"
    out_dir = group_dir / (output_name or default_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "per_seed_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "n",
                "lds_spearman",
                "lds_percent",
                "pred_mean",
                "pred_std",
                "true_mean",
                "true_std",
                "source_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    all_csv = out_dir / "all_seed_points.csv"
    all_fields = sorted({key for row in all_rows for key in row})
    with all_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(all_rows)

    lds_values = [float(row["lds_spearman"]) for row in summary_rows]
    payload = {
        "source_group_dir": str(group_dir),
        "num_seeds": len(summary_rows),
        "num_points": len(all_rows),
        "lds_percent_mean": 100.0 * _mean(lds_values),
        "lds_percent_std": 100.0 * _std(lds_values),
        "pred_mean_mean": _mean([float(row["pred_mean"]) for row in summary_rows]),
        "true_mean_mean": _mean([float(row["true_mean"]) for row in summary_rows]),
        "summary_csv": str(summary_csv),
        "all_seed_points_csv": str(all_csv),
        "per_seed": summary_rows,
    }
    (out_dir / "per_seed_summary.json").write_text(json.dumps(payload, indent=2))

    _svg_scatter_grid(out_dir / "per_seed_scatter_grid.svg", seed_points, title=group_dir.as_posix())
    _svg_all_points(out_dir / "all_points_scatter.svg", seed_points, title=f"All seeds | {group_dir.as_posix()}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate per-seed LDS results.")
    parser.add_argument("--eval-root", required=True, help="Path to result/<experiment>/eval")
    parser.add_argument("--target-function", required=True)
    parser.add_argument("--lds-m", type=int, required=True)
    parser.add_argument("--lds-k", type=int, required=True)
    parser.add_argument("--initial-seed", default="42")
    parser.add_argument("--queries", nargs="*", default=None, help="Optional query folder names, e.g. query_horse")
    parser.add_argument("--algorithms", nargs="*", default=None)
    parser.add_argument(
        "--prediction-dir",
        default=None,
        help="Optional prediction directory under target-function, e.g. pred_kept_sign_m1 or pred_removed_sign_1.",
    )
    parser.add_argument("--eval-kind", default="lds", choices=["lds", "lds_unprompted"])
    parser.add_argument("--model-glob", default=None)
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()

    eval_root = Path(args.eval_root)
    query_dirs = [eval_root / q for q in args.queries] if args.queries else sorted(eval_root.glob("query_*"))
    model_glob = args.model_glob or f"m_{args.lds_m}_k_{args.lds_k}_seed_*"
    output_name = args.output_name or f"aggregate_m_{args.lds_m}_k_{args.lds_k}"
    created = []
    for query_dir in query_dirs:
        lds_root = query_dir / f"initial_seed_{args.initial_seed}" / args.eval_kind
        if not lds_root.is_dir():
            continue
        alg_dirs = [lds_root / a for a in args.algorithms] if args.algorithms else sorted(p for p in lds_root.iterdir() if p.is_dir())
        for alg_dir in alg_dirs:
            target_dir = alg_dir / args.target_function
            if not target_dir.is_dir():
                continue
            if args.prediction_dir:
                candidate = target_dir / args.prediction_dir
                group_dirs = [candidate] if candidate.is_dir() and any(candidate.glob(model_glob)) else []
            else:
                group_dirs = []
                if any(target_dir.glob(model_glob)):
                    group_dirs.append(target_dir)
                group_dirs.extend(
                    child for child in sorted(target_dir.iterdir())
                    if child.is_dir() and any(child.glob(model_glob))
                )
            for group_dir in group_dirs:
                out = aggregate_group(group_dir, model_glob=model_glob, output_name=output_name)
                if out is not None:
                    created.append(str(out))
                    print(f"Saved per-seed aggregate to {out}")
    if not created:
        raise SystemExit("No per-seed LDS result folders matched the requested filters.")


if __name__ == "__main__":
    main()
