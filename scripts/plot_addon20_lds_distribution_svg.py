#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape


OLD_RE = re.compile(
    r"\s*(10\d\d)\s+"
    r"(query_\S+)\s+"
    r"(endpoint_counterfactual|noise_trajectory|simple_loss|traj_counterfactual)\s+"
    r"([-0-9.]+)\s+"
    r"([-0-9.]+)\s+"
    r"([-0-9.]+)"
)

COMBINED_RE = re.compile(
    r"\s*(10\d\d)\s+"
    r"(query_\S+)\s+"
    r"(endpoint_counterfactual|noise_trajectory|simple_loss|traj_counterfactual)\s+"
    r"([-0-9.]+)\s+"
    r"([-0-9.]+)\s+"
    r"([-0-9.]+)\s+"
    r"([-0-9.]+)"
)

TARGETS = [
    ("endpoint_counterfactual", "Endpoint CF"),
    ("noise_trajectory", "Noise trajectory"),
    ("traj_counterfactual", "Trajectory CF"),
    ("simple_loss", "Simple loss"),
]

METHODS = [
    ("shared", "shared traj tracin\n(ours previously)", "#F58518"),
    ("aligned10", "aligned traj tracin (10)\n(ours by mistake)", "#4C78A8"),
    ("aligned20", "aligned traj tracin (20)", "#B279A2"),
    ("das", "DAS", "#54A24B"),
]


def parse_old(path: Path) -> dict[tuple[int, str, str], dict[str, float]]:
    rows: dict[tuple[int, str, str], dict[str, float]] = {}
    for line in path.read_text().splitlines():
        match = OLD_RE.match(line)
        if not match:
            continue
        seed, query, target, aligned, shared, das = match.groups()
        rows[(int(seed), query, target)] = {
            "aligned10": float(aligned),
            "shared": float(shared),
            "das": float(das),
        }
    return rows


def parse_combined(path: Path) -> dict[tuple[int, str, str], float]:
    rows: dict[tuple[int, str, str], float] = {}
    for line in path.read_text().splitlines():
        match = COMBINED_RE.match(line)
        if not match:
            continue
        seed, query, target, _base10, _addon10, combined20, _das = match.groups()
        rows[(int(seed), query, target)] = float(combined20)
    return rows


def percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * pct
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def ticks(y_min: float, y_max: float) -> list[float]:
    out = []
    t = math.ceil(y_min * 10) / 10
    end = math.floor(y_max * 10) / 10
    while t <= end + 1e-9:
        out.append(round(t, 1))
        t += 0.1
    return out


def render_svg(rows: list[dict[str, object]], output: Path) -> None:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        for key, _, _ in METHODS:
            grouped[(key, str(row["target"]))].append(float(row[key]))

    all_values = [v for values in grouped.values() for v in values]
    y_min = min(-0.1, math.floor((min(all_values) - 0.025) * 10) / 10)
    y_max = max(0.3, math.ceil((max(all_values) + 0.025) * 10) / 10)

    width = 1580
    height = 1120
    left = 235
    right = 54
    top = 132
    bottom = 76
    gap_x = 30
    gap_y = 44
    panel_w = (width - left - right - gap_x * (len(TARGETS) - 1)) / len(TARGETS)
    panel_h = (height - top - bottom - gap_y * (len(METHODS) - 1)) / len(METHODS)

    def y_scale(value: float, py: float) -> float:
        return py + panel_h - (value - y_min) / (y_max - y_min) * panel_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="36" text-anchor="middle" font-family="Arial" font-size="25" font-weight="700">LDS Spearman by target and method</text>',
        f'<text x="{width / 2:.1f}" y="62" text-anchor="middle" font-family="Arial" font-size="13" fill="#555">Dots are query seeds; box spans Q1-Q3; center line is median; text label is mean.</text>',
    ]

    legend_width = 1040
    legend_x = (width - legend_width) / 2
    for i, (_key, label, color) in enumerate(METHODS):
        x = legend_x + i * 270
        parts.append(f'<circle cx="{x:.1f}" cy="88" r="5.5" fill="{color}"/>')
        parts.append(
            f'<text x="{x + 13:.1f}" y="92" font-family="Arial" font-size="12" fill="#333">{escape(label.splitlines()[0])}</text>'
        )

    for c, (target, label) in enumerate(TARGETS):
        px = left + c * (panel_w + gap_x)
        parts.append(
            f'<text x="{px + panel_w / 2:.1f}" y="{top - 18:.1f}" text-anchor="middle" font-family="Arial" font-size="15" font-weight="700">{escape(label)}</text>'
        )

    for r, (method, method_label, color) in enumerate(METHODS):
        py = top + r * (panel_h + gap_y)
        lines = method_label.splitlines()
        label_y = py + panel_h / 2 - (len(lines) - 1) * 8
        for i, line in enumerate(lines):
            parts.append(
                f'<text x="{left - 25}" y="{label_y + i * 17:.1f}" text-anchor="end" font-family="Arial" font-size="{14 if i == 0 else 12}" font-weight="{500 if i == 0 else 400}" fill="{"#222" if i == 0 else "#555"}">{escape(line)}</text>'
            )

        for c, (target, _label) in enumerate(TARGETS):
            px = left + c * (panel_w + gap_x)
            parts.append(f'<rect x="{px:.1f}" y="{py:.1f}" width="{panel_w:.1f}" height="{panel_h:.1f}" fill="#fbfbfb" stroke="#ddd"/>')
            for tick in ticks(y_min, y_max):
                ty = y_scale(tick, py)
                stroke = "#333" if abs(tick) < 1e-9 else "#e8e8e8"
                dash = ' stroke-dasharray="4 4"' if abs(tick) < 1e-9 else ""
                parts.append(f'<line x1="{px:.1f}" y1="{ty:.1f}" x2="{px + panel_w:.1f}" y2="{ty:.1f}" stroke="{stroke}"{dash}/>')
                if c == 0:
                    parts.append(f'<text x="{px - 8:.1f}" y="{ty + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#555">{tick:.1f}</text>')

            values = grouped[(method, target)]
            if not values:
                continue
            mean = sum(values) / len(values)
            q1 = percentile(values, 0.25)
            med = percentile(values, 0.50)
            q3 = percentile(values, 0.75)
            vmin = min(values)
            vmax = max(values)
            center_x = px + panel_w / 2
            box_w = 48
            q1_y = y_scale(q1, py)
            q3_y = y_scale(q3, py)
            med_y = y_scale(med, py)
            mean_y = y_scale(mean, py)

            parts.append(f'<line x1="{center_x:.1f}" y1="{y_scale(vmin, py):.1f}" x2="{center_x:.1f}" y2="{y_scale(vmax, py):.1f}" stroke="{color}" stroke-opacity="0.42" stroke-width="1.2"/>')
            parts.append(f'<rect x="{center_x - box_w / 2:.1f}" y="{min(q1_y, q3_y):.1f}" width="{box_w}" height="{abs(q3_y - q1_y):.1f}" fill="{color}" fill-opacity="0.22" stroke="{color}" stroke-width="1.5"/>')
            parts.append(f'<line x1="{center_x - box_w / 2:.1f}" y1="{med_y:.1f}" x2="{center_x + box_w / 2:.1f}" y2="{med_y:.1f}" stroke="{color}" stroke-width="2.4"/>')

            for i, value in enumerate(values):
                jitter = ((i % 7) - 3) * 6.2 + ((i // 7) % 2) * 2.4
                parts.append(
                    f'<circle cx="{center_x + jitter:.1f}" cy="{y_scale(value, py):.1f}" r="3.7" fill="{color}" fill-opacity="0.72" stroke="white" stroke-width="0.8"/>'
                )

            parts.append(f'<circle cx="{center_x:.1f}" cy="{mean_y:.1f}" r="5.2" fill="{color}" stroke="white" stroke-width="1.2"/>')
            parts.append(f'<text x="{center_x:.1f}" y="{py + 18:.1f}" text-anchor="middle" font-family="Arial" font-size="11" fill="#333">mean={mean:.3f}</text>')

    axis_x = left - 72
    axis_y = top + (height - top - bottom) / 2
    parts.append(
        f'<text x="{axis_x}" y="{axis_y}" transform="rotate(-90 {axis_x} {axis_y})" text-anchor="middle" font-family="Arial" font-size="13" fill="#333">LDS Spearman</text>'
    )
    parts.append("</svg>")
    output.write_text("\n".join(parts) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot shared/aligned10/aligned20/DAS LDS distributions.")
    parser.add_argument("--old", type=Path, required=True, help="Text table with aligned/shared/das columns.")
    parser.add_argument("--combined", type=Path, required=True, help="Text table with base10/addon10/combined20/das_fixed columns.")
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()

    old = parse_old(args.old)
    combined = parse_combined(args.combined)
    keys = sorted(set(old) & set(combined))
    if not keys:
        raise SystemExit("No overlapping rows between old and combined tables.")

    rows = []
    for key in keys:
        seed, query, target = key
        rows.append(
            {
                "seed": seed,
                "query": query,
                "target": target,
                "shared": old[key]["shared"],
                "aligned10": old[key]["aligned10"],
                "aligned20": combined[key],
                "das": old[key]["das"],
            }
        )

    render_svg(rows, args.output)
    print(f"wrote {args.output} with {len(rows)} rows and {len({(r['seed'], r['query']) for r in rows})} query seeds")


if __name__ == "__main__":
    main()
