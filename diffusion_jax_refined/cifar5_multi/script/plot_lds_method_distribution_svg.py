#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape


METHODS = [
    ("aligned", "Aligned TrajTracIn\n(ours by mistake)", "#4C78A8"),
    ("shared", "Shared TrajTracIn\n(ours previously)", "#F58518"),
    ("das", "DAS", "#54A24B"),
]

TARGETS = [
    ("endpoint_counterfactual", "Endpoint CF"),
    ("noise_trajectory", "Noise trajectory"),
    ("traj_counterfactual", "Trajectory CF"),
    ("simple_loss", "Simple loss"),
]


ROW_RE = re.compile(
    r"\s*(10\d\d)\s+"
    r"(query_\S+)\s+"
    r"(endpoint_counterfactual|noise_trajectory|simple_loss|traj_counterfactual)\s+"
    r"([-0-9.]+)\s+"
    r"([-0-9.]+)\s+"
    r"([-0-9.]+)"
)


def parse_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text().splitlines():
        match = ROW_RE.match(line)
        if not match:
            continue
        seed, query, target, aligned, shared, das = match.groups()
        rows.append(
            {
                "seed": int(seed),
                "query": query,
                "target": target,
                "aligned": float(aligned),
                "shared": float(shared),
                "das": float(das),
            }
        )
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


def fmt(value: float) -> str:
    return f"{value:.3f}"


def y_ticks(y_min: float, y_max: float) -> list[float]:
    ticks = []
    start = math.ceil(y_min * 10) / 10
    end = math.floor(y_max * 10) / 10
    t = start
    while t <= end + 1e-9:
        ticks.append(round(t, 1))
        t += 0.1
    return ticks


def render_svg(rows: list[dict[str, object]], output: Path, title: str) -> None:
    values = [float(row[key]) for row in rows for key, _, _ in METHODS]
    y_min = min(-0.1, math.floor((min(values) - 0.02) * 10) / 10)
    y_max = max(0.3, math.ceil((max(values) + 0.02) * 10) / 10)

    width = 1480
    height = 980
    left = 180
    right = 42
    top = 126
    bottom = 72
    gap_x = 28
    gap_y = 42
    panel_w = (width - left - right - gap_x * (len(TARGETS) - 1)) / len(TARGETS)
    panel_h = (height - top - bottom - gap_y * (len(METHODS) - 1)) / len(METHODS)

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        target = str(row["target"])
        for key, _, _ in METHODS:
            grouped[(key, target)].append(float(row[key]))

    def y_scale(value: float, py: float) -> float:
        return py + panel_h - (value - y_min) / (y_max - y_min) * panel_h

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-family="Arial" font-size="24" font-weight="700">{escape(title)}</text>',
        f'<text x="{width / 2}" y="58" text-anchor="middle" font-family="Arial" font-size="13" fill="#555">Each dot is one prompt-seed; box spans Q1-Q3; line is median; label reports mean.</text>',
    ]

    legend_x = width / 2 - 285
    for i, (key, label, color) in enumerate(METHODS):
        x = legend_x + i * 230
        one_line = label.splitlines()[0]
        parts.extend(
            [
                f'<circle cx="{x}" cy="82" r="5" fill="{color}"/>',
                f'<text x="{x + 12}" y="86" font-family="Arial" font-size="12" fill="#333">{escape(one_line)}</text>',
            ]
        )

    ticks = y_ticks(y_min, y_max)
    zero_y_cache: dict[tuple[int, int], float] = {}

    for c, (target, target_label) in enumerate(TARGETS):
        px = left + c * (panel_w + gap_x)
        parts.append(
            f'<text x="{px + panel_w / 2}" y="{top - 18}" text-anchor="middle" font-family="Arial" font-size="15" font-weight="700">{escape(target_label)}</text>'
        )

    for r, (method, method_label, color) in enumerate(METHODS):
        py = top + r * (panel_h + gap_y)
        label_lines = method_label.splitlines()
        base_y = py + panel_h / 2 - (len(label_lines) - 1) * 8
        for line_i, line in enumerate(label_lines):
            size = 14 if line_i == 0 else 11
            weight = "700" if line_i == 0 else "400"
            fill = "#222" if line_i == 0 else "#555"
            parts.append(
                f'<text x="{left - 22}" y="{base_y + line_i * 16:.1f}" text-anchor="end" font-family="Arial" font-size="{size}" font-weight="{weight}" fill="{fill}">{escape(line)}</text>'
            )

        for c, (target, _) in enumerate(TARGETS):
            px = left + c * (panel_w + gap_x)
            zero_y = y_scale(0.0, py)
            zero_y_cache[(r, c)] = zero_y
            parts.append(f'<rect x="{px:.1f}" y="{py:.1f}" width="{panel_w:.1f}" height="{panel_h:.1f}" fill="#fafafa" stroke="#ddd"/>')
            for tick in ticks:
                ty = y_scale(tick, py)
                stroke = "#333" if abs(tick) < 1e-9 else "#e6e6e6"
                dash = ' stroke-dasharray="4 4"' if abs(tick) < 1e-9 else ""
                parts.append(f'<line x1="{px:.1f}" y1="{ty:.1f}" x2="{px + panel_w:.1f}" y2="{ty:.1f}" stroke="{stroke}"{dash}/>')
                if c == 0:
                    parts.append(
                        f'<text x="{px - 8:.1f}" y="{ty + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#555">{tick:.1f}</text>'
                    )

            vals = grouped[(method, target)]
            mean = sum(vals) / len(vals)
            q1 = percentile(vals, 0.25)
            med = percentile(vals, 0.50)
            q3 = percentile(vals, 0.75)
            vmin = min(vals)
            vmax = max(vals)

            center_x = px + panel_w / 2
            bar_w = 46
            mean_y = y_scale(mean, py)
            q1_y = y_scale(q1, py)
            q3_y = y_scale(q3, py)
            rect_y = min(q1_y, q3_y)
            rect_h = abs(q3_y - q1_y)
            parts.append(
                f'<rect x="{center_x - bar_w / 2:.1f}" y="{rect_y:.1f}" width="{bar_w}" height="{rect_h:.1f}" fill="{color}" fill-opacity="0.24" stroke="{color}" stroke-width="1.5"/>'
            )
            parts.append(
                f'<line x1="{center_x:.1f}" y1="{y_scale(vmin, py):.1f}" x2="{center_x:.1f}" y2="{y_scale(vmax, py):.1f}" stroke="{color}" stroke-opacity="0.45" stroke-width="1.2"/>'
            )

            med_y = y_scale(med, py)
            parts.append(
                f'<line x1="{center_x - bar_w / 2:.1f}" y1="{med_y:.1f}" x2="{center_x + bar_w / 2:.1f}" y2="{med_y:.1f}" stroke="{color}" stroke-width="2.4"/>'
            )

            for i, value in enumerate(vals):
                jitter = ((i % 9) - 4) * 6 + ((i // 9) % 2) * 2.2
                x = center_x + jitter
                y = y_scale(value, py)
                parts.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.6" fill="{color}" fill-opacity="0.72" stroke="white" stroke-width="0.8">'
                    f'<title>{escape(method)} {escape(target)} LDS={fmt(value)}</title>'
                    '</circle>'
                )

            parts.append(
                f'<circle cx="{center_x:.1f}" cy="{mean_y:.1f}" r="5.0" fill="{color}" stroke="white" stroke-width="1.2"/>'
            )
            parts.append(
                f'<text x="{center_x:.1f}" y="{py + 18:.1f}" text-anchor="middle" font-family="Arial" font-size="11" fill="#333">mean={fmt(mean)}</text>'
            )

    parts.extend(
        [
            f'<text x="{left - 44}" y="{top + (height - top - bottom) / 2}" transform="rotate(-90 {left - 44} {top + (height - top - bottom) / 2})" text-anchor="middle" font-family="Arial" font-size="13" fill="#333">LDS Spearman</text>',
            "</svg>",
        ]
    )
    output.write_text("\n".join(parts) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot LDS distributions by method and target as a pure SVG.")
    parser.add_argument("input", type=Path, help="Text table with aligned/shared/das columns.")
    parser.add_argument("-o", "--output", type=Path, default=Path("aligned_filtered_method_distribution.svg"))
    parser.add_argument("--title", default="Filtered LDS Spearman Distribution by Method and Target")
    args = parser.parse_args()

    rows = parse_rows(args.input)
    if not rows:
        raise SystemExit(f"No rows parsed from {args.input}")
    render_svg(rows, args.output, args.title)
    print(f"wrote {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
