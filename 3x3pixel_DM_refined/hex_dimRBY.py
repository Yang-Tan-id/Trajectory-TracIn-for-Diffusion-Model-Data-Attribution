#!/usr/bin/env python3
# score_counter_batch_hex.py

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt


# -------------------------
# Config (edit me)
# -------------------------

ALGO = "end"
EXTRA1 = "traj_objective2000"
EXTRA2 = "endpoint_loss2000"
EXTRA3 = "endpoint_dtrak"
EXTRA4 = "endpoint_journey_trak"
EXTRA5 = "endpoint_das_projected"
EXTRA = EXTRA4  # Use endpoint_loss as the extra field
CFG = {
    # Put your runs here: each with a label + json path
    "runs": [
         {"label": "b", "input_json": f"tracein_{ALGO}_runs/model_109900_b_{EXTRA}/baseline/result_topk.json"},
         {"label": "by", "input_json": f"tracein_{ALGO}_runs/model_109900_by_{EXTRA}/baseline/result_topk.json"},
        {"label": "r", "input_json": f"tracein_{ALGO}_runs/model_109900_r_{EXTRA}/baseline/result_topk.json"},
        {"label": "rb", "input_json": f"tracein_{ALGO}_runs/model_109900_rb_{EXTRA}/baseline/result_topk.json"},
        {"label": "y", "input_json": f"tracein_{ALGO}_runs/model_109900_y_{EXTRA}/baseline/result_topk.json"},
        {"label": "yr", "input_json": f"tracein_{ALGO}_runs/model_109900_yr_{EXTRA}/baseline/result_topk.json"},
    ],

    "output_dir": f"RBY_reports_new/batch_hex/{ALGO}_{EXTRA}0.02",
    "summary_txt": "summary_triplets.txt",
    "summary_json": "summary_triplets.json",

    "list_key": "top",
    "src_field": "src",
    "score_field": "score",

    "src_values": [1, 2, 3, 4, 5, 6],
    "src_groups": {
        # original grouping stays the same
        "src 1+2": [1, 2],
        "src 3+4": [3, 4],
        "src 5+6": [5, 6],
    },

    # NEW: axis display names (12,34,56 -> blue, red, yellow)
    "axis_names": ("blue", "red", "yellow"),

    # Hex chart
    "hex_radius": 0.02,
    "tri_fill_alpha": 0.18,
    "tri_edge_alpha": 0.80,
    "point_size": 35,
}


# -------------------------
# Data structures
# -------------------------
@dataclass
class ScoreSums:
    pos_sum: float = 0.0
    neg_sum: float = 0.0
    zero_count: int = 0

    def add(self, score: float) -> None:
        if score > 0:
            self.pos_sum += score
        elif score < 0:
            self.neg_sum += score
        else:
            self.zero_count += 1

    @property
    def abs_neg_sum(self) -> float:
        return abs(self.neg_sum)


# -------------------------
# Helpers
# -------------------------
def safe_div(a: float, b: float) -> float:
    return 0.0 if b == 0 else a / b


def fmt4(x: float) -> str:
    return f"{x:.4f}"


def load_entries(path: Path, list_key: str) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or list_key not in data or not isinstance(data[list_key], list):
        raise ValueError(f"{path}: expected dict with list key '{list_key}'")
    return data[list_key]


def iter_src_scores(
    entries: Iterable[Dict[str, Any]],
    src_field: str,
    score_field: str,
    allowed_src: List[int] | None,
) -> Iterable[Tuple[int, float]]:
    allowed = set(allowed_src) if allowed_src is not None else None
    for e in entries:
        if not isinstance(e, dict):
            continue
        if src_field not in e or score_field not in e:
            continue
        try:
            src = int(e[src_field])
            score = float(e[score_field])
        except (TypeError, ValueError):
            continue
        if allowed is not None and src not in allowed:
            continue
        yield src, score


def compute_by_src_sums(src_scores: Iterable[Tuple[int, float]]) -> Dict[int, ScoreSums]:
    by_src_sums: Dict[int, ScoreSums] = defaultdict(ScoreSums)
    for src, score in src_scores:
        by_src_sums[src].add(score)
    return dict(by_src_sums)


def compute_group_triplets(
    by_src_sums: Dict[int, ScoreSums],
    src_values: List[int],
    src_groups: Dict[str, List[int]],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Return:
      pos_triplet = (pos_share12, pos_share34, pos_share56)
      neg_triplet = (abs_neg_share12, abs_neg_share34, abs_neg_share56)
    Shares are w.r.t. total pos_sum and total abs_neg_sum across ALL src_values.
    """
    total_pos = sum(by_src_sums.get(s, ScoreSums()).pos_sum for s in src_values)
    total_abs_neg = sum(by_src_sums.get(s, ScoreSums()).abs_neg_sum for s in src_values)

    group_names = list(src_groups.keys())
    pos_shares: List[float] = []
    neg_shares: List[float] = []

    for g in group_names:
        srcs = src_groups[g]
        gpos = sum(by_src_sums.get(s, ScoreSums()).pos_sum for s in srcs)
        gneg = sum(by_src_sums.get(s, ScoreSums()).abs_neg_sum for s in srcs)
        pos_shares.append(safe_div(gpos, total_pos))
        neg_shares.append(safe_div(gneg, total_abs_neg))

    return (pos_shares[0], pos_shares[1], pos_shares[2]), (neg_shares[0], neg_shares[1], neg_shares[2])


def mean_triplet(triplets: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    if not triplets:
        return (0.0, 0.0, 0.0)
    a = sum(t[0] for t in triplets) / len(triplets)
    b = sum(t[1] for t in triplets) / len(triplets)
    c = sum(t[2] for t in triplets) / len(triplets)
    return (a, b, c)


# -------------------------
# Plotting: hex projection triangle (axis-projection vertices)
# -------------------------
def plot_hex_projection_triangles(
    label: str,
    pos_triplet: Tuple[float, float, float],
    neg_triplet: Tuple[float, float, float],
    mu_pos: Tuple[float, float, float],
    mu_neg: Tuple[float, float, float],
    labels_3: Tuple[str, str, str],
    out_png: Path,
    radius: float,
    fill_alpha: float,
    edge_alpha: float,
    point_size: int,
) -> None:
    import numpy as np

    dirs = np.array(
        [
            [1.0, 0.0],
            [math.cos(2 * math.pi / 3), math.sin(2 * math.pi / 3)],
            [math.cos(4 * math.pi / 3), math.sin(4 * math.pi / 3)],
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(7.2, 7.2))

    # hex boundary
    angles = np.deg2rad([0, 60, 120, 180, 240, 300, 360])
    hx = radius * np.cos(angles)
    hy = radius * np.sin(angles)
    ax.plot(hx, hy, linewidth=2)

    # axes + labels + ticks
    for i in range(3):
        ax.plot([0, radius * dirs[i, 0]], [0, radius * dirs[i, 1]], linewidth=1.5)
        ax.plot([0, -radius * dirs[i, 0]], [0, -radius * dirs[i, 1]], linewidth=0.8)
        ax.text(
            1.14 * radius * dirs[i, 0],
            1.14 * radius * dirs[i, 1],
            labels_3[i],
            ha="center",
            va="center",
            fontsize=11,
        )
        for t in [-radius, 0.0, radius]:
            px, py = t * dirs[i]
            ax.plot(px, py, marker="o", markersize=3)
            ax.text(px * 1.06, py * 1.06, f"{t:+.2f}", fontsize=9, ha="center", va="center")

    def draw_one(triplet: Tuple[float, float, float], mu: Tuple[float, float, float], name: str) -> None:
        v = np.array(triplet, dtype=float)
        m = np.array(mu, dtype=float)
        off = np.clip(v - m, -radius, radius)

        tri = np.vstack([off[0] * dirs[0], off[1] * dirs[1], off[2] * dirs[2]])  # (3,2)
        xs = tri[:, 0].tolist()
        ys = tri[:, 1].tolist()

        ax.plot(xs + [xs[0]], ys + [ys[0]], linewidth=2, alpha=edge_alpha)
        ax.fill(xs, ys, alpha=fill_alpha)

        ax.scatter(tri[:, 0], tri[:, 1], s=point_size)
        cx, cy = tri.mean(axis=0)
        ax.text(cx, cy, name, fontsize=10, ha="center", va="center")

    draw_one(pos_triplet, mu_pos, "pos")
    draw_one(neg_triplet, mu_neg, "abs_neg")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.25 * radius, 1.25 * radius)
    ax.set_ylim(-1.25 * radius, 1.25 * radius)

    ax.set_title(
        f"{label}\n"
        f"mu_pos=({fmt4(mu_pos[0])},{fmt4(mu_pos[1])},{fmt4(mu_pos[2])})  "
        f"mu_neg=({fmt4(mu_neg[0])},{fmt4(mu_neg[1])},{fmt4(mu_neg[2])})  clip=±{radius}"
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.25)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


# -------------------------
# Main
# -------------------------
def main() -> None:
    runs = CFG["runs"]
    if not runs:
        raise ValueError("CFG['runs'] is empty. Add your labeled json files there.")

    out_dir = Path(CFG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Read all runs and compute triplets
    records: List[Dict[str, Any]] = []
    pos_triplets: List[Tuple[float, float, float]] = []
    neg_triplets: List[Tuple[float, float, float]] = []

    for r in runs:
        label = str(r["label"])
        in_path = Path(r["input_json"])
        if not in_path.exists():
            raise FileNotFoundError(f"[{label}] not found: {in_path}")

        entries = load_entries(in_path, CFG["list_key"])
        src_scores = iter_src_scores(
            entries,
            src_field=CFG["src_field"],
            score_field=CFG["score_field"],
            allowed_src=CFG.get("src_values"),
        )
        by_src_sums = compute_by_src_sums(src_scores)
        pos_t, neg_t = compute_group_triplets(
            by_src_sums=by_src_sums,
            src_values=CFG["src_values"],
            src_groups=CFG["src_groups"],
        )

        records.append(
            {
                "label": label,
                "input_json": str(in_path),
                "pos_triplet": pos_t,
                "neg_triplet": neg_t,
            }
        )
        pos_triplets.append(pos_t)
        neg_triplets.append(neg_t)

    # 2) Global centers (zero points): per-dim mean across n files
    mu_pos = mean_triplet(pos_triplets)
    mu_neg = mean_triplet(neg_triplets)

    # 3) Plot each run into the folder
    axis_names = tuple(CFG.get("axis_names", ("blue", "red", "yellow")))
    for rec in records:
        label = rec["label"]
        pos_t = rec["pos_triplet"]
        neg_t = rec["neg_triplet"]
        out_png = out_dir / f"hex_{label}.png"

        plot_hex_projection_triangles(
            label=label,
            pos_triplet=pos_t,
            neg_triplet=neg_t,
            mu_pos=mu_pos,
            mu_neg=mu_neg,
            labels_3=(axis_names[0], axis_names[1], axis_names[2]),
            out_png=out_png,
            radius=float(CFG["hex_radius"]),
            fill_alpha=float(CFG["tri_fill_alpha"]),
            edge_alpha=float(CFG["tri_edge_alpha"]),
            point_size=int(CFG["point_size"]),
        )

    # 4) Save summary files
    summary_txt = out_dir / CFG["summary_txt"]
    summary_json = out_dir / CFG["summary_json"]

    lines: List[str] = []
    lines.append("BATCH SUMMARY (triplets are group shares over score-mass)")
    lines.append(f"n_runs = {len(records)}")
    lines.append("")
    lines.append("AXES (1+2, 3+4, 5+6) renamed to:")
    lines.append(f"  ({axis_names[0]}, {axis_names[1]}, {axis_names[2]})")
    lines.append("")
    lines.append("GLOBAL ZERO (center) — per-dim mean across runs:")
    lines.append(f"  mu_pos (blue,red,yellow) = {fmt4(mu_pos[0])}, {fmt4(mu_pos[1])}, {fmt4(mu_pos[2])}")
    lines.append(f"  mu_neg (blue,red,yellow) = {fmt4(mu_neg[0])}, {fmt4(mu_neg[1])}, {fmt4(mu_neg[2])}")
    lines.append("")
    lines.append("PER RUN:")
    for rec in records:
        p = rec["pos_triplet"]
        n = rec["neg_triplet"]
        lines.append(
            f"- {rec['label']}\n"
            f"    pos = {fmt4(p[0])}, {fmt4(p[1])}, {fmt4(p[2])}\n"
            f"    neg = {fmt4(n[0])}, {fmt4(n[1])}, {fmt4(n[2])}\n"
            f"    file = {rec['input_json']}"
        )

    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary_json.write_text(
        json.dumps(
            {
                "n_runs": len(records),
                "axis_names": axis_names,
                "mu_pos": mu_pos,
                "mu_neg": mu_neg,
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[OK] Output dir: {out_dir}")
    print(f"[OK] Wrote: {summary_txt}")
    print(f"[OK] Wrote: {summary_json}")
    print(f"[OK] Saved {len(records)} hex images: hex_<label>.png")


if __name__ == "__main__":
    main()
