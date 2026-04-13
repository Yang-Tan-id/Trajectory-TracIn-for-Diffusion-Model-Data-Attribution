#!/usr/bin/env python3
# score_counter_singlefile.py

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt


# -------------------------
# Config dictionary (edit me)
# -------------------------
MODEL_NAME = "y"
ALGO = "end"
EXTRA1 = "traj_objective"
EXTRA2 = "endpoint_loss"
CFG = {
    "input_json": f"tracein_{ALGO}_runs/model_109900_{MODEL_NAME}_{EXTRA2}2000/baseline/result_topk.json",

    "output_dir": f"RBY_reports_new/{ALGO}/{MODEL_NAME}",
    "report_txt": "score_report.txt",
    "chart_png": "score_distribution.png",

    "list_key": "top",
    "src_field": "src",
    "score_field": "score",

    # histogram bin width ("gap of 10")
    "bin_width": 10,

    "src_values": [1, 2, 3, 4, 5, 6],
    "src_groups": {
        "src 1+2": [1, 2],
        "src 3+4": [3, 4],
        "src 5+6": [5, 6],
    },
}


@dataclass
class ScoreSums:
    pos_sum: float = 0.0     # sum(score) over score > 0
    neg_sum: float = 0.0     # sum(score) over score < 0 (negative)
    zero_count: int = 0      # how many exactly zero

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


def fmt_ratio(x: float) -> str:
    return f"{x:.4f}"


def safe_div(a: float, b: float) -> float:
    return 0.0 if b == 0 else a / b


def load_entries(path: Path, list_key: str) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON dict like {{'{list_key}': [...]}}")
    if list_key not in data or not isinstance(data[list_key], list):
        raise ValueError(f"{path}: missing list key '{list_key}' or it is not a list")

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


def compute_sums_and_scores(
    src_scores: Iterable[Tuple[int, float]]
) -> Tuple[Dict[int, ScoreSums], Dict[int, List[float]]]:
    by_src_sums: Dict[int, ScoreSums] = defaultdict(ScoreSums)
    by_src_scores: Dict[int, List[float]] = defaultdict(list)

    for src, score in src_scores:
        by_src_sums[src].add(score)
        by_src_scores[src].append(score)

    return dict(by_src_sums), dict(by_src_scores)


def group_scores(
    by_src_scores: Dict[int, List[float]],
    src_groups: Dict[str, List[int]],
) -> Dict[str, List[float]]:
    grouped: Dict[str, List[float]] = {}
    for name, srcs in src_groups.items():
        merged: List[float] = []
        for s in srcs:
            merged.extend(by_src_scores.get(s, []))
        grouped[name] = merged
    return grouped


def build_report_text(
    by_src_sums: Dict[int, ScoreSums],
    src_order: List[int],
    src_groups: Dict[str, List[int]],
) -> str:
    # totals across all srcs (for score-mass shares)
    total_pos_sum = sum(by_src_sums.get(s, ScoreSums()).pos_sum for s in src_order)
    total_abs_neg_sum = sum(by_src_sums.get(s, ScoreSums()).abs_neg_sum for s in src_order)

    lines: List[str] = []

    lines.append("PER-SRC SCORE SUMS + SHARES (score-based, not counts)")
    lines.append("src | pos_sum | neg_sum | abs_neg_sum | pos_share(all_pos_sum) | neg_share(all_abs_neg_sum) | zero_count")
    lines.append("-" * 118)

    for s in src_order:
        ss = by_src_sums.get(s, ScoreSums())
        pos_share = safe_div(ss.pos_sum, total_pos_sum)
        neg_share = safe_div(ss.abs_neg_sum, total_abs_neg_sum)
        lines.append(
            f"{s:>3} | "
            f"{ss.pos_sum:>8.3f} | {ss.neg_sum:>8.3f} | {ss.abs_neg_sum:>11.3f} | "
            f"{fmt_ratio(pos_share):>19} | {fmt_ratio(neg_share):>23} | "
            f"{ss.zero_count:>10}"
        )

    lines.append("")
    lines.append("TOTALS")
    lines.append("------")
    total_zero = sum(by_src_sums.get(s, ScoreSums()).zero_count for s in src_order)
    lines.append(f"sum_pos_sum={total_pos_sum:.6f}")
    lines.append(f"sum_neg_sum={sum(by_src_sums.get(s, ScoreSums()).neg_sum for s in src_order):.6f}  (negative)")
    lines.append(f"sum_abs_neg_sum={total_abs_neg_sum:.6f}")
    lines.append(f"sum_zero_count={total_zero}")

    # grouped shares for pos_sum and abs_neg_sum: (1+2):(3+4):(5+6)
    lines.append("")
    lines.append("GROUPED SHARES (1+2 : 3+4 : 5+6) — SCORE SUMS")
    lines.append("------------------------------------------------")

    ordered_group_names = list(src_groups.keys())

    group_pos_sum: Dict[str, float] = {}
    group_abs_neg_sum: Dict[str, float] = {}

    for gname, srcs in src_groups.items():
        group_pos_sum[gname] = sum(by_src_sums.get(s, ScoreSums()).pos_sum for s in srcs)
        group_abs_neg_sum[gname] = sum(by_src_sums.get(s, ScoreSums()).abs_neg_sum for s in srcs)

    pos_group_shares = {g: safe_div(v, total_pos_sum) for g, v in group_pos_sum.items()}
    neg_group_shares = {g: safe_div(v, total_abs_neg_sum) for g, v in group_abs_neg_sum.items()}

    lines.append("Positive score-mass share of ALL positive scores:")
    for g in ordered_group_names:
        lines.append(f"  {g}: pos_sum={group_pos_sum[g]:.6f}  (share={fmt_ratio(pos_group_shares[g])})")

    lines.append("Negative score-mass share of ALL negative scores (using abs values):")
    for g in ordered_group_names:
        lines.append(f"  {g}: abs_neg_sum={group_abs_neg_sum[g]:.6f}  (share={fmt_ratio(neg_group_shares[g])})")

    lines.append("")
    lines.append("Triplet ratios (as decimals, sum to 1.0 when totals > 0):")
    lines.append(
        "  pos_sum (1+2, 3+4, 5+6) = "
        + ", ".join(fmt_ratio(pos_group_shares[g]) for g in ordered_group_names)
    )
    lines.append(
        "  abs_neg_sum (1+2, 3+4, 5+6) = "
        + ", ".join(fmt_ratio(neg_group_shares[g]) for g in ordered_group_names)
    )

    return "\n".join(lines) + "\n"


def save_grouped_distribution_chart(
    grouped_scores: Dict[str, List[float]],
    out_png: Path,
    bin_width: float,
) -> None:
    all_scores = [x for xs in grouped_scores.values() for x in xs]
    if not all_scores:
        raise ValueError("No valid scores found to plot.")

    mn, mx = min(all_scores), max(all_scores)
    left = math.floor(mn / bin_width) * bin_width
    right = math.ceil(mx / bin_width) * bin_width
    if right == left:
        right = left + bin_width

    n_bins = int(round((right - left) / bin_width))
    edges = [left + i * bin_width for i in range(n_bins + 1)]

    plt.figure()
    for name in grouped_scores.keys():
        scores = grouped_scores[name]
        if not scores:
            continue
        plt.hist(scores, bins=edges, histtype="step", density=True, label=name)

    plt.title(f"Score distribution (grouped) | bin width = {bin_width}")
    plt.xlabel("score")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main() -> None:
    in_path = Path(CFG["input_json"])
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    out_dir = Path(CFG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / CFG["report_txt"]
    chart_path = out_dir / CFG["chart_png"]

    entries = load_entries(in_path, CFG["list_key"])
    src_scores = iter_src_scores(
        entries,
        src_field=CFG["src_field"],
        score_field=CFG["score_field"],
        allowed_src=CFG.get("src_values"),
    )

    by_src_sums, by_src_scores = compute_sums_and_scores(src_scores)

    report_text = build_report_text(
        by_src_sums=by_src_sums,
        src_order=CFG["src_values"],
        src_groups=CFG["src_groups"],
    )
    report_path.write_text(report_text, encoding="utf-8")

    grouped_scores = group_scores(by_src_scores, CFG["src_groups"])
    save_grouped_distribution_chart(
        grouped_scores=grouped_scores,
        out_png=chart_path,
        bin_width=float(CFG["bin_width"]),
    )

    print(f"Wrote report: {report_path}")
    print(f"Saved chart:  {chart_path}")


if __name__ == "__main__":
    main()
