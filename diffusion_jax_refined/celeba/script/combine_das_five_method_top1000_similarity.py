#!/usr/bin/env python3
"""Combine cached CelebA Top-1000 similarity results into one report."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = (
    ("clip_cosine", "CLIP COSINE", +1),
    ("dino_cosine", "DINO COSINE", +1),
    ("pixel_rmse", "PIXEL RMSE", -1),
    ("lpips", "LPIPS", -1),
)

DAS_LABEL = "DAS EMA / lambda=200"
PAPER_CONFIGURATIONS = (
    ("DAS", DAS_LABEL),
    ("PINT", "PINT ADAMW / BOTH-L2"),
    ("TracInCP", "ENDPOINT TRACIN ADAMW / RAW"),
    ("GAS", "ENDPOINT TRACIN ADAMW / BOTH-L2"),
    ("ReTrac", "RETRAC ADAMW / BOTH-L2"),
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    required = {"query_id", "configuration", *(name for name, _, _ in METRICS)}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    rows: list[dict[str, object]], labels: list[tuple[str, str]], query_ids: list[int]
) -> list[str]:
    lookup = {
        (int(row["query_id"]), str(row["configuration"])): row for row in rows
    }
    lines: list[str] = []
    for metric, display, _direction in METRICS:
        lines.extend(
            [
                "",
                display,
                f"{'METHOD':32s} {'MEAN':>11s} {'MEDIAN':>11s} {'STD':>11s}",
                "-" * 69,
            ]
        )
        for display_label, source_label in labels:
            values = [
                float(lookup[(query_id, source_label)][metric])
                for query_id in query_ids
            ]
            lines.append(
                f"{display_label:32s} {statistics.fmean(values):11.6f} "
                f"{statistics.median(values):11.6f} "
                f"{statistics.pstdev(values):11.6f}"
            )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--das-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    baseline_rows = [
        row
        for row in read_rows(args.baseline_csv)
        if row.get("selection", "top") == "top"
    ]
    das_source = read_rows(args.das_csv)
    das_rows = [row for row in das_source if row["configuration"] == DAS_LABEL]
    if not das_rows:
        raise ValueError(f"{args.das_csv}: no {DAS_LABEL!r} rows")

    baseline_queries = {int(row["query_id"]) for row in baseline_rows}
    das_queries = {int(row["query_id"]) for row in das_rows}
    query_ids = sorted(baseline_queries & das_queries)
    if not query_ids:
        raise ValueError("the two CSV files have no common query IDs")

    baseline_labels = list(dict.fromkeys(row["configuration"] for row in baseline_rows))
    counts: dict[tuple[int, str], int] = defaultdict(int)
    combined: list[dict[str, object]] = []
    for row in [*das_rows, *baseline_rows]:
        query_id = int(row["query_id"])
        if query_id not in query_ids:
            continue
        label = row["configuration"]
        key = (query_id, label)
        counts[key] += 1
        combined.append(
            {
                "query_id": query_id,
                "configuration": label,
                "topk": int(row.get("topk", row.get("k", 1000))),
                **{name: float(row[name]) for name, _, _ in METRICS},
            }
        )
    duplicates = [key for key, count in counts.items() if count != 1]
    if duplicates:
        raise ValueError(f"duplicate/missing combined records: {duplicates[:5]}")

    all_labels = [(DAS_LABEL, DAS_LABEL), *[(label, label) for label in baseline_labels]]
    missing_paper = [source for _, source in PAPER_CONFIGURATIONS if source not in dict(all_labels)]
    if missing_paper:
        raise ValueError(f"missing paper configurations: {missing_paper}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined.sort(key=lambda row: (int(row["query_id"]), str(row["configuration"])))
    write_csv(args.output_dir / "per_query.csv", combined)

    lines = [
        "CELEBA TOP-1000 TRAINING-IMAGE PERCEPTUAL SIMILARITY — DAS AND BASELINES",
        "=" * 82,
        f"queries={len(query_ids)} range={query_ids[0]}..{query_ids[-1]}; topk=1000",
        "No removal and no retraining. Reference is the same-query full-data endpoint.",
        "CLIP/DINO: higher is more similar; Pixel-RMSE/LPIPS: lower is more similar.",
        "",
        "PAPER CONFIGURATIONS",
    ]
    lines.extend(summarize(combined, list(PAPER_CONFIGURATIONS), query_ids))
    lines.extend(["", "ALL RAW/ADAMW CONFIGURATIONS"])
    lines.extend(summarize(combined, all_labels, query_ids))
    report = "\n".join(lines) + "\n"
    (args.output_dir / "report.txt").write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"SAVED: {args.output_dir / 'report.txt'}")
    print(f"SAVED: {args.output_dir / 'per_query.csv'}")


if __name__ == "__main__":
    main()
