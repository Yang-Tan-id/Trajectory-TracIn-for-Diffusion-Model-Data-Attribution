#!/usr/bin/env python3
"""Bundle CelebA and ArtBench extreme-K reports with per-query values.

The perceptual analyses already write ``report.txt``, ``summary.csv`` and
``per_query.csv``.  This exporter does no model inference: it validates those
artifacts and creates one self-contained text file for each dataset/K pair.
"""

from __future__ import annotations

import argparse
import csv
import io
from collections import Counter
from pathlib import Path


DATASETS = ("celeba", "artbench")
KS = (1000, 500, 100)
METRICS = ("clip_cosine", "dino_cosine", "pixel_rmse", "lpips")


def find_result(root: Path, k: int) -> Path:
    pattern = f"five_method_raw_vs_adamw_top_bottom{k}_endpoint_perceptual_*"
    candidates = sorted(
        path
        for path in root.rglob(pattern)
        if path.is_dir()
        and (path / "report.txt").is_file()
        and (path / "summary.csv").is_file()
        and (path / "per_query.csv").is_file()
    )
    exact = [path for path in candidates if path.name.endswith("q0_q59")]
    if len(exact) == 1:
        return exact[0]
    if len(candidates) == 1:
        return candidates[0]
    raise RuntimeError(
        f"expected one complete K={k} result under {root}; "
        f"found {len(candidates)}: {candidates}"
    )


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        return list(reader.fieldnames), list(reader)


def query_key(row: dict[str, str]) -> int:
    value = row.get("query_id", row.get("query_index"))
    if value is None:
        raise ValueError("per-query CSV has neither query_id nor query_index")
    return int(value)


def display_configuration(dataset: str, row: dict[str, str]) -> tuple[str, str]:
    if dataset == "artbench":
        return row["method"], row["transform"]
    configuration = row["configuration"]
    if " RAW-GRADIENT " in f" {configuration} ":
        transform = "raw_gradient"
    elif " ADAMW " in f" {configuration} ":
        transform = "adamw_full"
    else:
        transform = "unknown"
    return configuration, transform


def validate_per_query(
    dataset: str, k: int, rows: list[dict[str, str]], expected_queries: int
) -> list[int]:
    required = {"selection", "k", *METRICS}
    required |= {"method", "transform"} if dataset == "artbench" else {"configuration"}
    missing = sorted(required - set(rows[0] if rows else ()))
    if missing:
        raise ValueError(f"{dataset} K={k}: missing columns {missing}")
    if any(int(row["k"]) != k for row in rows):
        raise ValueError(f"{dataset} K={k}: per_query.csv contains a different K")

    query_ids = sorted({query_key(row) for row in rows})
    if len(query_ids) != expected_queries:
        raise ValueError(
            f"{dataset} K={k}: expected {expected_queries} queries, got "
            f"{len(query_ids)} ({query_ids})"
        )
    counts = Counter((row["selection"], query_key(row)) for row in rows)
    expected_configs = 10  # five methods x Raw-gradient/AdamW
    for selection in ("top", "bottom"):
        for query in query_ids:
            if counts[(selection, query)] != expected_configs:
                raise ValueError(
                    f"{dataset} K={k} {selection} Q{query}: expected "
                    f"{expected_configs} configurations, got {counts[(selection, query)]}"
                )
    return query_ids


def raw_csv_text(fieldnames: list[str], rows: list[dict[str, str]]) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def readable_per_query(dataset: str, rows: list[dict[str, str]]) -> str:
    ordered = sorted(
        rows,
        key=lambda row: (
            query_key(row),
            0 if row["selection"] == "top" else 1,
            display_configuration(dataset, row),
        ),
    )
    lines = [
        f"{'QUERY':<7} {'SELECT':<7} {'CONFIGURATION':<48} {'TRANSFORM':<13} "
        f"{'CLIP':>11} {'DINO':>11} {'PIXEL-RMSE':>12} {'LPIPS':>11}",
        "-" * 139,
    ]
    for row in ordered:
        method, transform = display_configuration(dataset, row)
        lines.append(
            f"Q{query_key(row):<6d} {row['selection']:<7} {method:<48.48} "
            f"{transform:<13.13} {float(row['clip_cosine']):11.6f} "
            f"{float(row['dino_cosine']):11.6f} "
            f"{float(row['pixel_rmse']):12.6f} "
            f"{float(row['lpips']):11.6f}"
        )
    return "\n".join(lines) + "\n"


def export_one(
    dataset: str,
    k: int,
    source: Path,
    output: Path,
    expected_queries: int,
) -> None:
    per_fields, per_rows = read_csv(source / "per_query.csv")
    summary_fields, summary_rows = read_csv(source / "summary.csv")
    query_ids = validate_per_query(dataset, k, per_rows, expected_queries)
    report = (source / "report.txt").read_text(encoding="utf-8").rstrip()
    title = f"{dataset.upper()} TOP/BOTTOM-{k} PERCEPTUAL RESULTS"
    sections = [
        title,
        "=" * len(title),
        f"Source: {source}",
        f"Queries: {len(query_ids)} (Q{query_ids[0]}--Q{query_ids[-1]})",
        "Configurations per query/selection: 10 (5 methods x 2 transforms)",
        "Metrics: CLIP cosine, DINO cosine, Pixel RMSE, LPIPS",
        "",
        "AGGREGATE REPORT",
        "=" * 80,
        report,
        "",
        "PER-QUERY VALUES (READABLE TABLE)",
        "=" * 80,
        readable_per_query(dataset, per_rows).rstrip(),
        "",
        "PER-QUERY VALUES (CSV; EXACT MACHINE-READABLE VALUES)",
        "=" * 80,
        raw_csv_text(per_fields, per_rows).rstrip(),
        "",
        "SUMMARY STATISTICS (CSV)",
        "=" * 80,
        raw_csv_text(summary_fields, summary_rows).rstrip(),
        "",
    ]
    output.write_text("\n".join(sections), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--celeba-analysis-root", type=Path)
    parser.add_argument("--artbench-analysis-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-queries", type=int, default=60)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    roots = {
        "celeba": args.celeba_analysis_root
        or repo
        / "diffusion_jax_refined/celeba/result/attr40_experiment1/analysis",
        "artbench": args.artbench_analysis_root
        or repo / "stable_diffusion_artbench_pytorch/outputs/analysis",
    }
    output_dir = args.output_dir or repo / "extreme_k_perceptual_combined_txt"
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        for k in KS:
            source = find_result(roots[dataset], k)
            output = output_dir / f"{dataset}_top_bottom_{k}_report_and_per_query.txt"
            export_one(dataset, k, source, output, args.expected_queries)
            print(f"[saved] {output}")
    print(f"[done] files={len(DATASETS) * len(KS)} output={output_dir}")


if __name__ == "__main__":
    main()
