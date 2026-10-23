#!/usr/bin/env python3
"""Bundle Shape3D Q0-Q99 endpoint attribution scores and cached LDS results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


SHAPES_ROOT = Path(__file__).resolve().parents[1]
if str(SHAPES_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT))

from dataset_config import _prompt_tag


VARIANTS = ("query_train_l2", "query_l2", "train_l2", "raw")


def parse_ids(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value]


def split_variant(name: str) -> tuple[str, str]:
    for variant in VARIANTS:
        suffix = f"_{variant}"
        if name.endswith(suffix):
            return name[: -len(suffix)], variant
    return name, "unspecified"


def describe_namespace(name: str) -> dict[str, str] | None:
    base, variant = split_variant(name)
    plain = base.removeprefix("traj_tracin_")
    if "q0_99" not in plain:
        return None

    if "polluted_endpoint" in plain:
        family = "endpoint_pollute"
    elif plain.startswith("endpoint_tracin_"):
        family = "endpoint_tracin"
    elif "retrac" in plain:
        family = "retrac"
    else:
        return None

    if "adamw_full" in plain:
        transform = "adamw_full"
    elif "adamw_residual" in plain:
        transform = "adamw_residual"
    elif "adamw" in plain:
        transform = "adamw"
    else:
        transform = "raw_gradient"

    if family == "endpoint_pollute":
        timestamp_count = "100" if "aligned100x1" in plain else "10"
    elif "10x" in plain or "endpoint10x1" in plain:
        timestamp_count = "10"
    elif "100x" in plain or "endpoint100x1" in plain:
        timestamp_count = "100"
    else:
        timestamp_count = "unspecified"

    return {
        "family": family,
        "transform": transform,
        "timestamp_count": timestamp_count,
        "namespace": base,
        "variant": variant,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_float(payload: dict[str, Any], key: str) -> float:
    try:
        return float(payload[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.fmean(finite) if finite else float("nan")


def population_std(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.pstdev(finite) if len(finite) > 1 else 0.0 if finite else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument(
        "--query-file",
        type=Path,
        default=SHAPES_ROOT / "queries_in_distribution_plus_zero_seed_100_219.json",
    )
    parser.add_argument("--query-ids", default=",".join(map(str, range(100))))
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive-name", default="shape3d_q0_99_endpoint_scores_lds")
    parser.add_argument("--no-archive", action="store_true")
    args = parser.parse_args()

    result_root = args.result_root or (SHAPES_ROOT / "result" / args.experiment)
    score_root = (
        result_root
        / "attribution_score"
        / "prompted_solo"
        / f"train_seed_{args.train_seed}"
    )
    eval_root = result_root / "eval" / "prompted_solo"
    query_ids = parse_ids(args.query_ids)
    print(f"[startup] result_root={result_root}", flush=True)
    print(f"[startup] output_dir={args.output_dir}", flush=True)
    print(f"[startup] queries={len(query_ids)}", flush=True)
    records = json.loads(args.query_file.read_text())["queries"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = args.output_dir / "metadata"
    if metadata.exists():
        shutil.rmtree(metadata)
    metadata.mkdir(parents=True)

    selected: set[Path] = set()
    score_rows: list[dict[str, Any]] = []
    lds_rows: list[dict[str, Any]] = []
    missing_query_roots: list[str] = []
    invalid_lds_summaries: list[str] = []

    for query_position, query_id in enumerate(query_ids, start=1):
        print(
            f"[scan] query={query_id} ({query_position}/{len(query_ids)})",
            flush=True,
        )
        record = records[query_id]
        prompt = str(record["prompt"])
        initial_seed = int(record.get("initial_seed", record.get("seed")))
        query_rel = Path(f"query_{_prompt_tag(prompt)}") / f"initial_seed_{initial_seed}"

        one_score_root = score_root / query_rel
        if not one_score_root.is_dir():
            missing_query_roots.append(str(one_score_root))
        else:
            for namespace_dir in sorted(one_score_root.iterdir()):
                if not namespace_dir.is_dir():
                    continue
                description = describe_namespace(namespace_dir.name)
                if description is None:
                    continue
                selected.add(namespace_dir)
                score_rows.append(
                    {
                        "query_id": query_id,
                        "prompt": prompt,
                        "initial_seed": initial_seed,
                        **description,
                        "directory_name": namespace_dir.name,
                        # Avoid a redundant recursive walk over every score bank.
                        # The archive pass below already traverses selected data once.
                        "file_count": "not_prescanned",
                        "bytes": "not_prescanned",
                        "relative_path": str(namespace_dir.relative_to(result_root)),
                    }
                )

        one_lds_root = eval_root / query_rel / "lds"
        if not one_lds_root.is_dir():
            missing_query_roots.append(str(one_lds_root))
            continue
        for namespace_dir in sorted(one_lds_root.iterdir()):
            if not namespace_dir.is_dir():
                continue
            description = describe_namespace(namespace_dir.name)
            if description is None:
                continue
            selected.add(namespace_dir)
            summaries = sorted(namespace_dir.glob("*/pred_kept_sign_*/*/lds_summary.json"))
            for summary in summaries:
                relative = summary.relative_to(namespace_dir)
                target_function = relative.parts[0]
                prediction_sign = relative.parts[1].removeprefix("pred_kept_sign_")
                subset_group = relative.parts[2]
                try:
                    payload = json.loads(summary.read_text())
                    if not isinstance(payload, dict):
                        raise TypeError(
                            f"expected a JSON object, got {type(payload).__name__}"
                        )
                except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as error:
                    relative_path = str(summary.relative_to(result_root))
                    invalid_lds_summaries.append(
                        f"{relative_path}\t{type(error).__name__}: {error}"
                    )
                    print(
                        f"[warning] skipping invalid LDS summary: {relative_path} "
                        f"({type(error).__name__}: {error})",
                        flush=True,
                    )
                    continue
                lds_rows.append(
                    {
                        "query_id": query_id,
                        "prompt": prompt,
                        "initial_seed": initial_seed,
                        **description,
                        "directory_name": namespace_dir.name,
                        "target_function": target_function,
                        "prediction_sign": prediction_sign,
                        "subset_group": subset_group,
                        "lds_spearman": safe_float(payload, "lds_spearman"),
                        "lds_percent": safe_float(payload, "lds_percent"),
                        "relative_path": str(summary.relative_to(result_root)),
                    }
                )

    score_fields = [
        "query_id", "prompt", "initial_seed", "family", "transform",
        "timestamp_count", "namespace", "variant", "directory_name",
        "file_count", "bytes", "relative_path",
    ]
    lds_fields = [
        "query_id", "prompt", "initial_seed", "family", "transform",
        "timestamp_count", "namespace", "variant", "directory_name",
        "target_function", "prediction_sign", "subset_group", "lds_spearman",
        "lds_percent", "relative_path",
    ]
    write_csv(metadata / "score_inventory.csv", score_rows, score_fields)
    write_csv(metadata / "lds_all_results.csv", lds_rows, lds_fields)

    group_fields = [
        "family", "transform", "timestamp_count", "namespace", "variant",
        "target_function", "prediction_sign", "subset_group",
    ]
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in lds_rows:
        key = tuple(str(row[field]) for field in group_fields)
        grouped[key].append(float(row["lds_percent"]))
    group_rows: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        finite = [value for value in values if math.isfinite(value)]
        group_rows.append(
            {
                **dict(zip(group_fields, key)),
                "queries": len(finite),
                "mean_lds_percent": mean(values),
                "std_lds_percent": population_std(values),
                "min_lds_percent": min(finite) if finite else float("nan"),
                "max_lds_percent": max(finite) if finite else float("nan"),
            }
        )
    summary_fields = group_fields + [
        "queries", "mean_lds_percent", "std_lds_percent",
        "min_lds_percent", "max_lds_percent",
    ]
    write_csv(metadata / "lds_group_summary.csv", group_rows, summary_fields)

    report = [
        "SHAPE3D Q0-Q99 ENDPOINT SCORE / LDS EXPORT",
        f"created: {datetime.now().isoformat(timespec='seconds')}",
        f"result_root: {result_root}",
        f"queries requested: {len(query_ids)}",
        f"selected score namespace directories: {len(score_rows)}",
        f"selected LDS namespace directories: {sum(1 for p in selected if '/lds/' in str(p))}",
        f"LDS result rows: {len(lds_rows)}",
        f"invalid LDS summaries skipped: {len(invalid_lds_summaries)}",
        f"missing query roots: {len(missing_query_roots)}",
        "",
        "FAMILY                  TRANSFORM       T       VARIANT            TARGET                         SIGN    N      MEAN       STD",
        "-" * 125,
    ]
    for row in group_rows:
        report.append(
            f"{str(row['family']):23s} {str(row['transform']):15s} "
            f"{str(row['timestamp_count']):>3s} "
            f"{str(row['variant']):18s} {str(row['target_function']):30s} "
            f"{str(row['prediction_sign']):>4s} {int(row['queries']):4d} "
            f"{float(row['mean_lds_percent']):+9.3f}% "
            f"{float(row['std_lds_percent']):8.3f}%  {row['namespace']}"
        )
    (metadata / "lds_group_summary.txt").write_text("\n".join(report) + "\n")
    (metadata / "missing_query_roots.txt").write_text(
        "\n".join(missing_query_roots) + ("\n" if missing_query_roots else "")
    )
    (metadata / "invalid_lds_summaries.txt").write_text(
        "\n".join(invalid_lds_summaries)
        + ("\n" if invalid_lds_summaries else "")
    )

    selected_rel = sorted(str(path.relative_to(result_root)) for path in selected)
    (metadata / "selected_result_directories.txt").write_text(
        "\n".join(selected_rel) + ("\n" if selected_rel else "")
    )
    readme = (
        "This bundle contains existing score banks and cached LDS outputs only.\n"
        "It does not contain model checkpoints, train/query gradient artifacts, samples,\n"
        "datasets, or top-K retrained models. Paths under data/ are relative to the\n"
        "Shape3D experiment result root. See metadata/*.csv and *.txt for inventories.\n"
    )
    (metadata / "README.txt").write_text(readme)

    print("\n".join(report[:8]), flush=True)
    print(f"[metadata] {metadata}", flush=True)
    if args.no_archive:
        return
    if not selected_rel:
        raise RuntimeError("No matching score/LDS directories were found")

    staging = args.output_dir / "staging"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "data").mkdir(parents=True)
    shutil.copytree(metadata, staging / "metadata")
    for relative in selected_rel:
        source = result_root / relative
        destination = staging / "data" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source, target_is_directory=True)

    zstd = shutil.which("zstd")
    if zstd:
        archive = args.output_dir / f"{args.archive_name}.tar.zst"
        command = [
            "tar", "--dereference", "-C", str(staging),
            "-I", f"{zstd} -T0 -3", "-cf", str(archive), ".",
        ]
    else:
        archive = args.output_dir / f"{args.archive_name}.tar.gz"
        command = [
            "tar", "--dereference", "-C", str(staging),
            "-I", "gzip -1", "-cf", str(archive), ".",
        ]
    if archive.exists():
        archive.unlink()
    print(f"[archive] creating {archive}", flush=True)
    subprocess.run(command, check=True)
    shutil.rmtree(staging)
    print(f"[saved] {archive}", flush=True)
    print(f"[bytes] {archive.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
