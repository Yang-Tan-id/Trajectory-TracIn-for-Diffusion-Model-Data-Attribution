"""Compare exact removal Top-10 and full-ranking End-10 with CLIP similarity."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .config import ArtBenchConfig
from .plot_nine_row_topk import (
    font,
    image_tile,
    load_queries,
    load_training_records,
    shortened,
)
from .topk_removal import METHODS, method_scores


METHOD_LABELS = {
    "endpoint_pollute_query_train_l2": "Endpoint-Pollute | BOTH-L2",
    "endpoint_tracin_raw": "Endpoint-TracIn | RAW",
    "endpoint_tracin_query_train_l2": "Endpoint-TracIn | BOTH-L2",
    "retrac_query_train_l2": "ReTrac | BOTH-L2",
}


def parse_ints(value: str) -> list[int]:
    result = [int(item) for item in value.replace(",", " ").split() if item]
    if not result:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return result


def parse_strings(value: str) -> list[str]:
    result = [item for item in value.replace(",", " ").split() if item]
    if not result:
        raise argparse.ArgumentTypeError("expected at least one method")
    return result


def parse_args() -> argparse.Namespace:
    defaults = ArtBenchConfig()
    root = defaults.index_root.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint-root",
        type=Path,
        default=root / "outputs/attribution/endpoint-q60-t50-full-linear-hooks",
    )
    parser.add_argument(
        "--retrac-path",
        type=Path,
        default=root / "outputs/attribution/retrac-q60-t50-full-linear-hooks/retrac.npz",
    )
    parser.add_argument(
        "--query-manifest",
        type=Path,
        default=root / "outputs/artbench2-queries-60/manifest.json",
    )
    parser.add_argument("--metadata-csv", type=Path, default=defaults.metadata_csv)
    parser.add_argument("--image-root", type=Path, default=defaults.image_root)
    parser.add_argument(
        "--base-index-path",
        type=Path,
        default=defaults.index_root / "idx-train.pkl",
    )
    parser.add_argument(
        "--clip-similarities",
        type=Path,
        default=root / "outputs/clip-similarity-q60/similarities.npy",
    )
    parser.add_argument(
        "--removal-root",
        type=Path,
        default=root / "outputs/topk-removal-adamw-q0-9",
    )
    parser.add_argument("--query-indices", type=parse_ints, default=list(range(10)))
    parser.add_argument("--methods", type=parse_strings, default=list(METHODS))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--end-k", type=int, default=10)
    parser.add_argument("--removal-k", type=int, default=1000)
    parser.add_argument("--tile-size", type=int, default=150)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs/attribution/top10-vs-end10-adamw-q0-9",
    )
    return parser.parse_args()


def load_base_indices(path: Path) -> list[int]:
    with path.open("rb") as handle:
        result = [int(value) for value in pickle.load(handle)]
    if len(result) != len(set(result)):
        raise ValueError("base training indices contain duplicates")
    return result


def stable_ranking(scores: np.ndarray, ranking_sign: int) -> tuple[np.ndarray, np.ndarray]:
    ranking_scores = ranking_sign * np.asarray(scores, dtype=np.float64)
    if not np.isfinite(ranking_scores).all():
        raise ValueError("ranking contains non-finite scores")
    order = np.argsort(-ranking_scores, kind="stable")
    return order, ranking_scores


def validate_removal_manifest(
    *,
    path: Path,
    expected_indices: np.ndarray,
    method: str,
    query_index: int,
) -> str:
    if not path.is_file():
        return "missing"
    payload = json.loads(path.read_text(encoding="utf-8"))
    actual = np.asarray(payload["removed_metadata_indices"], dtype=np.int64)
    if not np.array_equal(actual, expected_indices):
        mismatch = int(np.flatnonzero(actual != expected_indices)[0])
        raise ValueError(
            f"removal manifest mismatch: method={method} query={query_index} "
            f"first_mismatch={mismatch} actual={actual[mismatch]} "
            f"expected={expected_indices[mismatch]}"
        )
    if payload.get("transform") != "adamw_full":
        raise ValueError(f"removal manifest is not adamw_full: {path}")
    return "exact_match"


def make_figure(
    *,
    query: dict[str, Any],
    methods: Sequence[str],
    rows: dict[tuple[str, str], list[dict[str, Any]]],
    top_k: int,
    end_k: int,
    tile_size: int,
) -> Image.Image:
    maximum = max(top_k, end_k)
    label_width = 270
    header_height = 76
    caption_height = 48
    row_gap = 7
    cell_width = tile_size + 8
    row_height = tile_size + caption_height
    row_count = 2 * len(methods)
    width = label_width + (maximum + 1) * cell_width + 8
    height = header_height + row_count * (row_height + row_gap) + 8
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = font(21, bold=True)
    label_font = font(13, bold=True)
    text_font = font(11)
    small_font = font(10)
    query_index = int(query["query_index"])
    draw.text(
        (10, 7),
        f"Exact AdamW removal ranking | Q{query_index:03d} | Top vs End",
        fill="black",
        font=title_font,
    )
    draw.text(
        (10, 38),
        f"seed={query['seed']} | {shortened(str(query['prompt']), 120)}",
        fill="black",
        font=text_font,
    )
    query_tile = image_tile(Path(query["image_path"]), tile_size)
    row_index = 0
    for method in methods:
        for tail in ("top", "end"):
            y = header_height + row_index * (row_height + row_gap)
            fill = (238, 249, 238) if tail == "top" else (248, 238, 238)
            draw.rectangle((0, y, width, y + row_height), fill=fill)
            direction = rows[(method, tail)][0]["ranking_direction"]
            draw.multiline_text(
                (10, y + 40),
                f"{METHOD_LABELS[method]}\n{tail.upper()}-{len(rows[(method, tail)])}\n"
                f"removal order: {direction}",
                fill="black",
                font=label_font,
                spacing=4,
            )
            canvas.paste(query_tile, (label_width, y))
            draw.text((label_width + 4, y + tile_size + 5), "QUERY", fill="black", font=text_font)
            for column, record in enumerate(rows[(method, tail)], start=1):
                x = label_width + column * cell_width
                canvas.paste(image_tile(Path(record["train_file"]), tile_size), (x, y))
                draw.text(
                    (x + 3, y + tile_size + 3),
                    f"rank={record['overall_rank']} src={record['source_index']}",
                    fill="black",
                    font=small_font,
                )
                draw.text(
                    (x + 3, y + tile_size + 20),
                    f"attr={record['stored_score']:+.2e} clip={record['clip_similarity']:.3f}",
                    fill="black",
                    font=small_font,
                )
            row_index += 1
    return canvas


def main() -> None:
    args = parse_args()
    if min(args.top_k, args.end_k, args.removal_k) < 1 or args.tile_size < 64:
        raise ValueError("K values must be positive and tile-size must be at least 64")
    unknown_methods = sorted(set(args.methods).difference(METHODS))
    if unknown_methods:
        raise ValueError(f"unknown methods: {unknown_methods}")
    if len(args.methods) != len(set(args.methods)):
        raise ValueError("methods must be unique")

    queries = load_queries(args.query_manifest)
    base_indices = load_base_indices(args.base_index_path)
    training = load_training_records(args.metadata_csv, args.image_root, base_indices)
    clip = np.asarray(np.load(args.clip_similarities), dtype=np.float64)
    if clip.ndim != 2 or clip.shape[0] < len(queries) or clip.shape[1] != len(base_indices):
        raise ValueError(
            f"CLIP similarities have shape {clip.shape}; expected at least "
            f"[{len(queries)}, {len(base_indices)}]"
        )
    clip_position = {source_index: position for position, source_index in enumerate(base_indices)}
    training_by_source = {record["source_index"]: record for record in training}
    if not args.query_indices or min(args.query_indices) < 0 or max(args.query_indices) >= len(queries):
        raise ValueError(f"query indices must be inside [0, {len(queries) - 1}]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranking_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    figures = 0

    for query_index in args.query_indices:
        figure_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for method in args.methods:
            scores, source_indices, ranking_sign, definition = method_scores(
                method=method,
                query_index=query_index,
                endpoint_root=args.endpoint_root,
                retrac_path=args.retrac_path,
            )
            if len(scores) != len(base_indices):
                raise ValueError(f"candidate count differs for {method}")
            order, ranking_scores = stable_ranking(scores, ranking_sign)
            expected_removed = source_indices[order[: args.removal_k]]
            manifest_path = (
                args.removal_root
                / method
                / f"query-{query_index:03d}"
                / f"topk-{args.removal_k:04d}"
                / "removal-manifest.json"
            )
            manifest_status = validate_removal_manifest(
                path=manifest_path,
                expected_indices=expected_removed,
                method=method,
                query_index=query_index,
            )
            validation.append(
                {
                    "method": method,
                    "query_index": query_index,
                    "manifest": str(manifest_path),
                    "status": manifest_status,
                }
            )
            print(
                f"[top1000-check] method={method} query={query_index} "
                f"status={manifest_status}",
                flush=True,
            )

            selections = {
                "top": order[: args.top_k],
                "end": order[-args.end_k :],
            }
            for tail, positions in selections.items():
                selected_rows = []
                for position in positions:
                    position = int(position)
                    source_index = int(source_indices[position])
                    record = training_by_source[source_index]
                    clip_value = float(clip[query_index, clip_position[source_index]])
                    overall_rank = int(np.flatnonzero(order == position)[0]) + 1
                    row = {
                        "query_index": query_index,
                        "prompt": queries[query_index]["prompt"],
                        "seed": queries[query_index]["seed"],
                        "method": method,
                        "score_definition": definition,
                        "ranking_direction": "descending" if ranking_sign == 1 else "ascending",
                        "tail": tail,
                        "overall_rank": overall_rank,
                        "train_position": position,
                        "source_index": source_index,
                        "train_label": record["label"],
                        "train_file": str(record["image_path"]),
                        "stored_score": float(scores[position]),
                        "signed_ranking_score": float(ranking_scores[position]),
                        "clip_similarity": clip_value,
                    }
                    selected_rows.append(row)
                    ranking_rows.append(row)
                figure_rows[(method, tail)] = selected_rows
                clip_values = np.asarray(
                    [row["clip_similarity"] for row in selected_rows], dtype=np.float64
                )
                summary_rows.append(
                    {
                        "query_index": query_index,
                        "method": method,
                        "tail": tail,
                        "count": len(selected_rows),
                        "clip_mean": float(clip_values.mean()),
                        "clip_median": float(np.median(clip_values)),
                        "clip_min": float(clip_values.min()),
                        "clip_max": float(clip_values.max()),
                        "top1000_manifest_status": manifest_status,
                    }
                )

            top_summary = summary_rows[-2]
            end_summary = summary_rows[-1]
            comparison = {
                "query_index": query_index,
                "method": method,
                "top10_clip_mean": top_summary["clip_mean"],
                "end10_clip_mean": end_summary["clip_mean"],
                "top_minus_end_clip_mean": (
                    float(top_summary["clip_mean"]) - float(end_summary["clip_mean"])
                ),
                "top10_clip_median": top_summary["clip_median"],
                "end10_clip_median": end_summary["clip_median"],
                "top1000_manifest_status": manifest_status,
            }
            comparison_rows.append(comparison)
            print(
                f"[clip] method={method} query={query_index} "
                f"top10_mean={comparison['top10_clip_mean']:.4f} "
                f"end10_mean={comparison['end10_clip_mean']:.4f} "
                f"delta={comparison['top_minus_end_clip_mean']:+.4f}",
                flush=True,
            )

        figure = make_figure(
            query=queries[query_index],
            methods=args.methods,
            rows=figure_rows,
            top_k=args.top_k,
            end_k=args.end_k,
            tile_size=args.tile_size,
        )
        figure_path = args.output_dir / f"Q{query_index:03d}_top10_vs_end10.jpg"
        figure.save(figure_path, quality=94, subsampling=0)
        figures += 1
        print(f"[saved] {figure_path}", flush=True)

    for filename, rows in (
        ("selected_rankings.csv", ranking_rows),
        ("clip_summary.csv", summary_rows),
        ("clip_top_vs_end.csv", comparison_rows),
    ):
        with (args.output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.output_dir / "top1000_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[done] figures={figures} selected_rows={len(ranking_rows)} "
        f"output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
