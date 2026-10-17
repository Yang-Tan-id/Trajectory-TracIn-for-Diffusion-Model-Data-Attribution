#!/usr/bin/env python3
"""Plot query endpoint and top training images for three attribution methods."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
REFINE_ROOT = SHAPES_ROOT.parent
if str(SHAPES_ROOT / "script") not in sys.path:
    sys.path.insert(0, str(SHAPES_ROOT / "script"))

from plot_endpoint_top6_datapoints import (
    Query,
    display_image,
    endpoint_path,
    format_datapoint_labels,
    load_dataset,
    load_top_scores,
    score_root,
)


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    namespace: str
    score_directory: str
    ranking_sign: int


METHODS = (
    Method(
        key="endpoint_pollute_raw",
        label="Endpoint-Pollute\nAdamW · RAW · P1",
        namespace=(
            "recreate_adamw_full_polluted_endpoint_delta_l2normalized_"
            "timestamp_sum_squared_aligned100x1_q0_99"
        ),
        score_directory="score",
        ranking_sign=1,
    ),
    Method(
        key="retrac_query_train_l2",
        label="ReTrac\nAdamW · BOTH-L2 · M1",
        namespace="paper_retrac_adamw_full_exact4_endpoint100x1_q0_99",
        score_directory="score_query_train_l2_normalized",
        ranking_sign=-1,
    ),
    Method(
        key="endpoint_tracin_raw",
        label="Endpoint-TracIn\nAdamW · RAW · M1",
        namespace="endpoint_tracin_adamw_full_train100x1_query100x1_q0_99",
        score_directory="score",
        ranking_sign=-1,
    ),
)


def parse_ids(text: str) -> list[int]:
    values = [int(token) for token in text.replace(",", " ").split()]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("query ids must be nonempty and unique")
    return values


def load_queries(path: Path, query_ids: list[int]) -> list[Query]:
    records = json.loads(path.read_text())["queries"]
    if min(query_ids) < 0 or max(query_ids) >= len(records):
        raise ValueError(
            f"query ids must be in [0, {len(records) - 1}], got {query_ids}"
        )
    return [
        Query(
            query_id=query_id,
            prompt=str(records[query_id]["prompt"]),
            initial_seed=int(records[query_id]["initial_seed"]),
        )
        for query_id in query_ids
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    default_data_root = Path(
        os.environ.get(
            "THREEDSHAPES_DATA_ROOT",
            str(REFINE_ROOT / "dataset" / "3dshapes" / "20000"),
        )
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-file",
        type=Path,
        default=SHAPES_ROOT / "queries_in_distribution_plus_zero_seed_100_219.json",
    )
    parser.add_argument(
        "--query-ids",
        type=parse_ids,
        default=list(range(20)),
        help="Comma/space-separated query ids (default: 0-19).",
    )
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dataset", type=Path, default=default_data_root / "dataset.npz")
    parser.add_argument("--sample-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=170)
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    queries = load_queries(args.query_file, args.query_ids)
    images, labels = load_dataset(args.dataset)
    result_root = SHAPES_ROOT / "result" / args.experiment
    sample_root = args.sample_root or result_root / "sample_ddim_eta0_1000"
    output_dir = args.output_dir or (
        result_root / "eval" / "endpoint_top10_three_method_q0_19"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_stem = f"seed_{args.train_seed}_epoch_{args.epochs:04d}"

    ranking_rows: list[dict[str, object]] = []
    overlap_rows: list[dict[str, object]] = []
    for query_position, query in enumerate(queries, start=1):
        endpoint = display_image(
            np.load(
                endpoint_path(sample_root, query, checkpoint_stem),
                allow_pickle=False,
            )
        )
        selected: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        root = score_root(result_root, args.train_seed, query)
        for method in METHODS:
            directory = (
                root
                / f"traj_tracin_{method.namespace}"
                / method.score_directory
            )
            selected[method.key] = load_top_scores(
                directory,
                args.top_k,
                len(images),
                method.ranking_sign,
            )

        figure, axes = plt.subplots(
            len(METHODS),
            args.top_k + 1,
            figsize=(2.15 * (args.top_k + 1), 6.9),
            gridspec_kw={"width_ratios": [1.12] + [1.0] * args.top_k},
            squeeze=False,
        )
        for row, method in enumerate(METHODS):
            axes[row, 0].imshow(endpoint)
            axes[row, 0].set_ylabel(
                method.label,
                rotation=0,
                ha="right",
                va="center",
                fontsize=8,
                labelpad=8,
            )
            indices, scores = selected[method.key]
            for rank, (index, score) in enumerate(
                zip(indices, scores), start=1
            ):
                axis = axes[row, rank]
                axis.imshow(display_image(images[int(index)]))
                axis.set_xlabel(
                    f"idx {int(index)}\n{float(score):+.2e}\n"
                    f"{format_datapoint_labels(labels[int(index)])}",
                    fontsize=5.8,
                )
                ranking_rows.append(
                    {
                        "query_id": query.query_id,
                        "method": method.key,
                        "rank": rank,
                        "dataset_index": int(index),
                        "stored_score": float(score),
                        "ranking_sign": method.ranking_sign,
                        "ranking_score": method.ranking_sign * float(score),
                        "labels": "|".join(labels[int(index)]),
                    }
                )
            for axis in axes[row]:
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_visible(False)

        axes[0, 0].set_title("Query endpoint", fontsize=9, fontweight="bold")
        for rank in range(1, args.top_k + 1):
            axes[0, rank].set_title(
                f"Top {rank}", fontsize=9, fontweight="bold"
            )
        prompt_title = " · ".join(query.prompt.split(","))
        figure.suptitle(
            f"Q{query.query_id} · seed {query.initial_seed} · {prompt_title}",
            fontsize=13,
            fontweight="bold",
            y=0.995,
        )
        figure.subplots_adjust(
            left=0.16,
            right=0.995,
            top=0.94,
            bottom=0.06,
            wspace=0.08,
            hspace=0.58,
        )
        output = output_dir / f"Q{query.query_id:03d}_top{args.top_k}_comparison.png"
        figure.savefig(
            output, dpi=args.dpi, bbox_inches="tight", facecolor="white"
        )
        plt.close(figure)

        for left_position, left in enumerate(METHODS):
            left_set = set(int(x) for x in selected[left.key][0])
            for right in METHODS[left_position + 1 :]:
                right_set = set(int(x) for x in selected[right.key][0])
                intersection = len(left_set & right_set)
                overlap_rows.append(
                    {
                        "query_id": query.query_id,
                        "method_left": left.key,
                        "method_right": right.key,
                        "top_k": args.top_k,
                        "intersection": intersection,
                        "overlap_fraction": intersection / args.top_k,
                    }
                )
        print(
            f"[saved {query_position}/{len(queries)}] {output}", flush=True
        )

    write_csv(output_dir / "top10_rankings.csv", ranking_rows)
    write_csv(output_dir / "top10_method_overlap.csv", overlap_rows)
    manifest = {
        "query_ids": args.query_ids,
        "top_k": args.top_k,
        "methods": [method.__dict__ for method in METHODS],
        "ranking_semantics": {
            "endpoint_pollute_raw": "stored score descending (P1)",
            "retrac_query_train_l2": "stored score ascending (-stored, M1)",
            "endpoint_tracin_raw": "stored score ascending (-stored, M1)",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[done] figures={len(queries)} output={output_dir}")


if __name__ == "__main__":
    main()
