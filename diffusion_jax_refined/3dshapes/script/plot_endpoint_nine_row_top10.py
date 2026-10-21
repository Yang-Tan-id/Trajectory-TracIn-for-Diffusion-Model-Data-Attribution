#!/usr/bin/env python3
"""Plot nine-row Shape3D endpoint attribution comparisons.

For each query and each requested train transform, the figure contains four
Endpoint-Pollute normalization variants, four Endpoint-TracIn variants, and
the standard query+train-L2 ReTrac score.  Endpoint-Pollute ranks stored P1
timestamp-sum-squared scores descending; Endpoint-TracIn and ReTrac rank their
stored linear scores ascending (M1).
"""

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


VARIANTS = (
    ("raw", "score", "RAW"),
    ("query_l2", "score_query_normalized", "QUERY-L2"),
    ("train_l2", "score_train_l2_normalized", "TRAIN-L2"),
    (
        "query_train_l2",
        "score_query_train_l2_normalized",
        "BOTH-L2",
    ),
)


@dataclass(frozen=True)
class Transform:
    key: str
    label: str
    pollute_namespace: str
    tracin_namespace: str
    retrac_namespace: str


@dataclass(frozen=True)
class Bank:
    key: str
    label: str
    namespace: str
    score_directory: str
    ranking_sign: int


TRANSFORMS = {
    "raw_gradient": Transform(
        key="raw_gradient",
        label="RAW GRADIENT",
        pollute_namespace=(
            "recreate_raw_polluted_endpoint_delta_l2normalized_"
            "timestamp_sum_squared_q0_99"
        ),
        tracin_namespace="endpoint_tracin_train100x1_query100x1_q0_99",
        retrac_namespace="paper_retrac_exact4_endpoint100x1_q0_99",
    ),
    "adamw_full": Transform(
        key="adamw_full",
        label="ADAMW FULL",
        pollute_namespace=(
            "recreate_adamw_full_polluted_endpoint_delta_l2normalized_"
            "timestamp_sum_squared_q0_99"
        ),
        tracin_namespace=(
            "endpoint_tracin_adamw_full_train100x1_query100x1_q0_99"
        ),
        retrac_namespace=(
            "paper_retrac_adamw_full_exact4_endpoint100x1_q0_99"
        ),
    ),
}


def parse_values(text: str) -> list[str]:
    return [
        token.strip()
        for token in text.replace(" ", ",").split(",")
        if token.strip()
    ]


def load_queries(path: Path, query_ids: list[int]) -> list[Query]:
    records = json.loads(path.read_text())["queries"]
    if not query_ids or min(query_ids) < 0 or max(query_ids) >= len(records):
        raise ValueError(
            f"query ids must be within [0, {len(records) - 1}]: {query_ids}"
        )
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("query ids must be unique")
    return [
        Query(
            query_id=query_id,
            prompt=str(records[query_id]["prompt"]),
            initial_seed=int(records[query_id]["initial_seed"]),
        )
        for query_id in query_ids
    ]


def banks_for_transform(transform: Transform) -> list[Bank]:
    banks: list[Bank] = []
    for variant, directory, label in VARIANTS:
        banks.append(
            Bank(
                key=f"pollute_{variant}",
                label=f"Endpoint-Pollute\n{label} · largest",
                namespace=transform.pollute_namespace,
                score_directory=directory,
                ranking_sign=+1,
            )
        )
    for variant, directory, label in VARIANTS:
        banks.append(
            Bank(
                key=f"tracin_{variant}",
                label=f"Endpoint-TracIn\n{label} · smallest",
                namespace=transform.tracin_namespace,
                score_directory=directory,
                ranking_sign=-1,
            )
        )
    banks.append(
        Bank(
            key="retrac_query_train_l2",
            label="ReTrac\nBOTH-L2 · smallest",
            namespace=transform.retrac_namespace,
            score_directory="score_query_train_l2_normalized",
            ranking_sign=-1,
        )
    )
    if len(banks) != 9:
        raise AssertionError(f"expected nine banks, got {len(banks)}")
    return banks


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
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
        default=(
            SHAPES_ROOT
            / "queries_in_distribution_plus_zero_seed_100_219.json"
        ),
    )
    parser.add_argument("--query-ids", default=",".join(map(str, range(10))))
    parser.add_argument(
        "--transforms", default="raw_gradient,adamw_full"
    )
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--dataset", type=Path, default=default_data_root / "dataset.npz"
    )
    parser.add_argument("--sample-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    query_ids = [int(value) for value in parse_values(args.query_ids)]
    transform_names = parse_values(args.transforms)
    unknown = sorted(set(transform_names) - set(TRANSFORMS))
    if unknown:
        parser.error(f"unknown transforms: {unknown}")
    if args.top_k <= 0 or not transform_names:
        parser.error("--top-k must be positive and transforms must be nonempty")

    queries = load_queries(args.query_file, query_ids)
    images, labels = load_dataset(args.dataset)
    result_root = SHAPES_ROOT / "result" / args.experiment
    sample_root = args.sample_root or result_root / "sample_ddim_eta0_1000"
    output_dir = args.output_dir or (
        result_root / "eval" / "endpoint_top10_nine_row_q0_9"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_stem = f"seed_{args.train_seed}_epoch_{args.epochs:04d}"

    ranking_rows: list[dict[str, object]] = []
    overlap_rows: list[dict[str, object]] = []
    figure_count = 0
    for query in queries:
        endpoint = display_image(
            np.load(
                endpoint_path(sample_root, query, checkpoint_stem),
                allow_pickle=False,
            )
        )
        query_score_root = score_root(result_root, args.train_seed, query)
        for transform_name in transform_names:
            transform = TRANSFORMS[transform_name]
            banks = banks_for_transform(transform)
            selections: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for bank in banks:
                directory = (
                    query_score_root
                    / f"traj_tracin_{bank.namespace}"
                    / bank.score_directory
                )
                indices, scores = load_top_scores(
                    directory,
                    args.top_k,
                    len(images),
                    bank.ranking_sign,
                )
                selections[bank.key] = (indices, scores)
                for rank, (index, score) in enumerate(
                    zip(indices, scores, strict=True), start=1
                ):
                    ranking_rows.append(
                        {
                            "query_id": query.query_id,
                            "prompt": query.prompt,
                            "initial_seed": query.initial_seed,
                            "transform": transform_name,
                            "method_variant": bank.key,
                            "namespace": bank.namespace,
                            "score_directory": bank.score_directory,
                            "ranking_sign": bank.ranking_sign,
                            "rank": rank,
                            "dataset_index": int(index),
                            "stored_score": float(score),
                            "ranking_score": bank.ranking_sign * float(score),
                            "labels": "|".join(labels[int(index)]),
                        }
                    )

            figure, axes = plt.subplots(
                len(banks),
                args.top_k + 1,
                figsize=(2.05 * (args.top_k + 1), 2.05 * len(banks)),
                gridspec_kw={"width_ratios": [1.12] + [1.0] * args.top_k},
                squeeze=False,
            )
            for row, bank in enumerate(banks):
                axes[row, 0].imshow(endpoint)
                axes[row, 0].set_ylabel(
                    bank.label,
                    rotation=0,
                    ha="right",
                    va="center",
                    fontsize=6.8,
                    labelpad=8,
                )
                indices, scores = selections[bank.key]
                for rank, (index, score) in enumerate(
                    zip(indices, scores, strict=True), start=1
                ):
                    axis = axes[row, rank]
                    axis.imshow(display_image(images[int(index)]))
                    axis.set_xlabel(
                        f"idx {int(index)}\n{float(score):+.2e}\n"
                        f"{format_datapoint_labels(labels[int(index)])}",
                        fontsize=5.2,
                    )
                for axis in axes[row]:
                    axis.set_xticks([])
                    axis.set_yticks([])
                    for spine in axis.spines.values():
                        spine.set_visible(False)

            axes[0, 0].set_title(
                "Query endpoint", fontsize=9, fontweight="bold"
            )
            for rank in range(1, args.top_k + 1):
                axes[0, rank].set_title(
                    f"Top {rank}", fontsize=9, fontweight="bold"
                )
            prompt_title = " · ".join(query.prompt.split(","))
            figure.suptitle(
                f"Shape3D endpoint attribution · Q{query.query_id} · "
                f"{transform.label}\n{prompt_title}",
                fontsize=12,
                fontweight="bold",
                y=0.997,
            )
            figure.subplots_adjust(
                left=0.18,
                right=0.995,
                top=0.955,
                bottom=0.025,
                wspace=0.07,
                hspace=0.54,
            )
            output = (
                output_dir
                / f"Q{query.query_id:03d}_{transform_name}_top{args.top_k}.png"
            )
            figure.savefig(
                output,
                dpi=args.dpi,
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close(figure)
            figure_count += 1
            print(f"[saved {figure_count}] {output}", flush=True)

            for left_position, left in enumerate(banks):
                left_set = set(map(int, selections[left.key][0]))
                for right in banks[left_position + 1 :]:
                    right_set = set(map(int, selections[right.key][0]))
                    intersection = len(left_set & right_set)
                    overlap_rows.append(
                        {
                            "query_id": query.query_id,
                            "transform": transform_name,
                            "method_left": left.key,
                            "method_right": right.key,
                            "top_k": args.top_k,
                            "intersection": intersection,
                            "overlap_fraction": intersection / args.top_k,
                        }
                    )

    write_csv(output_dir / f"top{args.top_k}_rankings.csv", ranking_rows)
    write_csv(output_dir / f"top{args.top_k}_overlap.csv", overlap_rows)
    manifest = {
        "query_ids": query_ids,
        "transforms": transform_names,
        "top_k": args.top_k,
        "rows_per_figure": 9,
        "ranking": {
            "endpoint_pollute": "stored timestamp-sum-squared score descending (P1)",
            "endpoint_tracin": "stored linear score ascending (M1)",
            "retrac": "stored linear score ascending (M1)",
        },
        "transform_namespaces": {
            key: value.__dict__ for key, value in TRANSFORMS.items()
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        f"[done] queries={len(queries)} transforms={len(transform_names)} "
        f"figures={figure_count} output={output_dir}"
    )


if __name__ == "__main__":
    main()
