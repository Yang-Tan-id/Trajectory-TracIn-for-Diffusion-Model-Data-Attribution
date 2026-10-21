#!/usr/bin/env python3
"""Plot full-10k CelebA endpoint top training images for every score variant."""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_endpoint_top10_comparison import (
    REFINE_ROOT,
    display_image,
    endpoint_path,
    load_dataset,
    load_queries,
    prompt_tag,
    save_figure,
    short_attributes,
)


CELEBA_ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (
    ("raw", "score"),
    ("query-L2", "score_query_normalized"),
    ("train-L2", "score_train_l2_normalized"),
    ("both-L2", "score_query_train_l2_normalized"),
)


@dataclass(frozen=True)
class Bank:
    method: str
    transform: str
    variant: str
    namespace: str
    score_directory: str
    ranking_sign: int

    @property
    def title(self) -> str:
        direction = "largest" if self.ranking_sign == 1 else "smallest"
        return f"{self.method} · {self.variant} · {direction}"


def parse_values(text: str) -> list[str]:
    return [value.strip() for value in text.replace(" ", ",").split(",") if value.strip()]


def banks_for_transform(transform: str) -> list[Bank]:
    pollute_namespace = (
        f"celeba_endpoint_pollute_{transform}_train1x50_query1x50_"
        "timestamp_sum_squared_shape3d_semantics_full10000"
    )
    tracin_namespace = (
        f"celeba_endpoint_tracin_{transform}_"
        "train_mean50_query_mean50_linear_shape3d_semantics_full10000"
    )
    output: list[Bank] = []
    for variant, directory in VARIANTS:
        output.append(
            Bank(
                "Pollute",
                transform,
                variant,
                pollute_namespace,
                directory,
                +1,
            )
        )
    for variant, directory in VARIANTS:
        output.append(
            Bank(
                "TracIn",
                transform,
                variant,
                tracin_namespace,
                directory,
                -1,
            )
        )
    return output


def score_directory(result_root: Path, train_seed: int, query, bank: Bank) -> Path:
    return (
        result_root
        / "attribution_score"
        / "prompted_solo"
        / f"train_seed_{train_seed}"
        / f"query_{prompt_tag(query.prompt)}"
        / f"initial_seed_{query.initial_seed}"
        / f"traj_tracin_{bank.namespace}"
        / bank.score_directory
    )


def select_top(
    directory: Path, *, top_k: int, ranking_sign: int, dataset_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = np.asarray(np.load(directory / "scores.npy"), dtype=np.float64).reshape(-1)
    indices = np.asarray(
        np.load(directory / "score_indices.npy"), dtype=np.int64
    ).reshape(-1)
    if scores.shape != indices.shape:
        raise ValueError(f"score/index mismatch under {directory}")
    if len(scores) != dataset_size or not np.array_equal(
        indices, np.arange(dataset_size, dtype=np.int64)
    ):
        raise ValueError(f"score bank is not the complete ordered dataset under {directory}")
    ranking = np.where(np.isfinite(scores), ranking_sign * scores, -np.inf)
    order = np.argsort(-ranking, kind="stable")[:top_k]
    if len(order) != top_k or np.any(~np.isfinite(ranking[order])):
        raise ValueError(f"fewer than {top_k} finite scores under {directory}")
    return indices[order], scores[order], ranking[order]


def main() -> None:
    data_root = Path(
        os.environ.get(
            "CELEBA_ATTR40_DATA_ROOT",
            str(REFINE_ROOT / "dataset" / "celeba" / "10000_attr40"),
        )
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--query-ids", default=",".join(str(value) for value in range(10)))
    parser.add_argument(
        "--transforms", default="raw_gradient,adamw_full"
    )
    parser.add_argument("--experiment", default="attr40_experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--dataset", type=Path, default=data_root / "dataset.npz")
    parser.add_argument(
        "--original-indices",
        type=Path,
        default=data_root / "attribution_5k_indices.npy",
    )
    parser.add_argument("--sample-root", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    query_ids = [int(value) for value in parse_values(args.query_ids)]
    transforms = parse_values(args.transforms)
    unknown = sorted(set(transforms) - {"raw_gradient", "adamw_full"})
    if unknown:
        parser.error(f"unknown transforms: {unknown}")
    if not query_ids or not transforms or args.top_k <= 0:
        parser.error("queries/transforms must be nonempty and --top-k must be positive")

    queries = load_queries(args.query_file, query_ids)
    images, labels, label_names, filenames = load_dataset(args.dataset)
    original_indices = set(
        map(
            int,
            np.asarray(np.load(args.original_indices), dtype=np.int64).reshape(-1),
        )
    )
    if len(original_indices) != 5000:
        raise ValueError(f"expected 5000 original indices: {args.original_indices}")

    result_root = CELEBA_ROOT / "result" / args.experiment
    sample_root = args.sample_root or result_root / "sample_ddim_eta0_50"
    output_dir = args.output_dir or (
        result_root / "eval" / "endpoint_full10k_top10_all_variants"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_stem = f"seed_{args.train_seed}_epoch_{args.epochs:04d}"

    csv_rows: list[dict[str, object]] = []
    for query in queries:
        query_image = display_image(
            np.load(
                endpoint_path(sample_root, checkpoint_stem, query),
                allow_pickle=False,
            )
        )
        for transform in transforms:
            banks = banks_for_transform(transform)
            selections = []
            for bank in banks:
                selected = select_top(
                    score_directory(result_root, args.train_seed, query, bank),
                    top_k=args.top_k,
                    ranking_sign=bank.ranking_sign,
                    dataset_size=len(images),
                )
                selections.append(selected)
                for rank, (index, score, ranking) in enumerate(
                    zip(*selected, strict=True), start=1
                ):
                    idx = int(index)
                    csv_rows.append(
                        {
                            "query_id": query.query_id,
                            "prompt": query.prompt,
                            "initial_seed": query.initial_seed,
                            "method": bank.method,
                            "transform": bank.transform,
                            "variant": bank.variant,
                            "ranking_sign": bank.ranking_sign,
                            "rank": rank,
                            "dataset_index": idx,
                            "candidate_half": (
                                "original5000" if idx in original_indices else "remaining5000"
                            ),
                            "source_filename": filenames[idx],
                            "stored_score": float(score),
                            "ranking_score": float(ranking),
                            "positive_attributes": ",".join(
                                label_names[position]
                                for position in np.flatnonzero(labels[idx] > 0)
                            ),
                        }
                    )

            figure, axes = plt.subplots(
                len(banks),
                args.top_k + 1,
                figsize=(2.05 * (args.top_k + 1), 2.15 * len(banks)),
                squeeze=False,
            )
            for row, (bank, selected) in enumerate(zip(banks, selections, strict=True)):
                axes[row, 0].imshow(query_image)
                axes[row, 0].set_ylabel(
                    bank.title, rotation=0, ha="right", va="center", fontsize=7
                )
                if row == len(banks) - 1:
                    axes[row, 0].set_xlabel(
                        f"Q{query.query_id} seed={query.initial_seed}\n{query.prompt}",
                        fontsize=6,
                    )
                indices, scores, rankings = selected
                for rank, (index, score, ranking) in enumerate(
                    zip(indices, scores, rankings, strict=True), start=1
                ):
                    idx = int(index)
                    source = "OLD" if idx in original_indices else "NEW"
                    color = "#2f6db0" if source == "OLD" else "#d97706"
                    axis = axes[row, rank]
                    axis.imshow(display_image(images[idx]))
                    axis.set_xlabel(
                        f"{source} idx={idx}\nscore={score:+.2e}\n"
                        f"{short_attributes(labels[idx], label_names)}",
                        fontsize=5.4,
                        color=color,
                    )
                    for spine in axis.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(1.5)
                        spine.set_edgecolor(color)
                for axis in axes[row]:
                    axis.set_xticks([])
                    axis.set_yticks([])
                for spine in axes[row, 0].spines.values():
                    spine.set_visible(False)

            axes[0, 0].set_title("Generated query", fontsize=9, fontweight="bold")
            for rank in range(1, args.top_k + 1):
                axes[0, rank].set_title(f"Top {rank}", fontsize=9, fontweight="bold")
            figure.suptitle(
                f"CelebA full-10k endpoint attribution · Q{query.query_id} · {transform}\n"
                "blue=original 5k, orange=remaining 5k",
                fontsize=12,
                fontweight="bold",
            )
            figure.subplots_adjust(
                left=0.18,
                right=0.995,
                top=0.94,
                bottom=0.035,
                hspace=0.55,
                wspace=0.07,
            )
            save_figure(
                figure,
                output_dir
                / f"q{query.query_id:03d}_{transform}_top{args.top_k}.png",
                args.dpi,
            )

    csv_path = output_dir / f"top{args.top_k}_rankings.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"[saved] {csv_path}")
    print(
        f"[done] queries={len(queries)} transforms={transforms} "
        f"figures={len(queries) * len(transforms)} output={output_dir}"
    )


if __name__ == "__main__":
    main()
