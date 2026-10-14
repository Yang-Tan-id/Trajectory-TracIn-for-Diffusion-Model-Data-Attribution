#!/usr/bin/env python3
"""Compare top-ranked CelebA training images for two endpoint scores.

The default comparison is the raw-variant AdamW-full endpoint-TracIn score
after squaring each final datapoint score versus the raw-variant AdamW-full
endpoint-pollute timestamp-sum-squared score.  ``score_indices.npy`` is always
used to map score positions back to rows in the 10k CelebA training dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CELEBA_ROOT = Path(__file__).resolve().parents[1]
REFINE_ROOT = CELEBA_ROOT.parent

DEFAULT_ENDPOINT_TRACIN = (
    "celeba_endpoint_tracin_adamw_full_"
    "train_mean50_query_mean50_linear_shape3d_semantics"
)
DEFAULT_ENDPOINT_POLLUTE = (
    "celeba_endpoint_pollute_adamw_full_"
    "train1x50_query1x50_timestamp_sum_squared_shape3d_semantics"
)


@dataclass(frozen=True)
class Query:
    query_id: int
    prompt: str
    initial_seed: int


@dataclass(frozen=True)
class Method:
    key: str
    title: str
    namespace: str
    score_component: str
    transform: str


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split() if value]


def prompt_tag(prompt: str) -> str:
    # Keep this identical to dataset_config_attr40._prompt_path_tag without
    # importing the training config (which may initialize environment state).
    import re

    text = str(prompt).strip().replace(",", "__")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")[:80] or "empty_prompt"


def query_id(record: dict[str, object], fallback: int) -> int:
    return int(record.get("query_id", fallback))


def query_seed(record: dict[str, object]) -> int:
    return int(record.get("initial_seed", record.get("seed")))


def load_queries(path: Path, selected_ids: list[int]) -> list[Query]:
    records = json.loads(path.read_text())["queries"]
    by_id = {query_id(record, index): record for index, record in enumerate(records)}
    missing = [value for value in selected_ids if value not in by_id]
    if missing:
        raise ValueError(f"query IDs absent from {path}: {missing}")
    return [
        Query(value, str(by_id[value]["prompt"]), query_seed(by_id[value]))
        for value in selected_ids
    ]


def load_dataset(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        images = np.asarray(payload["images"])
        labels = np.asarray(payload["labels"])
        label_names = [str(value) for value in payload["label_names"].tolist()]
        if "source_filenames" in payload.files:
            filenames = [str(value) for value in payload["source_filenames"].tolist()]
        else:
            filenames = [f"row_{index}" for index in range(len(images))]
    if images.ndim != 4 or len(labels) != len(images):
        raise ValueError(f"invalid CelebA dataset arrays in {path}")
    return images, labels, label_names, filenames


def display_image(value: np.ndarray) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError(f"expected one image, got {image.shape}")
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.ndim != 3 or image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"expected HWC image, got {image.shape}")
    if image.shape[-1] == 1:
        image = image[..., 0]
    if image.dtype == np.uint8:
        return image
    image = image.astype(np.float32)
    finite = image[np.isfinite(image)]
    if not finite.size:
        raise ValueError("image has no finite pixels")
    if float(finite.min()) < -0.01:
        image = (image + 1.0) / 2.0
    elif float(finite.max()) > 1.01:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0)


def score_dir(result_root: Path, train_seed: int, query: Query, method: Method) -> Path:
    return (
        result_root
        / "attribution_score"
        / "prompted_solo"
        / f"train_seed_{train_seed}"
        / f"query_{prompt_tag(query.prompt)}"
        / f"initial_seed_{query.initial_seed}"
        / f"traj_tracin_{method.namespace}"
        / method.score_component
    )


def endpoint_path(sample_root: Path, checkpoint_stem: str, query: Query) -> Path:
    root = (
        sample_root
        / "cifar"
        / f"prompt_{prompt_tag(query.prompt)}"
        / f"model_prompted_solo__ckpt_{checkpoint_stem}"
        / f"seed_{query.initial_seed:06d}"
    )
    for name in ("decoded_final.npy", "final_state.npy"):
        candidate = root / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"query endpoint absent: {root}")


def transformed_scores(scores: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return scores
    if transform == "square":
        return np.square(scores)
    raise ValueError(transform)


def top_scores(directory: Path, method: Method, top_k: int, dataset_size: int):
    score_path = directory / "scores.npy"
    index_path = directory / "score_indices.npy"
    if not score_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(f"missing score artifact under {directory}")
    stored = np.asarray(np.load(score_path), dtype=np.float64).reshape(-1)
    indices = np.asarray(np.load(index_path), dtype=np.int64).reshape(-1)
    if stored.shape != indices.shape:
        raise ValueError(f"score/index mismatch under {directory}")
    if np.any(indices < 0) or np.any(indices >= dataset_size):
        raise IndexError(f"dataset index outside [0, {dataset_size}) under {directory}")
    ranking = transformed_scores(stored, method.transform)
    finite_ranking = np.where(np.isfinite(ranking), ranking, -np.inf)
    order = np.argsort(-finite_ranking, kind="stable")[:top_k]
    if len(order) != top_k or np.any(~np.isfinite(finite_ranking[order])):
        raise ValueError(f"fewer than {top_k} finite scores under {directory}")
    return indices[order], stored[order], ranking[order]


def short_attributes(row: np.ndarray, names: list[str], limit: int = 4) -> str:
    active = [names[index] for index in np.flatnonzero(row > 0)]
    if not active:
        return "no positive attrs"
    suffix = "…" if len(active) > limit else ""
    return ", ".join(active[:limit]) + suffix


def draw_row(
    axes,
    *,
    query: Query,
    query_image: np.ndarray,
    method: Method,
    top_indices: np.ndarray,
    stored_scores: np.ndarray,
    ranking_scores: np.ndarray,
    images: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    filenames: list[str],
    show_query: bool,
):
    axes[0].imshow(query_image)
    axes[0].set_ylabel(
        f"Q{query.query_id}  {method.title}", rotation=0, ha="right", va="center", fontsize=7
    )
    if show_query:
        axes[0].set_xlabel(f"query seed {query.initial_seed}\n{query.prompt}", fontsize=6.3)
    for rank, (index, stored, ranking) in enumerate(
        zip(top_indices, stored_scores, ranking_scores, strict=True), start=1
    ):
        idx = int(index)
        axes[rank].imshow(display_image(images[idx]))
        score_text = (
            f"score²={ranking:.2e}\nraw={stored:+.2e}"
            if method.transform == "square"
            else f"score={ranking:.2e}"
        )
        axes[rank].set_xlabel(
            f"#{rank} idx={idx} {filenames[idx]}\n{score_text}\n"
            f"{short_attributes(labels[idx], label_names)}",
            fontsize=5.6,
        )
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)


def save_figure(figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"[saved] {path}", flush=True)


def main() -> None:
    data_root = Path(
        os.environ.get(
            "CELEBA_ATTR40_DATA_ROOT",
            str(REFINE_ROOT / "dataset" / "celeba" / "10000_attr40"),
        )
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--query-ids", default=",".join(str(i) for i in range(10)))
    parser.add_argument("--experiment", default="attr40_experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--dataset", type=Path, default=data_root / "dataset.npz")
    parser.add_argument("--sample-root", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--endpoint-tracin-namespace", default=DEFAULT_ENDPOINT_TRACIN)
    parser.add_argument("--endpoint-pollute-namespace", default=DEFAULT_ENDPOINT_POLLUTE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    selected_ids = parse_ints(args.query_ids)
    if not selected_ids or args.top_k <= 0:
        parser.error("--query-ids must be nonempty and --top-k must be positive")

    queries = load_queries(args.query_file, selected_ids)
    images, labels, label_names, filenames = load_dataset(args.dataset)
    result_root = CELEBA_ROOT / "result" / args.experiment
    sample_root = args.sample_root or result_root / "sample_ddim_eta0_50"
    output_dir = args.output_dir or result_root / "eval" / "endpoint_top10_comparison"
    checkpoint_stem = f"seed_{args.train_seed}_epoch_{args.epochs:04d}"
    methods = (
        Method(
            "endpoint_tracin_square",
            "endpoint-TracIn AdamW-full raw → final score²",
            args.endpoint_tracin_namespace,
            "score",
            "square",
        ),
        Method(
            "endpoint_pollute_timestamp_sum_squared",
            "endpoint-pollute AdamW-full timestamp-sum-squared raw",
            args.endpoint_pollute_namespace,
            "score",
            "identity",
        ),
    )

    rankings = {}
    endpoints = {}
    rows: list[dict[str, object]] = []
    for query in queries:
        endpoints[query.query_id] = display_image(
            np.load(endpoint_path(sample_root, checkpoint_stem, query), allow_pickle=False)
        )
        for method in methods:
            values = top_scores(
                score_dir(result_root, args.train_seed, query, method),
                method,
                args.top_k,
                len(images),
            )
            rankings[(query.query_id, method.key)] = values
            indices, stored, ranking = values
            for rank, (index, raw_value, rank_value) in enumerate(
                zip(indices, stored, ranking, strict=True), start=1
            ):
                idx = int(index)
                rows.append(
                    {
                        "query_id": query.query_id,
                        "prompt": query.prompt,
                        "initial_seed": query.initial_seed,
                        "method": method.key,
                        "rank": rank,
                        "dataset_index": idx,
                        "source_filename": filenames[idx],
                        "stored_score": float(raw_value),
                        "ranking_score": float(rank_value),
                        "positive_attributes": ",".join(
                            label_names[i] for i in np.flatnonzero(labels[idx] > 0)
                        ),
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "top10_rankings.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {csv_path}", flush=True)

    for query in queries:
        figure, axes = plt.subplots(
            2,
            args.top_k + 1,
            figsize=(2.05 * (args.top_k + 1), 4.7),
            squeeze=False,
        )
        for row_index, method in enumerate(methods):
            draw_row(
                axes[row_index],
                query=query,
                query_image=endpoints[query.query_id],
                method=method,
                top_indices=rankings[(query.query_id, method.key)][0],
                stored_scores=rankings[(query.query_id, method.key)][1],
                ranking_scores=rankings[(query.query_id, method.key)][2],
                images=images,
                labels=labels,
                label_names=label_names,
                filenames=filenames,
                show_query=row_index == 1,
            )
        axes[0, 0].set_title("Generated query", fontsize=9, fontweight="bold")
        for rank in range(1, args.top_k + 1):
            axes[0, rank].set_title(f"Top {rank}", fontsize=9, fontweight="bold")
        figure.suptitle(f"CelebA attribution comparison · Q{query.query_id}", fontsize=12)
        figure.subplots_adjust(left=0.17, right=0.995, top=0.91, bottom=0.08, hspace=0.48, wspace=0.06)
        save_figure(figure, output_dir / f"q{query.query_id:03d}_top{args.top_k}.png", args.dpi)

    figure, axes = plt.subplots(
        2 * len(queries),
        args.top_k + 1,
        figsize=(2.05 * (args.top_k + 1), 3.7 * len(queries)),
        squeeze=False,
    )
    for query_index, query in enumerate(queries):
        for method_index, method in enumerate(methods):
            row_index = 2 * query_index + method_index
            draw_row(
                axes[row_index],
                query=query,
                query_image=endpoints[query.query_id],
                method=method,
                top_indices=rankings[(query.query_id, method.key)][0],
                stored_scores=rankings[(query.query_id, method.key)][1],
                ranking_scores=rankings[(query.query_id, method.key)][2],
                images=images,
                labels=labels,
                label_names=label_names,
                filenames=filenames,
                show_query=method_index == 1,
            )
    axes[0, 0].set_title("Generated query", fontsize=9, fontweight="bold")
    for rank in range(1, args.top_k + 1):
        axes[0, rank].set_title(f"Top {rank}", fontsize=9, fontweight="bold")
    figure.suptitle("CelebA top-attribution comparison", fontsize=14, fontweight="bold")
    figure.subplots_adjust(left=0.17, right=0.995, top=0.975, bottom=0.01, hspace=0.60, wspace=0.06)
    save_figure(figure, output_dir / f"q{selected_ids[0]:03d}_q{selected_ids[-1]:03d}_top{args.top_k}_overview.png", args.dpi)
    print(f"[done] queries={selected_ids} top_k={args.top_k} output={output_dir}", flush=True)


if __name__ == "__main__":
    main()
