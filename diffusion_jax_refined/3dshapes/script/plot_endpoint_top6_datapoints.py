#!/usr/bin/env python3
"""Plot each query endpoint beside its six highest-scoring training images.

The default invocation produces two 10-row figures:

* checkpoint-own trajectory, 100-timestamp next-checkpoint objective,
  AdamW FOUR_RESIDUAL query/train-L2 score;
* factorized DAS MC4 endpoint score at lambda=100.

Score indices are always read from the score artifact.  They are dataset row
indices, not positions within ``scores.npy`` or within the attribution subset.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SHAPES_ROOT = Path(__file__).resolve().parents[1]
REFINE_ROOT = SHAPES_ROOT.parent

SHAPE_NAMES = ("cube", "cylinder", "sphere", "capsule")
QUERY_LABELS = (
    tuple(f"shape_{name}" for name in SHAPE_NAMES)
    + tuple(f"object_hue_{i}" for i in range(10))
    + tuple(f"wall_hue_{i}" for i in range(10))
    + tuple(f"floor_hue_{i}" for i in range(10))
)


@dataclass(frozen=True)
class Query:
    query_id: int
    prompt: str
    initial_seed: int


def prompt_tag(prompt: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(prompt).replace(",", "__"))
    return re.sub(r"_+", "_", text).strip("_")[:160] or "empty"


def damping_tag(value: float) -> str:
    return ("%g" % float(value)).replace("+", "_").replace("-", "neg_").replace(".", "p") or "0"


def default_queries() -> list[Query]:
    records = []
    for seed in range(10):
        labels = np.random.default_rng(seed).choice(QUERY_LABELS, size=4, replace=False)
        records.append(Query(seed, ",".join(str(x) for x in labels), seed))
    return records


def load_queries(path: Path) -> list[Query]:
    if not path.is_file():
        print(f"[queries] {path} not found; regenerating the deterministic seed-0..9 queries")
        return default_queries()
    payload = json.loads(path.read_text())
    records = payload["queries"]
    if len(records) < 10:
        raise ValueError(f"{path} contains {len(records)} queries; expected at least 10")
    return [
        Query(index, str(record["prompt"]), int(record["initial_seed"]))
        for index, record in enumerate(records[:10])
    ]


def load_dataset_images(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"3D Shapes dataset not found: {path}")
    with np.load(path, allow_pickle=False) as payload:
        if "images" not in payload:
            raise KeyError(f"{path} does not contain an images array")
        images = np.asarray(payload["images"])
    if images.ndim != 4 or images.shape[-1] not in (1, 3, 4):
        raise ValueError(f"expected NHWC dataset images, got {images.shape} from {path}")
    return images


def display_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError(f"expected one endpoint image, got {image.shape}")
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"expected HWC image, got {image.shape}")
    if image.dtype == np.uint8:
        return image
    image = image.astype(np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise ValueError("image contains no finite pixels")
    # decoded_final.npy is [0, 1].  This fallback also supports a model-space
    # endpoint in [-1, 1] if an older run omitted decoded_final.npy.
    if float(finite.min()) < -0.01:
        image = (image + 1.0) / 2.0
    elif float(finite.max()) > 1.01:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0)


def endpoint_path(
    sample_root: Path,
    query: Query,
    checkpoint_stem: str,
) -> Path:
    query_root = sample_root / "cifar" / f"prompt_{prompt_tag(query.prompt)}"
    exact = (
        query_root
        / f"model_prompted_solo__ckpt_{checkpoint_stem}"
        / f"seed_{query.initial_seed:06d}"
        / "decoded_final.npy"
    )
    if exact.is_file():
        return exact
    candidates = sorted(query_root.glob(f"model_*/*seed_{query.initial_seed:06d}/decoded_final.npy"))
    if len(candidates) == 1:
        print(f"[endpoint] exact checkpoint path absent; using {candidates[0]}")
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"endpoint not found for Q{query.query_id}: expected {exact}")
    raise FileNotFoundError(
        f"endpoint is ambiguous for Q{query.query_id}; expected {exact}, candidates={candidates}"
    )


def score_root(result_root: Path, train_seed: int, query: Query) -> Path:
    return (
        result_root
        / "attribution_score"
        / "prompted_solo"
        / f"train_seed_{train_seed}"
        / f"query_{prompt_tag(query.prompt)}"
        / f"initial_seed_{query.initial_seed}"
    )


def load_score_indices(directory: Path, score_count: int) -> np.ndarray:
    npy_path = directory / "score_indices.npy"
    if npy_path.is_file():
        indices = np.asarray(np.load(npy_path), dtype=np.int64).reshape(-1)
    else:
        json_path = directory / "score_indices.json"
        if not json_path.is_file():
            raise FileNotFoundError(
                f"missing score_indices.npy and score_indices.json in {directory}"
            )
        payload = json.loads(json_path.read_text())
        for key in ("score_indices", "picked_indices", "indices"):
            if key in payload:
                indices = np.asarray(payload[key], dtype=np.int64).reshape(-1)
                break
        else:
            raise KeyError(f"cannot find score indices in {json_path}")
    if len(indices) != score_count:
        raise ValueError(
            f"score/index length mismatch in {directory}: {score_count} scores vs {len(indices)} indices"
        )
    return indices


def load_top_scores(
    directory: Path,
    top_k: int,
    dataset_size: int,
    ranking_sign: int,
) -> tuple[np.ndarray, np.ndarray]:
    score_path = directory / "scores.npy"
    if not score_path.is_file():
        raise FileNotFoundError(str(score_path))
    scores = np.asarray(np.load(score_path), dtype=np.float64).reshape(-1)
    indices = load_score_indices(directory, len(scores))
    if len(scores) < top_k:
        raise ValueError(f"{score_path} has only {len(scores)} scores; top_k={top_k}")
    if np.any(indices < 0) or np.any(indices >= dataset_size):
        bad = indices[(indices < 0) | (indices >= dataset_size)][:10]
        raise IndexError(f"dataset indices out of bounds in {directory}: {bad.tolist()}")
    if ranking_sign not in (-1, 1):
        raise ValueError(f"ranking_sign must be -1 or +1, got {ranking_sign}")
    ranking_values = np.where(np.isfinite(scores), ranking_sign * scores, -np.inf)
    order = np.argsort(-ranking_values, kind="stable")[:top_k]
    if np.any(~np.isfinite(ranking_values[order])):
        raise ValueError(f"{score_path} does not contain {top_k} finite scores")
    return indices[order], scores[order]


def short_prompt(prompt: str) -> str:
    return "\n".join(prompt.split(","))


def plot_method(
    *,
    queries: list[Query],
    dataset_images: np.ndarray,
    endpoint_paths: dict[int, Path],
    score_dirs: dict[int, Path],
    top_k: int,
    title: str,
    output: Path,
    dpi: int,
    ranking_sign: int,
) -> None:
    figure, axes = plt.subplots(
        len(queries),
        top_k + 1,
        figsize=(2.15 * (top_k + 1), 2.05 * len(queries)),
        gridspec_kw={"width_ratios": [1.12] + [1.0] * top_k},
        squeeze=False,
    )
    for row, query in enumerate(queries):
        endpoint = display_image(np.load(endpoint_paths[query.query_id], allow_pickle=False))
        axes[row, 0].imshow(endpoint)
        axes[row, 0].set_ylabel(
            f"Q{query.query_id}\n{short_prompt(query.prompt)}",
            rotation=0,
            ha="right",
            va="center",
            fontsize=7,
            labelpad=7,
        )
        top_indices, top_scores = load_top_scores(
            score_dirs[query.query_id], top_k, len(dataset_images), ranking_sign
        )
        for rank, (index, score) in enumerate(zip(top_indices, top_scores), start=1):
            axis = axes[row, rank]
            axis.imshow(display_image(dataset_images[int(index)]))
            axis.set_xlabel(f"#{rank}  idx {int(index)}\n{float(score):+.3e}", fontsize=7)
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)

    axes[0, 0].set_title("Endpoint", fontsize=10, fontweight="bold")
    for rank in range(1, top_k + 1):
        axes[0, rank].set_title(f"Top {rank}", fontsize=10, fontweight="bold")
    figure.suptitle(title, fontsize=14, fontweight="bold", y=0.997)
    figure.subplots_adjust(left=0.19, right=0.995, top=0.973, bottom=0.02, wspace=0.08, hspace=0.30)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"[saved] {output}")


def main() -> None:
    default_data_root = Path(
        os.environ.get(
            "THREEDSHAPES_DATA_ROOT",
            str(REFINE_ROOT / "dataset" / "3dshapes" / "20000"),
        )
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--query-file", type=Path, default=SHAPES_ROOT / "queries_seed_0_9.json")
    parser.add_argument("--dataset", type=Path, default=default_data_root / "dataset.npz")
    parser.add_argument("--sample-root", type=Path, default=None)
    parser.add_argument(
        "--traj-namespace",
        default="loss_direction_original_f_checkpoint_own_trajectory_100t",
    )
    parser.add_argument(
        "--das-namespace",
        default="factorized_mc4_original100x1",
    )
    parser.add_argument("--das-lambda", type=float, default=100.0)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    queries = load_queries(args.query_file)
    dataset_images = load_dataset_images(args.dataset)
    result_root = SHAPES_ROOT / "result" / args.experiment
    sample_root = args.sample_root or result_root / "sample_ddim_eta0_1000"
    output_dir = args.output_dir or result_root / "eval" / "endpoint_top6_panels"
    checkpoint_stem = f"seed_{args.train_seed}_epoch_{args.epochs:04d}"

    endpoints = {
        query.query_id: endpoint_path(sample_root, query, checkpoint_stem)
        for query in queries
    }
    traj_dirs = {
        query.query_id: score_root(result_root, args.train_seed, query)
        / f"traj_tracin_{args.traj_namespace}"
        / "score"
        for query in queries
    }
    das_lambda_dir = f"lambda_{damping_tag(args.das_lambda)}"
    das_dirs = {
        query.query_id: score_root(result_root, args.train_seed, query)
        / f"das_{args.das_namespace}"
        / "score"
        / das_lambda_dir
        for query in queries
    }

    plot_method(
        queries=queries,
        dataset_images=dataset_images,
        endpoint_paths=endpoints,
        score_dirs=traj_dirs,
        top_k=args.top_k,
        title="Own trajectory 100t · next checkpoint · reversed score · AdamW FOUR_RESIDUAL",
        output=output_dir / "own_trajectory_next_raw_endpoint_top6.png",
        dpi=args.dpi,
        ranking_sign=-1,
    )
    plot_method(
        queries=queries,
        dataset_images=dataset_images,
        endpoint_paths=endpoints,
        score_dirs=das_dirs,
        top_k=args.top_k,
        title=f"DAS factorized MC4 · endpoint · lambda={args.das_lambda:g}",
        output=output_dir / f"das_mc4_lambda_{damping_tag(args.das_lambda)}_endpoint_top6.png",
        dpi=args.dpi,
        ranking_sign=1,
    )


if __name__ == "__main__":
    main()
