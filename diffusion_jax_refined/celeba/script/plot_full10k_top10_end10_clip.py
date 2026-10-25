#!/usr/bin/env python3
"""Sanity-check CelebA full-10k removal Top-10 versus End-10 with CLIP."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from plot_endpoint_full10k_top10_all_variants import (
    Bank,
    banks_for_transform,
    score_directory,
)
from plot_endpoint_top10_comparison import (
    REFINE_ROOT,
    display_image,
    endpoint_path,
    load_dataset,
    load_queries,
    save_figure,
    short_attributes,
)


CELEBA_ROOT = Path(__file__).resolve().parents[1]
METHOD_SPECS = (
    (
        "endpoint_pollute_adamw_timestamp_both_l2_full10k_pos",
        "Pollute Both-L2",
        "Pollute",
        "both-L2",
    ),
    (
        "endpoint_tracin_adamw_linear_raw_full10k_neg",
        "TracIn raw",
        "TracIn",
        "raw",
    ),
    (
        "endpoint_tracin_adamw_linear_both_l2_full10k_neg",
        "TracIn Both-L2",
        "TracIn",
        "both-L2",
    ),
    (
        "retrac_adamw_query_train_l2_full10k_neg",
        "ReTrac Both-L2",
        "ReTrac",
        "both-L2",
    ),
)


@dataclass(frozen=True)
class Method:
    removal_key: str
    label: str
    bank: Bank


def parse_ints(text: str) -> list[int]:
    values = [int(value) for value in text.replace(",", " ").split() if value]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one query ID")
    return values


def methods() -> list[Method]:
    banks = banks_for_transform("adamw_full")
    output = []
    for removal_key, label, bank_method, variant in METHOD_SPECS:
        matches = [
            bank
            for bank in banks
            if bank.method == bank_method and bank.variant == variant
        ]
        if len(matches) != 1:
            raise AssertionError(
                f"expected one score bank for {bank_method}/{variant}, found {len(matches)}"
            )
        output.append(Method(removal_key, label, matches[0]))
    return output


def load_full_ranking(
    directory: Path, *, ranking_sign: int, dataset_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stored = np.asarray(np.load(directory / "scores.npy"), dtype=np.float64).reshape(-1)
    indices = np.asarray(
        np.load(directory / "score_indices.npy"), dtype=np.int64
    ).reshape(-1)
    expected = np.arange(dataset_size, dtype=np.int64)
    if stored.shape != indices.shape:
        raise ValueError(f"score/index shape mismatch under {directory}")
    if len(stored) != dataset_size or not np.array_equal(indices, expected):
        raise ValueError(
            f"score bank is not the complete ordered {dataset_size}-example dataset: "
            f"{directory}"
        )
    ranking = ranking_sign * stored
    if not np.isfinite(ranking).all():
        raise ValueError(f"non-finite ranking score under {directory}")
    order = np.argsort(-ranking, kind="stable")
    return indices, stored, ranking, order


def validate_manifest(
    path: Path, *, expected: np.ndarray, method: Method, query_id: int
) -> str:
    if not path.is_file():
        return "missing"
    payload = json.loads(path.read_text())
    actual = np.asarray(payload["removed_dataset_indices"], dtype=np.int64)
    if not np.array_equal(actual, expected):
        if actual.shape != expected.shape:
            detail = f"shape actual={actual.shape} expected={expected.shape}"
        else:
            offset = int(np.flatnonzero(actual != expected)[0])
            detail = (
                f"first mismatch at {offset}: actual={actual[offset]} "
                f"expected={expected[offset]}"
            )
        raise ValueError(
            f"Top-1000 manifest mismatch for {method.removal_key} Q{query_id}: {detail}"
        )
    if int(payload.get("ranking_sign")) != method.bank.ranking_sign:
        raise ValueError(f"ranking sign mismatch in {path}")
    return "exact_match"


def to_pil(value: np.ndarray) -> Image.Image:
    image = display_image(value)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        image = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(image, mode="RGB")


def clip_features(
    arrays: list[np.ndarray], *, model_name: str, device_name: str, batch_size: int
) -> np.ndarray:
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as error:
        raise RuntimeError(
            "CLIP requires torch and transformers; run this script in the "
            "sd_artbench Conda environment"
        ) from error

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(device_name)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()
    output = []
    for start in range(0, len(arrays), batch_size):
        batch = [to_pil(value) for value in arrays[start : start + batch_size]]
        try:
            inputs = processor(images=batch, return_tensors="pt")
            pixels = inputs["pixel_values"].to(device=device, dtype=dtype)
            with torch.inference_mode():
                features = model.get_image_features(pixel_values=pixels)
                features = torch.nn.functional.normalize(features.float(), dim=-1)
            output.append(features.cpu().numpy())
        finally:
            for image in batch:
                image.close()
    return np.concatenate(output, axis=0).astype(np.float32, copy=False)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    data_root = Path(
        os.environ.get(
            "CELEBA_ATTR40_DATA_ROOT",
            str(REFINE_ROOT / "dataset" / "celeba" / "10000_attr40"),
        )
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--query-ids", type=parse_ints, default=list(range(10, 20)))
    parser.add_argument("--experiment", default="attr40_experiment1")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--dataset", type=Path, default=data_root / "dataset.npz")
    parser.add_argument("--sample-root", type=Path)
    parser.add_argument("--removal-k", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--end-k", type=int, default=10)
    parser.add_argument(
        "--removal-root",
        type=Path,
        help="Defaults to result/<experiment>/endpoint_top1000_removal_q0_69",
    )
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip-batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=130)
    args = parser.parse_args()
    if min(args.removal_k, args.top_k, args.end_k, args.clip_batch_size) < 1:
        parser.error("K values and CLIP batch size must be positive")
    if len(args.query_ids) != len(set(args.query_ids)):
        parser.error("query IDs must be unique")

    selected_methods = methods()
    queries = load_queries(args.query_file, args.query_ids)
    images, labels, label_names, filenames = load_dataset(args.dataset)
    if len(images) != 10_000:
        raise ValueError(f"expected 10,000 CelebA training images, found {len(images)}")
    result_root = CELEBA_ROOT / "result" / args.experiment
    sample_root = args.sample_root or result_root / "sample_ddim_eta0_50"
    removal_root = args.removal_root or (
        result_root / "endpoint_top1000_removal_q0_69"
    )
    output_dir = args.output_dir or (
        result_root / "eval" / "full10k_top10_vs_end10_clip"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_stem = f"seed_{args.train_seed}_epoch_{args.epochs:04d}"

    query_images: dict[int, np.ndarray] = {}
    selections: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    validation_rows: list[dict[str, Any]] = []
    for query in queries:
        query_images[query.query_id] = display_image(
            np.load(
                endpoint_path(sample_root, checkpoint_stem, query),
                allow_pickle=False,
            )
        )
        for method in selected_methods:
            directory = score_directory(result_root, args.train_seed, query, method.bank)
            indices, stored, ranking, order = load_full_ranking(
                directory,
                ranking_sign=method.bank.ranking_sign,
                dataset_size=len(images),
            )
            expected_removed = indices[order[: args.removal_k]]
            manifest_path = (
                removal_root
                / method.removal_key
                / f"query_{query.query_id:03d}_seed_{query.initial_seed:03d}"
                / "removal_manifest.json"
            )
            status = validate_manifest(
                manifest_path,
                expected=expected_removed,
                method=method,
                query_id=query.query_id,
            )
            validation_rows.append(
                {
                    "query_id": query.query_id,
                    "method": method.removal_key,
                    "manifest": str(manifest_path),
                    "status": status,
                }
            )
            print(
                f"[top1000-check] Q{query.query_id} method={method.removal_key} "
                f"status={status}",
                flush=True,
            )
            for tail, selected_positions in (
                ("top", order[: args.top_k]),
                ("end", order[-args.end_k :]),
            ):
                rows = []
                for position in selected_positions:
                    position = int(position)
                    dataset_index = int(indices[position])
                    rows.append(
                        {
                            "query_id": query.query_id,
                            "prompt": query.prompt,
                            "initial_seed": query.initial_seed,
                            "method": method.removal_key,
                            "method_label": method.label,
                            "tail": tail,
                            "ranking_direction": (
                                "descending" if method.bank.ranking_sign == 1 else "ascending"
                            ),
                            "overall_rank": int(np.flatnonzero(order == position)[0]) + 1,
                            "dataset_index": dataset_index,
                            "source_filename": filenames[dataset_index],
                            "stored_score": float(stored[position]),
                            "signed_ranking_score": float(ranking[position]),
                            "positive_attributes": ",".join(
                                label_names[index]
                                for index in np.flatnonzero(labels[dataset_index] > 0)
                            ),
                            "top1000_manifest_status": status,
                        }
                    )
                selections[(query.query_id, method.removal_key, tail)] = rows

    dataset_indices = sorted(
        {
            int(row["dataset_index"])
            for rows in selections.values()
            for row in rows
        }
    )
    arrays = [query_images[query.query_id] for query in queries]
    arrays.extend(images[index] for index in dataset_indices)
    features = clip_features(
        arrays,
        model_name=args.clip_model,
        device_name=args.device,
        batch_size=args.clip_batch_size,
    )
    query_feature = {
        query.query_id: features[position] for position, query in enumerate(queries)
    }
    dataset_feature = {
        index: features[len(queries) + position]
        for position, index in enumerate(dataset_indices)
    }

    ranking_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for query in queries:
        for method in selected_methods:
            tail_summaries = {}
            for tail in ("top", "end"):
                rows = selections[(query.query_id, method.removal_key, tail)]
                for row in rows:
                    row["clip_similarity"] = float(
                        np.dot(
                            query_feature[query.query_id],
                            dataset_feature[int(row["dataset_index"])],
                        )
                    )
                    ranking_rows.append(row)
                values = np.asarray(
                    [row["clip_similarity"] for row in rows], dtype=np.float64
                )
                summary = {
                    "query_id": query.query_id,
                    "method": method.removal_key,
                    "tail": tail,
                    "count": len(rows),
                    "clip_mean": float(values.mean()),
                    "clip_median": float(np.median(values)),
                    "clip_min": float(values.min()),
                    "clip_max": float(values.max()),
                    "top1000_manifest_status": rows[0]["top1000_manifest_status"],
                }
                summary_rows.append(summary)
                tail_summaries[tail] = summary
            comparison = {
                "query_id": query.query_id,
                "method": method.removal_key,
                "top10_clip_mean": tail_summaries["top"]["clip_mean"],
                "end10_clip_mean": tail_summaries["end"]["clip_mean"],
                "top_minus_end_clip_mean": (
                    float(tail_summaries["top"]["clip_mean"])
                    - float(tail_summaries["end"]["clip_mean"])
                ),
                "top10_clip_median": tail_summaries["top"]["clip_median"],
                "end10_clip_median": tail_summaries["end"]["clip_median"],
                "top1000_manifest_status": tail_summaries["top"][
                    "top1000_manifest_status"
                ],
            }
            comparison_rows.append(comparison)
            print(
                f"[clip] Q{query.query_id} method={method.removal_key} "
                f"top10_mean={comparison['top10_clip_mean']:.4f} "
                f"end10_mean={comparison['end10_clip_mean']:.4f} "
                f"delta={comparison['top_minus_end_clip_mean']:+.4f}",
                flush=True,
            )

    for query in queries:
        figure, axes = plt.subplots(
            2 * len(selected_methods),
            max(args.top_k, args.end_k) + 1,
            figsize=(2.0 * (max(args.top_k, args.end_k) + 1), 2.1 * 2 * len(selected_methods)),
            squeeze=False,
        )
        row_index = 0
        for method in selected_methods:
            for tail in ("top", "end"):
                rows = selections[(query.query_id, method.removal_key, tail)]
                axes[row_index, 0].imshow(query_images[query.query_id])
                axes[row_index, 0].set_ylabel(
                    f"{method.label}\n{tail.upper()}-{len(rows)}",
                    rotation=0,
                    ha="right",
                    va="center",
                    fontsize=7,
                )
                for column, row in enumerate(rows, start=1):
                    index = int(row["dataset_index"])
                    axes[row_index, column].imshow(display_image(images[index]))
                    axes[row_index, column].set_xlabel(
                        f"rank={row['overall_rank']} idx={index}\n"
                        f"attr={row['stored_score']:+.2e} clip={row['clip_similarity']:.3f}\n"
                        f"{short_attributes(labels[index], label_names)}",
                        fontsize=5.2,
                    )
                for axis in axes[row_index]:
                    axis.set_xticks([])
                    axis.set_yticks([])
                    for spine in axis.spines.values():
                        spine.set_visible(False)
                row_index += 1
        axes[0, 0].set_title("Generated query", fontsize=9, fontweight="bold")
        for column in range(1, max(args.top_k, args.end_k) + 1):
            axes[0, column].set_title(f"slot {column}", fontsize=8, fontweight="bold")
        figure.suptitle(
            f"CelebA full-10k AdamW removal ranking · Q{query.query_id} · "
            f"seed {query.initial_seed}\n{query.prompt}",
            fontsize=12,
        )
        figure.subplots_adjust(
            left=0.16, right=0.995, top=0.94, bottom=0.03, hspace=0.55, wspace=0.06
        )
        save_figure(
            figure,
            output_dir / f"q{query.query_id:03d}_top10_vs_end10_clip.png",
            args.dpi,
        )

    write_csv(output_dir / "selected_rankings.csv", ranking_rows)
    write_csv(output_dir / "clip_summary.csv", summary_rows)
    write_csv(output_dir / "clip_top_vs_end.csv", comparison_rows)
    (output_dir / "top1000_validation.json").write_text(
        json.dumps(validation_rows, indent=2) + "\n"
    )
    manifest = {
        "query_file": str(args.query_file.resolve()),
        "query_ids": args.query_ids,
        "methods": [method.removal_key for method in selected_methods],
        "candidate_count": len(images),
        "removal_k": args.removal_k,
        "top_k": args.top_k,
        "end_k": args.end_k,
        "clip_model": args.clip_model,
        "score_transform": "adamw_full",
        "stable_sort": True,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"[done] queries={len(queries)} selected={len(ranking_rows)} output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
