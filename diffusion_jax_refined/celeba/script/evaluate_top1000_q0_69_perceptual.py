#!/usr/bin/env python3
"""Compute CLIP, LPIPS, and DINO distances for saved CelebA endpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import lpips
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor


METHODS = (
    "endpoint_pollute_adamw_timestamp_both_l2_full10k_pos",
    "endpoint_tracin_adamw_linear_raw_full10k_neg",
    "endpoint_tracin_adamw_linear_both_l2_full10k_neg",
    "retrac_adamw_query_train_l2_full10k_neg",
    "random_full10k_seeded",
)
METHOD_LABELS = {
    METHODS[0]: "PINT-BOTH-L2",
    METHODS[1]: "TRACIN-RAW",
    METHODS[2]: "TRACIN-BOTH-L2",
    METHODS[3]: "RETRAC-BOTH-L2",
    METHODS[4]: "RANDOM",
}


def parse_ids(text: str) -> list[int]:
    values = [int(value) for value in text.replace(",", " ").split() if value]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one query id")
    return values


def batched(values: Sequence[Any], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def find_query_dir(root: Path, query_id: int) -> Path:
    matches = sorted(root.glob(f"query_{query_id:03d}_seed_*"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Q{query_id}: expected one output directory, found {len(matches)}")
    return matches[0]


def load_images(root: Path, query_ids: Sequence[int]):
    images: list[Image.Image] = []
    positions: dict[tuple[int, str], int] = {}
    query_dirs: dict[int, Path] = {}
    for query_id in query_ids:
        query_dir = find_query_dir(root, query_id)
        query_dirs[query_id] = query_dir
        names = {"original": "original_endpoint.png"}
        names.update({method: f"{method}_endpoint.png" for method in METHODS})
        for method, filename in names.items():
            path = query_dir / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            positions[(query_id, method)] = len(images)
            with Image.open(path) as source:
                images.append(source.convert("RGB"))
    return images, positions, query_dirs


def normalized_features(values: list[torch.Tensor]) -> np.ndarray:
    result = torch.cat(values, dim=0).float()
    return torch.nn.functional.normalize(result, dim=-1).cpu().numpy()


def clip_features(images, model_name: str, batch_size: int, device: torch.device) -> np.ndarray:
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).eval().to(device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    if device.type == "cuda":
        model = model.half()
    output = []
    for batch in batched(images, batch_size):
        pixels = processor(images=list(batch), return_tensors="pt")["pixel_values"]
        with torch.inference_mode():
            output.append(model.get_image_features(pixel_values=pixels.to(device, dtype=dtype)))
    return normalized_features(output)


def dino_features(
    images, repository: str, model_name: str, batch_size: int, device: torch.device
) -> np.ndarray:
    model = torch.hub.load(repository, model_name).eval().to(device)
    preprocess = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    output = []
    for batch in batched(images, batch_size):
        values = torch.stack([preprocess(image) for image in batch]).to(device)
        with torch.inference_mode():
            output.append(model(values))
    return normalized_features(output)


def lpips_distances(images, positions, query_ids, network: str, device: torch.device):
    model = lpips.LPIPS(net=network).eval().to(device)
    to_tensor = transforms.ToTensor()
    output = {}
    with torch.inference_mode():
        for query_id in query_ids:
            reference = to_tensor(images[positions[(query_id, "original")]])
            reference = (reference * 2.0 - 1.0).unsqueeze(0).to(device)
            candidates = torch.stack(
                [
                    to_tensor(images[positions[(query_id, method)]]) * 2.0 - 1.0
                    for method in METHODS
                ]
            ).to(device)
            values = model(reference.expand_as(candidates), candidates).flatten().cpu().numpy()
            for method, value in zip(METHODS, values):
                output[(query_id, method)] = float(value)
    return output


def cosine_distance(features: np.ndarray, left: int, right: int) -> float:
    return float(1.0 - np.dot(features[left], features[right]))


def read_pixel_rows(query_dirs: dict[int, Path]) -> list[dict[str, Any]]:
    rows = []
    for query_id, query_dir in query_dirs.items():
        with (query_dir / "pixel_trajectory_metrics.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if int(row["query_id"]) != query_id:
                    raise ValueError(f"query mismatch in {query_dir}")
                rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "CELEBA Q0-Q69 FULL-10K TOP-1000 REMOVAL — FIVE METRIC FAMILIES",
        "Larger values mean a larger change from the full-data model.",
        "",
        (
            f"{'METHOD':<20} {'END-MSE':>11} {'END-RMSE':>11} "
            f"{'TRAJ-MSE':>11} {'TRAJ-RMSE':>11} {'CLIP-D':>11} "
            f"{'LPIPS':>11} {'DINO-D':>11}"
        ),
        "-" * 105,
    ]
    numeric = (
        "endpoint_mse",
        "endpoint_rmse",
        "trajectory_mse",
        "trajectory_rmse",
        "clip_distance",
        "lpips",
        "dino_distance",
    )
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        means = [float(np.mean([float(row[key]) for row in selected])) for key in numeric]
        lines.append(
            f"{METHOD_LABELS[method]:<20}" + "".join(f" {value:>10.6f}" for value in means)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--query-ids", type=parse_ids, default=list(range(70)))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--dino-repository", default="facebookresearch/dino:main")
    parser.add_argument("--dino-model", default="dino_vits16")
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="alex")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("perceptual evaluation requires a CUDA GPU")
    device = torch.device("cuda")
    images, positions, query_dirs = load_images(args.root, args.query_ids)
    print(f"[load] images={len(images)} queries={len(args.query_ids)}", flush=True)
    clip = clip_features(images, args.clip_model, args.batch_size, device)
    print("[metric] CLIP complete", flush=True)
    dino = dino_features(
        images, args.dino_repository, args.dino_model, args.batch_size, device
    )
    print("[metric] DINO complete", flush=True)
    perceptual = lpips_distances(
        images, positions, args.query_ids, args.lpips_net, device
    )
    print("[metric] LPIPS complete", flush=True)

    pixel_rows = read_pixel_rows(query_dirs)
    by_key = {(int(row["query_id"]), row["method"]): row for row in pixel_rows}
    rows = []
    for query_id in args.query_ids:
        reference_position = positions[(query_id, "original")]
        for method in METHODS:
            row = dict(by_key[(query_id, method)])
            candidate_position = positions[(query_id, method)]
            row.update(
                {
                    "clip_distance": cosine_distance(
                        clip, reference_position, candidate_position
                    ),
                    "lpips": perceptual[(query_id, method)],
                    "dino_distance": cosine_distance(
                        dino, reference_position, candidate_position
                    ),
                }
            )
            rows.append(row)

    write_csv(args.root / "all_metrics_per_query.csv", rows)
    write_report(args.root / "all_metrics_summary.txt", rows)
    (args.root / "all_metrics_manifest.json").write_text(
        json.dumps(
            {
                "query_ids": args.query_ids,
                "methods": list(METHODS),
                "clip_model": args.clip_model,
                "lpips_network": args.lpips_net,
                "dino_repository": args.dino_repository,
                "dino_model": args.dino_model,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[done] {args.root / 'all_metrics_summary.txt'}", flush=True)


if __name__ == "__main__":
    main()
