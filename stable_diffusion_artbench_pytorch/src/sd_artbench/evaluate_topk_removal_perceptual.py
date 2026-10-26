"""Evaluate saved ArtBench Top-K removal images with perceptual metrics."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any, Sequence

import lpips
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor

from .config import ArtBenchConfig


METHODS = (
    "endpoint_pollute_query_train_l2",
    "endpoint_tracin_raw",
    "endpoint_tracin_query_train_l2",
    "retrac_query_train_l2",
)
METHOD_LABELS = {
    "endpoint_pollute_query_train_l2": "POLLUTE-BOTH-L2",
    "endpoint_tracin_raw": "TRACIN-RAW",
    "endpoint_tracin_query_train_l2": "TRACIN-BOTH-L2",
    "retrac_query_train_l2": "RETRAC-BOTH-L2",
}
PAIRS = (
    ("endpoint_pollute_query_train_l2", "endpoint_tracin_raw"),
    ("endpoint_pollute_query_train_l2", "endpoint_tracin_query_train_l2"),
    ("endpoint_pollute_query_train_l2", "retrac_query_train_l2"),
    ("endpoint_tracin_query_train_l2", "endpoint_tracin_raw"),
    ("retrac_query_train_l2", "endpoint_tracin_query_train_l2"),
    ("retrac_query_train_l2", "endpoint_tracin_raw"),
)
METRICS = ("pixel_l2", "clip_distance", "lpips", "dino_distance")


def parse_csv_ints(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return result


def parse_args() -> argparse.Namespace:
    root = ArtBenchConfig().index_root.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-root",
        type=Path,
        default=root / "outputs/topk-removal-trajectory-mse-q0-9",
    )
    parser.add_argument("--query-indices", type=parse_csv_ints, default=list(range(10)))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--dino-repository", default="facebookresearch/dino:main")
    parser.add_argument("--dino-model", default="dino_vits16")
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="alex")
    return parser.parse_args()


def image_paths(root: Path, query_indices: Sequence[int]) -> tuple[list[Path], dict[tuple[int, str], int]]:
    paths: list[Path] = []
    positions: dict[tuple[int, str], int] = {}
    for query_index in query_indices:
        query_dir = root / f"query-{query_index:03d}"
        names = {"full_data": "full-data.png"}
        names.update({method: f"{method}.png" for method in METHODS})
        for method, name in names.items():
            path = query_dir / name
            if not path.is_file():
                raise FileNotFoundError(path)
            positions[(query_index, method)] = len(paths)
            paths.append(path)
    return paths, positions


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as source:
        return source.convert("RGB")


def image_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.float32) / 255.0


def pixel_l2(reference: Image.Image, candidate: Image.Image) -> float:
    left = image_array(reference).astype(np.float64)
    right = image_array(candidate).astype(np.float64)
    if left.shape != right.shape:
        raise ValueError(f"image shape mismatch: {left.shape} vs {right.shape}")
    return float(np.sqrt(np.mean(np.square(left - right), dtype=np.float64)))


def batched(values: Sequence[Any], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def clip_features(
    images: Sequence[Image.Image],
    *,
    model_name: str,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).eval().to(device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    if device.type == "cuda":
        model = model.half()
    features = []
    for batch in batched(images, batch_size):
        inputs = processor(images=list(batch), return_tensors="pt")
        pixels = inputs["pixel_values"].to(device=device, dtype=dtype)
        with torch.inference_mode():
            value = model.get_image_features(pixel_values=pixels).float()
            value = torch.nn.functional.normalize(value, dim=-1)
        features.append(value.cpu().numpy())
    return np.concatenate(features, axis=0)


def dino_features(
    images: Sequence[Image.Image],
    *,
    repository: str,
    model_name: str,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model = torch.hub.load(repository, model_name).eval().to(device)
    preprocess = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )
    features = []
    for batch in batched(images, batch_size):
        values = torch.stack([preprocess(image) for image in batch]).to(device)
        with torch.inference_mode():
            result = model(values).float()
            result = torch.nn.functional.normalize(result, dim=-1)
        features.append(result.cpu().numpy())
    return np.concatenate(features, axis=0)


def lpips_distances(
    images: Sequence[Image.Image],
    positions: dict[tuple[int, str], int],
    query_indices: Sequence[int],
    *,
    network: str,
    device: torch.device,
) -> dict[tuple[int, str], float]:
    model = lpips.LPIPS(net=network).eval().to(device)
    to_tensor = transforms.ToTensor()
    output: dict[tuple[int, str], float] = {}
    with torch.inference_mode():
        for query_index in query_indices:
            reference = to_tensor(images[positions[(query_index, "full_data")]])
            reference = (reference * 2.0 - 1.0).unsqueeze(0).to(device)
            candidates = torch.stack(
                [
                    to_tensor(images[positions[(query_index, method)]]) * 2.0 - 1.0
                    for method in METHODS
                ]
            ).to(device)
            repeated = reference.expand(candidates.shape[0], -1, -1, -1)
            values = model(repeated, candidates).flatten().cpu().numpy()
            for method, value in zip(METHODS, values):
                output[(query_index, method)] = float(value)
    return output


def exact_label_swap(left: Sequence[float], right: Sequence[float]) -> dict[str, float | int]:
    differences = [float(a) - float(b) for a, b in zip(left, right)]
    observed = sum(differences) / len(differences)
    null = [
        sum(sign * value for sign, value in zip(signs, differences)) / len(differences)
        for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
    ]
    tolerance = 1e-15
    return {
        "mean_difference": observed,
        "wins": sum(value > 0.0 for value in differences),
        "count": len(differences),
        "p_one_sided": sum(value >= observed - tolerance for value in null) / len(null),
        "p_two_sided": sum(abs(value) >= abs(observed) - tolerance for value in null) / len(null),
    }


def cosine_distance(features: np.ndarray, left: int, right: int) -> float:
    return float(1.0 - np.dot(features[left], features[right]))


def write_report(path: Path, rows: list[dict[str, Any]], tests: list[dict[str, Any]]) -> None:
    lines = [
        "ARTBENCH Q0-Q9 TOP-1000 REMOVAL — PERCEPTUAL IMAGE DIFFERENCE",
        "Larger Pixel-L2, CLIP-distance, LPIPS and DINO-distance is better.",
        "",
        f"{'QUERY':<7} {'METHOD':<22} {'PIXEL-L2':>11} {'CLIP-DIST':>11} {'LPIPS':>11} {'DINO-DIST':>11}",
        "-" * 82,
    ]
    for row in rows:
        lines.append(
            f"Q{row['query_index']:<6d} {row['method_label']:<22} "
            f"{row['pixel_l2']:>11.6f} {row['clip_distance']:>11.6f} "
            f"{row['lpips']:>11.6f} {row['dino_distance']:>11.6f}"
        )
    lines.extend(["", "METHOD SUMMARY", "-" * 82])
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        lines.append(
            f"{METHOD_LABELS[method]:<22} "
            + " ".join(
                f"{metric.upper()}={np.mean([row[metric] for row in selected]):.6f}"
                for metric in METRICS
            )
        )
    lines.extend(["", "PAIRED EXACT LABEL-SWAP (LEFT - RIGHT)", "-" * 82])
    for result in tests:
        lines.append(
            f"{result['metric'].upper():<14} {result['left_label']} - {result['right_label']}: "
            f"mean={result['mean_difference']:+.8f} wins={result['wins']}/{result['count']} "
            f"p1={result['p_one_sided']:.6f} p2={result['p_two_sided']:.6f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("perceptual evaluation requires a CUDA GPU")
    device = torch.device("cuda")
    paths, positions = image_paths(args.image_root, args.query_indices)
    images = [load_rgb(path) for path in paths]
    print(f"[load] images={len(images)} queries={len(args.query_indices)}", flush=True)
    clip = clip_features(
        images, model_name=args.clip_model, batch_size=args.batch_size, device=device
    )
    print("[metric] CLIP complete", flush=True)
    dino = dino_features(
        images,
        repository=args.dino_repository,
        model_name=args.dino_model,
        batch_size=args.batch_size,
        device=device,
    )
    print("[metric] DINO complete", flush=True)
    perceptual = lpips_distances(
        images,
        positions,
        args.query_indices,
        network=args.lpips_net,
        device=device,
    )
    print("[metric] LPIPS complete", flush=True)

    rows: list[dict[str, Any]] = []
    for query_index in args.query_indices:
        reference_position = positions[(query_index, "full_data")]
        reference = images[reference_position]
        for method in METHODS:
            candidate_position = positions[(query_index, method)]
            rows.append(
                {
                    "query_index": query_index,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "pixel_l2": pixel_l2(reference, images[candidate_position]),
                    "clip_distance": cosine_distance(clip, reference_position, candidate_position),
                    "lpips": perceptual[(query_index, method)],
                    "dino_distance": cosine_distance(dino, reference_position, candidate_position),
                }
            )

    tests: list[dict[str, Any]] = []
    for metric in METRICS:
        for left, right in PAIRS:
            left_values = [row[metric] for row in rows if row["method"] == left]
            right_values = [row[metric] for row in rows if row["method"] == right]
            result = exact_label_swap(left_values, right_values)
            tests.append(
                {
                    "metric": metric,
                    "left": left,
                    "right": right,
                    "left_label": METHOD_LABELS[left],
                    "right_label": METHOD_LABELS[right],
                    **result,
                }
            )

    with (args.image_root / "perceptual_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.image_root / "perceptual_label_swap.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tests[0]))
        writer.writeheader()
        writer.writerows(tests)
    write_report(args.image_root / "perceptual_report.txt", rows, tests)
    metadata = {
        "image_root": str(args.image_root.resolve()),
        "query_indices": args.query_indices,
        "methods": list(METHODS),
        "clip_model": args.clip_model,
        "lpips_network": args.lpips_net,
        "dino_repository": args.dino_repository,
        "dino_model": args.dino_model,
    }
    (args.image_root / "perceptual_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[done] {args.image_root / 'perceptual_report.txt'}", flush=True)


if __name__ == "__main__":
    main()
