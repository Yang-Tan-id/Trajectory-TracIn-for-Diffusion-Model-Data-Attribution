from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from diffusers import DDPMScheduler, UNet2DModel
from torchvision import transforms
from tqdm.auto import tqdm

try:
    from .config_loader import load_config, require_attr
except ImportError:
    import sys

    refine_root = Path(__file__).resolve().parents[1]
    if str(refine_root) not in sys.path:
        sys.path.insert(0, str(refine_root))
    from common.config_loader import load_config, require_attr


def _split_ranges(text: str | None, default: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    if not text:
        return default
    ranges = []
    for part in text.replace(",", " ").split():
        start_end = part.strip().replace(":", "-").split("-")
        if len(start_end) != 2:
            raise ValueError("Ranges must look like '1-2500,2501-5000'.")
        ranges.append((int(start_end[0]), int(start_end[1])))
    return tuple(ranges)


def _candidate_indices(n: int, ranges: tuple[tuple[int, int], ...], index_base: int) -> np.ndarray:
    picked: list[int] = []
    for start, end in ranges:
        lo = int(start) - int(index_base)
        hi = int(end) - int(index_base)
        if lo < 0 or hi < lo:
            raise ValueError(f"Invalid score range: {(start, end)}")
        picked.extend(range(lo, min(hi, n - 1) + 1))
    return np.asarray(sorted(set(picked)), dtype=np.int64)


def _range_suffix(ranges: tuple[tuple[int, int], ...]) -> str:
    return "range_" + "__".join(f"{int(s)}_{int(e)}" for s, e in ranges)


def _timesteps_for_algorithm(algorithm: str, total_timesteps: int) -> list[int]:
    explicit = os.environ.get("UNPROMPTED_TIMESTEPS")
    if explicit:
        return [int(x) for x in explicit.replace(",", " ").split() if x]
    if algorithm == "end_tracin":
        return [total_timesteps - 1]
    if algorithm in ("traj_tracin", "journey_trak"):
        k = int(os.environ.get("UNPROMPTED_NUM_TRAJ_STEPS", "20"))
        return np.linspace(0, total_timesteps - 1, k, dtype=np.int64).tolist()
    return [0, 200, 400, 600, 800, total_timesteps - 1]


def _load_dataset(dataset_name_or_path: str, index_path: str | None):
    if os.path.exists(dataset_name_or_path):
        ds = load_from_disk(os.path.join(dataset_name_or_path, "train"))
    else:
        ds = load_dataset(dataset_name_or_path, split="train")
    if index_path:
        with open(index_path, "rb") as handle:
            sub_idx = pickle.load(handle)
        ds = ds.select(sub_idx)
    return ds


def _image_key(example) -> str:
    if "img" in example:
        return "img"
    if "image" in example:
        return "image"
    raise KeyError("Expected dataset examples to contain an 'img' or 'image' field.")


def _make_transform(resolution: int):
    return transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


def _batch_images(ds, indices: np.ndarray, transform, batch_size: int) -> Iterable[tuple[np.ndarray, torch.Tensor]]:
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        images = []
        for idx in batch_indices.tolist():
            ex = ds[int(idx)]
            img = ex[_image_key(ex)].convert("RGB")
            images.append(transform(img))
        yield batch_indices, torch.stack(images, dim=0)


def _score_batch(
    model,
    scheduler,
    clean_images: torch.Tensor,
    timesteps: list[int],
    num_mc: int,
    seed: int,
    device,
) -> torch.Tensor:
    scores = torch.zeros((clean_images.shape[0],), dtype=torch.float64, device=device)
    clean_images = clean_images.to(device)
    total = 0
    for t_int in timesteps:
        t = torch.full((clean_images.shape[0],), int(t_int), dtype=torch.long, device=device)
        for mc in range(num_mc):
            gen = torch.Generator(device=device).manual_seed(int(seed) + 100000 * int(t_int) + mc)
            noise = torch.randn(clean_images.shape, generator=gen, device=device, dtype=clean_images.dtype)
            noisy = scheduler.add_noise(clean_images, noise, t)
            with torch.no_grad():
                pred = model(noisy, t).sample
                per_item_loss = F.mse_loss(pred, noise, reduction="none").flatten(1).mean(dim=1)
            scores += (-per_item_loss).to(torch.float64)
            total += 1
    return scores / max(1, total)


def run_attribution(config_path: str, algorithm: str) -> None:
    cfg = load_config(config_path)
    dataset_name = require_attr(cfg, "DATASET_NAME")
    experiment_tag = require_attr(cfg, "EXPERIMENT_TAG")
    model_root = require_attr(cfg, "MODEL_ROOT")
    attribution_root = require_attr(cfg, "ATTRIBUTION_ROOT")
    hf_root = require_attr(cfg, "HF_DATASET_ROOT")
    lds_index_root = getattr(cfg, "LDS_INDEX_ROOT", None)

    subset_index = int(os.environ.get("SUBSET_INDEX", "0"))
    train_seed = int(os.environ.get("TRAIN_SEED", os.environ.get("UNPROMPTED_MODEL_SEED", "0")))
    model_dir = Path(os.environ.get(
        "UNPROMPTED_MODEL_DIR",
        str(Path(model_root) / algorithm / "unprompted" / f"ddpm-sub-{subset_index}-{train_seed}"),
    ))
    out_dir = Path(os.environ.get("UNPROMPTED_ATTRIBUTION_OUT_DIR", str(Path(attribution_root) / f"{algorithm}_unprompted")))

    ranges = _split_ranges(
        os.environ.get("SCORE_INDEX_RANGES") or os.environ.get("ATTRIBUTION_RANGES"),
        getattr(cfg, "SCORE_INDEX_RANGES", ((1, 10000),)),
    )
    if ranges:
        suffix = _range_suffix(ranges)
        if not out_dir.name.endswith(suffix):
            out_dir = out_dir.with_name(f"{out_dir.name}_{suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)

    index_path = None
    if lds_index_root is not None:
        candidate = Path(lds_index_root) / f"sub-idx-{subset_index}.pkl"
        if candidate.exists():
            index_path = str(candidate)

    ds = _load_dataset(hf_root, index_path)
    score_index_base = int(getattr(cfg, "COMMON_CIFAR", {}).get("score_index_base", 1))
    picked = _candidate_indices(len(ds), ranges, score_index_base)

    device = torch.device("cuda" if torch.cuda.is_available() and os.environ.get("UNPROMPTED_DEVICE", "gpu") != "cpu" else "cpu")
    model = UNet2DModel.from_pretrained(model_dir / "unet").to(device)
    scheduler = DDPMScheduler.from_pretrained(model_dir / "scheduler")
    model.eval()

    timesteps = _timesteps_for_algorithm(algorithm, int(scheduler.config.num_train_timesteps))
    batch_size = int(os.environ.get("UNPROMPTED_SCORE_BATCH_SIZE", "64"))
    num_mc = int(os.environ.get("UNPROMPTED_SCORE_MC", "1"))
    resolution = int(os.environ.get("UNPROMPTED_RESOLUTION", "32"))
    transform = _make_transform(resolution)

    scores = np.zeros((len(picked),), dtype=np.float64)
    start_time = time.time()
    cursor = 0
    for batch_indices, batch in tqdm(
        _batch_images(ds, picked, transform, batch_size),
        total=math.ceil(len(picked) / batch_size),
        desc=f"unprompted {algorithm}",
    ):
        batch_scores = _score_batch(
            model=model,
            scheduler=scheduler,
            clean_images=batch,
            timesteps=timesteps,
            num_mc=num_mc,
            seed=train_seed,
            device=device,
        )
        n = len(batch_indices)
        scores[cursor : cursor + n] = batch_scores.detach().cpu().numpy()
        cursor += n

    topk = min(int(os.environ.get("TOPK", "10000")), len(picked))
    order = np.argsort(-scores)[:topk]
    top = [
        {"idx": int(picked[i]), "idx_1based": int(picked[i]) + 1, "score": float(scores[i])}
        for i in order
    ]
    run_config = {
        "backend": "diffusers_unprompted",
        "dataset_name": dataset_name,
        "experiment_tag": experiment_tag,
        "algorithm": algorithm,
        "model_dir": str(model_dir),
        "hf_dataset_root": hf_root,
        "index_path": index_path,
        "subset_index": subset_index,
        "train_seed": train_seed,
        "score_index_ranges": ranges,
        "score_index_base": score_index_base,
        "timesteps": timesteps,
        "num_mc": num_mc,
        "batch_size": batch_size,
        "device": str(device),
        "elapsed_s": time.time() - start_time,
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)
    with open(out_dir / "result_topk.json", "w") as f:
        json.dump({"top": top, "num_scored": int(len(picked))}, f, indent=2)
    np.save(out_dir / "scores.npy", scores)
    np.save(out_dir / "score_indices.npy", picked.astype(np.int64))
    with open(out_dir / "score_indices.json", "w") as f:
        json.dump(
            {
                "score_indices": [int(x) for x in picked],
                "score_indices_1based": [int(x) + 1 for x in picked],
                "score_index_ranges": ranges,
                "score_index_base": score_index_base,
            },
            f,
            indent=2,
        )
    print(f"Saved unprompted attribution to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unprompted diffusers attribution scorer.")
    parser.add_argument("config", type=str)
    parser.add_argument("--algorithm", default=os.environ.get("ALGORITHM", "das"))
    args = parser.parse_args()
    run_attribution(args.config, args.algorithm)


if __name__ == "__main__":
    main()
