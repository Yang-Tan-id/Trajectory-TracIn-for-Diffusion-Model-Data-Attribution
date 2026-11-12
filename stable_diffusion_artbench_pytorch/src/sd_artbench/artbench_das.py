"""ArtBench DAS with exact LoRA gradients and latent noise residuals."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr
from tqdm.auto import tqdm

from .attribution.autograd import LinearGradSampleCapture
from .attribution.projection import CountSketchProjector, stable_seed
from .attribution.runtime import (
    AttributionArtBenchDataset,
    build_runtime,
    deterministic_noise,
    encode_image,
    encode_prompt,
    list_checkpoints,
    load_queries,
    load_raw_checkpoint,
    select_timesteps,
)
from .config import ArtBenchConfig


DEFAULT_LAMBDAS = (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000)
TARGETS = ("endpoint_difference", "trajectory_difference", "simple_loss_50x1", "noise_trajectory")


def parse_ints(text: str) -> list[int]:
    result: list[int] = []
    for token in text.replace(",", " ").split():
        if "-" in token[1:]:
            left, right = token.split("-", 1)
            result.extend(range(int(left), int(right) + 1))
        else:
            result.append(int(token))
    return result


def parse_floats(text: str) -> list[float]:
    return [float(value) for value in text.replace(",", " ").split()]


def save_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def output_probe(reference: torch.Tensor, timestep: int) -> torch.Tensor:
    return deterministic_noise(reference[:1], "artbench-das-output-probe", timestep)


def prediction_target(runtime, latents, noise, timesteps):
    noisy = runtime.scheduler.add_noise(latents, noise, timesteps)
    prediction = runtime.unet(noisy, timesteps, runtime._das_hidden).sample
    if runtime.scheduler.config.prediction_type == "epsilon":
        target = noise
    elif runtime.scheduler.config.prediction_type == "v_prediction":
        target = runtime.scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(runtime.scheduler.config.prediction_type)
    return prediction, target


def cache_training_inputs(runtime, dataset):
    latents, hidden = [], []
    for position in tqdm(range(len(dataset)), desc="cache ArtBench latents/prompts"):
        item = dataset[position]
        latents.append(encode_image(runtime, item["pixel_values"]).detach().cpu())
        hidden.append(encode_prompt(runtime, item["input_ids"]).detach().cpu())
    return latents, hidden


def projected_term(runtime, latents, hidden, *, sample_keys, timestep, projector, batch_size, train):
    count = len(latents)
    features = np.empty((count, projector.proj_dim), dtype=np.float32)
    residuals = np.empty((count,), dtype=np.float32) if train else None
    capture = LinearGradSampleCapture(runtime.unet, runtime.parameters)
    for start in tqdm(range(0, count, batch_size), desc=f"DAS t={timestep}"):
        stop = min(start + batch_size, count)
        latent = torch.cat(latents[start:stop]).to(runtime.device, dtype=runtime.weight_dtype)
        hidden_batch = torch.cat(hidden[start:stop]).to(runtime.device, dtype=runtime.weight_dtype)
        noise = torch.cat([
            deterministic_noise(latent[row:row + 1], "artbench-das", timestep, sample_keys[start + row])
            for row in range(stop - start)
        ])
        t = torch.full((stop - start,), timestep, device=runtime.device, dtype=torch.long)
        runtime._das_hidden = hidden_batch
        capture.start(stop - start)
        prediction, target = prediction_target(runtime, latent, noise, t)
        probe = output_probe(prediction, timestep)
        scale = math.sqrt(prediction[0].numel())
        scalar = (prediction.float() * probe.float()).flatten(1).sum(1) / scale
        residual = ((prediction.float() - target.float()) * probe.float()).flatten(1).sum(1) / scale
        _, gradients = next(capture.loss_gradients(scalar, chunk_size=stop - start))
        projected, _ = projector.project_batch(gradients)
        projected = projected / projected.norm(dim=1, keepdim=True).clamp_min(1e-12)
        features[start:stop] = projected.detach().cpu().numpy()
        if residuals is not None:
            residuals[start:stop] = residual.detach().cpu().numpy()
        del prediction, target, scalar, gradients, projected
    capture.close()
    return features, residuals


def runtime_and_checkpoint(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    runtime = build_runtime(
        model_name=args.model,
        rank=args.rank,
        device=device,
        mixed_precision=args.mixed_precision,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    checkpoint = list_checkpoints(args.checkpoint_root)[-1]
    load_raw_checkpoint(runtime, checkpoint)
    return runtime, checkpoint


def train_mode(args):
    runtime, checkpoint = runtime_and_checkpoint(args)
    config = ArtBenchConfig()
    dataset = AttributionArtBenchDataset(
        args.metadata_csv, args.image_root, args.index_path, runtime.tokenizer
    )
    if len(dataset) != 5000:
        raise ValueError(f"ArtBench DAS requires all 5000 points, found {len(dataset)}")
    latents, hidden = cache_training_inputs(runtime, dataset)
    projector = CountSketchProjector(args.proj_dim, stable_seed("artbench-das", checkpoint.name))
    all_features, all_residuals, all_grams = [], [], []
    for timestep in args.timesteps:
        features, residuals = projected_term(
            runtime, latents, hidden, timestep=timestep, projector=projector,
            sample_keys=list(range(len(dataset))), batch_size=args.batch_size, train=True,
        )
        all_features.append(features)
        all_residuals.append(residuals)
        feature_device = torch.from_numpy(features).to(runtime.device)
        all_grams.append((feature_device.T @ feature_device).cpu().numpy())
        del feature_device
    save_npz(
        args.output,
        train_features=np.stack(all_features), residuals=np.stack(all_residuals),
        gram_undamped=np.stack(all_grams), timesteps=np.asarray(args.timesteps),
        train_metadata_indices=np.asarray(dataset.metadata_indices),
        manifest_json=np.asarray(json.dumps({
            "dataset": "artbench2", "checkpoint": checkpoint.name, "mc": 1,
            "proj_dim": args.proj_dim, "residual": "latent_noise_prediction_projection",
            "gradient_parameters": "lora_only", "normalized_projected_gradients": True,
        }, sort_keys=True)),
    )
    print(f"[saved] {args.output}", flush=True)


def query_mode(args):
    runtime, checkpoint = runtime_and_checkpoint(args)
    records = load_queries(args.query_manifest)
    selected = [records[index] for index in args.query_ids]
    latents, hidden = [], []
    for record in tqdm(selected, desc="cache query latents/prompts"):
        with Image.open(record.image_path) as image:
            latents.append(encode_image(runtime, image).detach().cpu())
        hidden.append(encode_prompt(runtime, record.prompt).detach().cpu())
    projector = CountSketchProjector(args.proj_dim, stable_seed("artbench-das", checkpoint.name))
    terms = []
    for timestep in select_timesteps(50):
        features, _ = projected_term(
            runtime, latents, hidden, timestep=timestep, projector=projector,
            sample_keys=args.query_ids, batch_size=args.query_batch_size, train=False,
        )
        terms.append(features)
    save_npz(
        args.output, query_features=np.stack(terms),
        query_ids=np.asarray(args.query_ids), timesteps=np.asarray(select_timesteps(50)),
        manifest_json=np.asarray(json.dumps({"checkpoint": checkpoint.name, "proj_dim": args.proj_dim})),
    )
    print(f"[saved] {args.output}", flush=True)


def merge_mode(args):
    payloads = [np.load(path, allow_pickle=False) for path in args.inputs]
    try:
        if args.merge_kind == "train":
            order = np.argsort(np.concatenate([p["timesteps"] for p in payloads]))[::-1]
            fields = {
                "train_features": np.concatenate([p["train_features"] for p in payloads])[order],
                "residuals": np.concatenate([p["residuals"] for p in payloads])[order],
                "gram_undamped": np.concatenate([p["gram_undamped"] for p in payloads])[order],
                "timesteps": np.concatenate([p["timesteps"] for p in payloads])[order],
                "train_metadata_indices": payloads[0]["train_metadata_indices"],
            }
        else:
            query_ids = np.concatenate([p["query_ids"] for p in payloads])
            order = np.argsort(query_ids)
            fields = {
                "query_features": np.concatenate([p["query_features"] for p in payloads], axis=1)[:, order],
                "query_ids": query_ids[order], "timesteps": payloads[0]["timesteps"],
            }
        save_npz(args.output, **fields)
    finally:
        for payload in payloads:
            payload.close()
    print(f"[saved] {args.output}", flush=True)


def score_mode(args):
    with np.load(args.train_artifact, allow_pickle=False) as train_p, np.load(args.query_artifact, allow_pickle=False) as query_p:
        train = np.asarray(train_p["train_features"], dtype=np.float32)
        residual = np.asarray(train_p["residuals"], dtype=np.float32)
        gram = np.asarray(train_p["gram_undamped"], dtype=np.float32)
        indices = np.asarray(train_p["train_metadata_indices"], dtype=np.int64)
        query = np.asarray(query_p["query_features"], dtype=np.float32)
        query_ids = np.asarray(query_p["query_ids"], dtype=np.int64)
    if train.shape != (50, 5000, 4096) or query.shape[0] != 50:
        raise ValueError(f"unexpected train/query shapes: {train.shape}, {query.shape}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    eye = torch.eye(train.shape[2], device=device)
    for damping in args.lambdas:
        scores = np.zeros((query.shape[1], train.shape[1]), dtype=np.float64)
        for term in tqdm(range(50), desc=f"DAS lambda={damping:g}"):
            matrix = torch.from_numpy(gram[term]).to(device) + damping * eye
            train_t = torch.from_numpy(train[term]).to(device)
            query_t = torch.from_numpy(query[term]).to(device)
            rhs = torch.cat((query_t.T, train_t.T), dim=1)
            solved = torch.linalg.solve(matrix, rhs)
            query_solved = solved[:, :query.shape[1]]
            train_solved = solved[:, query.shape[1]:].T
            leverage = (train_t * train_solved).sum(1)
            denominator = (1.0 - leverage).clamp(min=-1e30, max=1e30)
            denominator = torch.where(
                denominator.abs() < 1e-6,
                torch.where(denominator >= 0, torch.full_like(denominator, 1e-6), torch.full_like(denominator, -1e-6)),
                denominator,
            )
            raw = (train_t @ query_solved) * torch.from_numpy(residual[term]).to(device)[:, None]
            raw = raw / denominator[:, None]
            scores += raw.T.double().square().cpu().numpy()
        scores /= 50.0
        output = args.output_dir / f"lambda_{damping:g}" / "scores.npz"
        save_npz(
            output, scores=scores, query_ids=query_ids,
            train_metadata_indices=indices,
            manifest_json=np.asarray(json.dumps({
                "algorithm": "das", "damping": damping, "timesteps": 50, "mc": 1,
                "proj_dim": 4096, "sherman_morrison_denominator": True,
                "score_contraction": "squared", "residual": "latent_noise_prediction_projection",
            }, sort_keys=True)),
        )
        print(f"[saved] {output}", flush=True)


def lds_mode(args):
    import csv
    config = ArtBenchConfig()
    with np.load(args.score_path, allow_pickle=False) as payload:
        scores = np.asarray(payload["scores"], dtype=np.float64)
        indices = np.asarray(payload["train_metadata_indices"], dtype=np.int64)
        query_ids = np.asarray(payload["query_ids"], dtype=np.int64)
    position = {int(value): slot for slot, value in enumerate(indices)}
    tasks = [(seed, subset) for seed in (0, 1, 2) for subset in range(64)]
    membership = np.zeros((192, 5000), dtype=np.float64)
    for row, (seed, subset) in enumerate(tasks):
        path = args.lds_root / "train_seed_42" / f"m_64_k_2500_subset_seed_{seed}" / "indices" / f"subset_{subset:04d}.pkl"
        with path.open("rb") as handle:
            subset_indices = pickle.load(handle)
        membership[row, [position[int(value)] for value in subset_indices]] = 1
    with args.true_f.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {(int(r["subset_seed"]), int(r["subset_id"]), int(r["query_index"])): r for r in rows}
    lines = ["ARTBENCH DAS LDS", "=" * 72, f"lambda={args.lds_lambda:g} queries={len(query_ids)} subsets=192"]
    for target in TARGETS:
        values = []
        for slot, query_id in enumerate(query_ids):
            actual = np.asarray([float(lookup[(seed, subset, int(query_id))][target]) for seed, subset in tasks])
            prediction = -(membership @ scores[slot])
            values.append(float(spearmanr(prediction, actual).statistic) * 100.0)
        lines.append(f"{target:28s} mean={np.mean(values):+8.3f}% median={np.median(values):+8.3f}% std={np.std(values):8.3f}%")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"[saved] {args.output}", flush=True)


def parser():
    config = ArtBenchConfig()
    project = config.index_root.parent.parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=("train", "query", "merge", "score", "lds"))
    p.add_argument("--model", default="lambdalabs/miniSD-diffusers")
    p.add_argument("--checkpoint-root", type=Path, default=project / "outputs/artbench2-lora")
    p.add_argument("--metadata-csv", type=Path, default=config.metadata_csv)
    p.add_argument("--image-root", type=Path, default=config.image_root)
    p.add_argument("--index-path", type=Path, default=config.index_root / "idx-train.pkl")
    p.add_argument("--query-manifest", type=Path, default=project / "outputs/artbench2-queries-60/manifest.json")
    p.add_argument("--query-ids", type=parse_ints, default=list(range(60)))
    p.add_argument("--timesteps", type=parse_ints, default=select_timesteps(50))
    p.add_argument("--proj-dim", type=int, default=4096)
    p.add_argument("--rank", type=int, default=128)
    p.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="fp16")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--query-batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path)
    p.add_argument("--inputs", type=Path, nargs="*")
    p.add_argument("--merge-kind", choices=("train", "query"))
    p.add_argument("--train-artifact", type=Path)
    p.add_argument("--query-artifact", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--lambdas", type=parse_floats, default=list(DEFAULT_LAMBDAS))
    p.add_argument("--score-path", type=Path)
    p.add_argument("--lds-lambda", type=float)
    p.add_argument("--lds-root", type=Path, default=project / "outputs/artbench2-lds")
    p.add_argument("--true-f", type=Path, default=project / "outputs/artbench2-lds/evaluation/four_targets_q0_59/true_f_four_targets.csv")
    return p


def main():
    args = parser().parse_args()
    {"train": train_mode, "query": query_mode, "merge": merge_mode, "score": score_mode, "lds": lds_mode}[args.mode](args)


if __name__ == "__main__":
    main()
