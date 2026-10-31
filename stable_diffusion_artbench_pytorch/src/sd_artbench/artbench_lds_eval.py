"""Evaluate four ArtBench LDS targets and score five attribution methods.

The four true measurements are endpoint image MSE, DDIM latent-trajectory MSE,
conditional denoising simple loss over 50 shared deterministic probes, and
predicted-noise deviation on the full-data model's DDIM reference states.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import pickle
import statistics
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .analyze_five_method_cross_query_topk_overlap import (
    METHODS,
    TRANSFORMS,
    load_artifact,
    score_bank,
)
from .config import ArtBenchConfig


SUBSET_BANKS = (0, 1, 2)
SUBSETS_PER_BANK = 64
EXPECTED_MODELS = len(SUBSET_BANKS) * SUBSETS_PER_BANK
EXPECTED_SUBSET_SIZE = 2500
TARGETS = (
    "endpoint_difference",
    "trajectory_difference",
    "simple_loss_50x1",
    "noise_trajectory",
)
TRUE_F_FIELDS = (
    "subset_seed",
    "subset_id",
    "query_index",
    "probe_count",
    "endpoint_difference",
    "trajectory_difference",
    "simple_loss_50x1",
    "noise_trajectory",
    "simple_loss_std",
    "noise_trajectory_std",
)


def parse_ints(value: str) -> list[int]:
    """Parse comma/space lists and inclusive ranges such as ``0-59``."""

    result: list[int] = []
    for token in value.replace(",", " ").split():
        if "-" in token[1:]:
            left, right = token.split("-", 1)
            start, stop = int(left), int(right)
            if stop < start:
                raise argparse.ArgumentTypeError(f"descending range: {token}")
            result.extend(range(start, stop + 1))
        else:
            result.append(int(token))
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("expected nonempty unique integers")
    return result


def expected_timesteps(training_timesteps: int, probe_count: int) -> list[int]:
    if probe_count < 1 or training_timesteps % probe_count:
        raise ValueError(
            f"{training_timesteps} training timesteps must be divisible by "
            f"probe count {probe_count}"
        )
    stride = training_timesteps // probe_count
    return [training_timesteps - 1 - stride * index for index in range(probe_count)]


def model_dir(root: Path, subset_seed: int, subset_id: int) -> Path:
    return (
        root
        / "train_seed_42"
        / f"m_64_k_2500_subset_seed_{subset_seed}"
        / "models"
        / f"subset_{subset_id:04d}"
    )


def index_path(root: Path, subset_seed: int, subset_id: int) -> Path:
    return (
        root
        / "train_seed_42"
        / f"m_64_k_2500_subset_seed_{subset_seed}"
        / "indices"
        / f"subset_{subset_id:04d}.pkl"
    )


def tasks() -> list[tuple[int, int]]:
    return [
        (subset_seed, subset_id)
        for subset_seed in SUBSET_BANKS
        for subset_id in range(SUBSETS_PER_BANK)
    ]


def load_queries(path: Path, query_ids: Sequence[int]) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"query manifest must be a nonempty list: {path}")
    records = []
    for query_id in query_ids:
        if not 0 <= query_id < len(payload):
            raise IndexError(f"query ID outside manifest: {query_id}")
        record = dict(payload[query_id])
        source = Path(record["file"])
        if not source.is_absolute():
            source = path.parent / source
        if not source.is_file():
            raise FileNotFoundError(source)
        records.append(
            {
                "query_index": query_id,
                "prompt": str(record["prompt"]),
                "seed": int(record["seed"]),
                "file": source.resolve(),
            }
        )
    return records


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_atomic(
    path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def true_f_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(row["subset_seed"]),
        int(row["subset_id"]),
        int(row["query_index"]),
    )


def merge_true_f_rows(
    shard_paths: Sequence[Path], query_ids: Sequence[int]
) -> list[dict[str, str]]:
    expected_queries = set(map(int, query_ids))
    merged: dict[tuple[int, int, int], dict[str, str]] = {}
    for path in shard_paths:
        for row in read_csv(path):
            key = true_f_key(row)
            if key[2] not in expected_queries:
                continue
            previous = merged.get(key)
            if previous is not None and previous != row:
                raise RuntimeError(f"conflicting duplicate true-f row: {key}")
            merged[key] = row
    expected = {
        (subset_seed, subset_id, query_id)
        for subset_seed, subset_id in tasks()
        for query_id in expected_queries
    }
    missing = sorted(expected - set(merged))
    if missing:
        raise RuntimeError(
            f"true-f coverage {len(merged)}/{len(expected)}; "
            f"first missing={missing[:8]}"
        )
    return [merged[key] for key in sorted(expected)]


def membership_matrix(
    lds_root: Path, candidate_indices: np.ndarray
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    candidates = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
    if len(candidates) != len(np.unique(candidates)):
        raise ValueError("candidate indices are not unique")
    position = {int(value): slot for slot, value in enumerate(candidates)}
    ordered_tasks = tasks()
    matrix = np.zeros((len(ordered_tasks), len(candidates)), dtype=np.float64)
    signatures: set[bytes] = set()
    for row, (subset_seed, subset_id) in enumerate(ordered_tasks):
        path = index_path(lds_root, subset_seed, subset_id)
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as handle:
            indices = np.asarray(pickle.load(handle), dtype=np.int64).reshape(-1)
        if len(indices) != EXPECTED_SUBSET_SIZE or len(np.unique(indices)) != len(indices):
            raise ValueError(f"invalid membership in {path}: {indices.shape}")
        try:
            slots = np.asarray([position[int(value)] for value in indices], dtype=np.int64)
        except KeyError as error:
            raise ValueError(f"{path}: membership outside attribution candidates") from error
        matrix[row, slots] = 1.0
        signature = np.packbits(matrix[row].astype(np.uint8)).tobytes()
        if signature in signatures:
            raise ValueError(f"duplicate LDS membership: {(subset_seed, subset_id)}")
        signatures.add(signature)
    return matrix, ordered_tasks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    from scipy.stats import spearmanr

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return float("nan")
    return float(spearmanr(left, right).statistic)


def score_lds(
    *,
    true_f_rows: Sequence[dict[str, Any]],
    membership: np.ndarray,
    ordered_tasks: Sequence[tuple[int, int]],
    artifacts: dict[str, dict[str, Any]],
    query_ids: Sequence[int],
) -> list[dict[str, Any]]:
    true_lookup = {
        target: {true_f_key(row): float(row[target]) for row in true_f_rows}
        for target in TARGETS
    }
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, np.ndarray]] = [("all_192", np.arange(len(ordered_tasks)))]
    for subset_seed in SUBSET_BANKS:
        scopes.append(
            (
                f"subset_seed_{subset_seed}",
                np.asarray(
                    [
                        index
                        for index, task in enumerate(ordered_tasks)
                        if task[0] == subset_seed
                    ],
                    dtype=np.int64,
                ),
            )
        )
    for method, artifact_name, variant, raw_sign, adamw_sign in METHODS:
        artifact = artifacts[artifact_name]
        for transform, orientation in zip(
            TRANSFORMS, (raw_sign, adamw_sign), strict=True
        ):
            bank = score_bank(artifact, transform=transform, variant=variant)
            if max(query_ids) >= bank.shape[0]:
                raise IndexError(f"{method} {transform}: query coverage={bank.shape[0]}")
            for query_id in query_ids:
                oriented = orientation * np.asarray(bank[query_id], dtype=np.float64)
                if not np.all(np.isfinite(oriented)):
                    raise ValueError(f"nonfinite attribution: {method} {transform} Q{query_id}")
                prediction = -(membership @ oriented)
                for target_name in TARGETS:
                    target = np.asarray(
                        [true_lookup[target_name][(seed, subset, query_id)]
                         for seed, subset in ordered_tasks], dtype=np.float64)
                    for scope, selected in scopes:
                        rows.append(
                            {
                                "target": target_name,
                                "method": method,
                                "transform": transform,
                                "query_index": query_id,
                                "scope": scope,
                                "subsets": len(selected),
                                "prediction_orientation": "-(membership@oriented_score)",
                                "stored_score_multiplier": orientation,
                                "spearman": spearman(prediction[selected], target[selected]),
                            }
                        )
    return rows


def _image_tensor(path: Path, device: Any, dtype: Any) -> Any:
    import torch
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        scale = 256.0 / min(width, height)
        resized = image.resize(
            (round(width * scale), round(height * scale)), Image.Resampling.BILINEAR
        )
        left, top = (resized.width - 256) // 2, (resized.height - 256) // 2
        cropped = resized.crop((left, top, left + 256, top + 256))
        array = np.asarray(cropped, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device, dtype)


def _build_probe(
    pipeline: Any,
    scheduler: Any,
    record: dict[str, Any],
    timesteps: Sequence[int],
    seed: int,
) -> tuple[Any, Any, Any, Any]:
    import torch

    device, dtype = pipeline.device, pipeline.unet.dtype
    pixels = _image_tensor(Path(record["file"]), device, pipeline.vae.dtype)
    latent_generator = torch.Generator(device=device).manual_seed(seed)
    with torch.inference_mode():
        latent = pipeline.vae.encode(pixels).latent_dist.sample(
            generator=latent_generator
        )
        latent = latent * pipeline.vae.config.scaling_factor
        input_ids = pipeline.tokenizer(
            record["prompt"],
            max_length=pipeline.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        hidden = pipeline.text_encoder(input_ids)[0].to(dtype=dtype)
    cpu_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    noise = torch.randn(
        (len(timesteps),) + tuple(latent.shape[1:]),
        generator=cpu_generator,
        dtype=torch.float32,
    ).to(device=device, dtype=dtype)
    t_values = torch.as_tensor(timesteps, device=device, dtype=torch.long)
    repeated = latent.to(dtype=dtype).expand(len(timesteps), -1, -1, -1)
    noisy = scheduler.add_noise(repeated, noise, t_values)
    if scheduler.config.prediction_type == "epsilon":
        target = noise
    elif scheduler.config.prediction_type == "v_prediction":
        target = scheduler.get_velocity(repeated, noise, t_values)
    else:
        raise ValueError(f"unsupported prediction type: {scheduler.config.prediction_type}")
    # CPU caching keeps all query probes inexpensive while each subset LoRA is loaded once.
    return tuple(value.detach().cpu() for value in (noisy, t_values, hidden, target))


def _simple_losses(pipeline: Any, probe: tuple[Any, ...], batch_size: int) -> np.ndarray:
    import torch

    noisy, timesteps, hidden, target = probe
    values: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(timesteps), batch_size):
            stop = min(start + batch_size, len(timesteps))
            count = stop - start
            prediction = pipeline.unet(
                noisy[start:stop].to(pipeline.device),
                timesteps[start:stop].to(pipeline.device),
                encoder_hidden_states=hidden.to(pipeline.device).expand(
                    count, -1, -1
                ),
            ).sample.float()
            expected = target[start:stop].to(pipeline.device).float()
            losses = torch.mean(
                (prediction - expected) ** 2,
                dim=tuple(range(1, prediction.ndim)),
            )
            values.extend(losses.cpu().tolist())
    return np.asarray(values, dtype=np.float64)


def _image_array(image: Any) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _mse(left: np.ndarray, right: np.ndarray) -> float:
    left64, right64 = np.asarray(left, np.float64), np.asarray(right, np.float64)
    if left64.shape != right64.shape:
        raise ValueError(f"shape mismatch: {left64.shape} vs {right64.shape}")
    return float(np.mean(np.square(left64 - right64), dtype=np.float64))


class _ReferenceCapture:
    """Capture compact UNet inputs and guided outputs along one DDIM path."""

    def __init__(self, guidance_scale: float) -> None:
        self.guidance_scale = float(guidance_scale)
        self.inputs: list[tuple[Any, int]] = []
        self.noises: list[np.ndarray] = []
        self.embeddings: Any | None = None
        self.pending: tuple[Any, int] | None = None

    def pre_hook(self, _module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        import torch

        sample = args[0] if args else kwargs["sample"]
        timestep = args[1] if len(args) > 1 else kwargs["timestep"]
        embeddings = kwargs["encoder_hidden_states"]
        value = int(timestep.flatten()[0].item()) if torch.is_tensor(timestep) else int(timestep)
        if self.embeddings is None:
            self.embeddings = embeddings.detach().to("cpu", torch.float16).clone()
        self.pending = (sample.detach().to("cpu", torch.float16).clone(), value)

    def post_hook(self, _module: Any, _args: tuple[Any, ...], _kwargs: dict[str, Any], output: Any) -> None:
        if self.pending is None:
            raise RuntimeError("UNet output without captured input")
        raw = output.sample.detach().float()
        if raw.shape[0] == 2:
            unconditional, conditional = raw.chunk(2)
            guided = unconditional + self.guidance_scale * (conditional - unconditional)
        else:
            raise RuntimeError(f"unexpected guidance batch: {raw.shape}")
        self.inputs.append(self.pending)
        self.noises.append(guided[0].cpu().numpy().copy())
        self.pending = None


def _load_lora(pipeline: Any, path: Path) -> None:
    from .train_lora import set_attention_dropout

    if not path.is_dir():
        raise FileNotFoundError(path)
    pipeline.unet.load_attn_procs(path)
    set_attention_dropout(pipeline.unet, 0.0)
    pipeline.unet.eval()


def _sample_reference(
    pipeline: Any, record: dict[str, Any], steps: int, guidance_scale: float,
    expected: Sequence[int],
) -> dict[str, Any]:
    import torch

    capture = _ReferenceCapture(guidance_scale)
    states: list[np.ndarray] = []

    def callback(_step: int, _timestep: int, latents: Any) -> None:
        states.append(latents[0].detach().float().cpu().numpy().copy())

    pre = pipeline.unet.register_forward_pre_hook(capture.pre_hook, with_kwargs=True)
    post = pipeline.unet.register_forward_hook(capture.post_hook, with_kwargs=True)
    try:
        generator = torch.Generator(device=pipeline.device).manual_seed(int(record["seed"]))
        with torch.inference_mode():
            output = pipeline(
                str(record["prompt"]), height=256, width=256,
                num_inference_steps=steps, guidance_scale=guidance_scale,
                generator=generator, callback=callback, callback_steps=1,
            )
    finally:
        pre.remove()
        post.remove()
    actual = [timestep for _, timestep in capture.inputs]
    if actual != list(expected) or len(states) != steps or len(capture.noises) != steps:
        raise RuntimeError(
            f"reference schedule mismatch: actual={actual}, expected={list(expected)} "
            f"states={len(states)} noise={len(capture.noises)}")
    if capture.embeddings is None:
        raise RuntimeError("reference prompt embeddings were not captured")
    return {
        "image": _image_array(output.images[0]),
        "states": np.stack(states),
        "inputs": capture.inputs,
        "embeddings": capture.embeddings,
        "noise": np.stack(capture.noises),
    }


def _sample_candidate(
    pipeline: Any, record: dict[str, Any], steps: int, guidance_scale: float,
    expected: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    states: list[np.ndarray] = []

    def callback(_step: int, _timestep: int, latents: Any) -> None:
        states.append(latents[0].detach().float().cpu().numpy().copy())

    generator = torch.Generator(device=pipeline.device).manual_seed(int(record["seed"]))
    with torch.inference_mode():
        output = pipeline(
            str(record["prompt"]), height=256, width=256,
            num_inference_steps=steps, guidance_scale=guidance_scale,
            generator=generator, callback=callback, callback_steps=1,
        )
    actual = [int(value) for value in pipeline.scheduler.timesteps.tolist()]
    if actual != list(expected) or len(states) != steps:
        raise RuntimeError(f"candidate schedule mismatch: {actual} vs {list(expected)}")
    return _image_array(output.images[0]), np.stack(states)


def _noise_deviation(
    pipeline: Any, reference: dict[str, Any], guidance_scale: float,
    batch_size: int,
) -> np.ndarray:
    import torch

    values: list[float] = []
    inputs = reference["inputs"]
    embeddings = reference["embeddings"]
    reference_noise = reference["noise"]
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            selected = inputs[start : start + batch_size]
            # Every captured sample already has the CFG batch dimension of two.
            sample = torch.cat([item[0] for item in selected], dim=0).to(
                pipeline.device, pipeline.unet.dtype)
            timestep = torch.as_tensor(
                [item[1] for item in selected for _ in range(2)],
                device=pipeline.device, dtype=torch.long)
            hidden = embeddings.to(pipeline.device, pipeline.unet.dtype).repeat(
                len(selected), 1, 1)
            raw = pipeline.unet(
                sample, timestep, encoder_hidden_states=hidden).sample.float()
            raw = raw.reshape(len(selected), 2, *raw.shape[1:])
            guided = raw[:, 0] + guidance_scale * (raw[:, 1] - raw[:, 0])
            expected = torch.from_numpy(reference_noise[start : start + len(selected)]).to(
                guided.device, torch.float32)
            mse = torch.mean((guided - expected) ** 2, dim=tuple(range(1, guided.ndim)))
            values.extend(mse.cpu().tolist())
    return np.asarray(values, dtype=np.float64)


def evaluate_true_f(args: argparse.Namespace) -> None:
    import torch
    from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("true-f evaluation requires CUDA")
    if not 0 <= args.model_shard_index < args.model_shard_count:
        raise ValueError("invalid model shard")
    records = load_queries(args.query_manifest, args.query_ids)
    selected_tasks = [
        task
        for index, task in enumerate(tasks())
        if index % args.model_shard_count == args.model_shard_index
    ]
    shard_path = (
        args.output_dir
        / "shards"
        / f"true_f_shard_{args.model_shard_index:02d}_of_{args.model_shard_count:02d}.csv"
    )
    existing_rows = read_csv(shard_path)
    existing = {true_f_key(row): row for row in existing_rows}

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(torch.device("cuda"))
    pipeline.set_progress_bar_config(disable=True)
    training_scheduler = DDPMScheduler.from_pretrained(args.model, subfolder="scheduler")
    timesteps = expected_timesteps(
        int(training_scheduler.config.num_train_timesteps), args.probe_count
    )
    stride = int(training_scheduler.config.num_train_timesteps) // args.probe_count
    pipeline.scheduler = DDIMScheduler.from_config(
        pipeline.scheduler.config, steps_offset=stride - 1)
    probes = {
        int(record["query_index"]): _build_probe(
            pipeline,
            training_scheduler,
            record,
            timesteps,
            args.probe_seed + int(record["query_index"]) * 10_000,
        )
        for record in records
    }
    print("[reference] loading full-data LoRA and caching four-target references", flush=True)
    _load_lora(pipeline, args.full_lora_dir)
    references = {
        int(record["query_index"]): _sample_reference(
            pipeline, record, args.probe_count, args.guidance_scale, timesteps)
        for record in records
    }
    print(
        f"[true-f setup] shard={args.model_shard_index}/{args.model_shard_count} "
        f"models={len(selected_tasks)} queries={len(records)} probes={args.probe_count}",
        flush=True,
    )
    started = time.monotonic()
    for model_position, (subset_seed, subset_id) in enumerate(selected_tasks, start=1):
        required = {
            (subset_seed, subset_id, int(record["query_index"])) for record in records
        }
        if required.issubset(existing):
            print(
                f"[skip model {model_position}/{len(selected_tasks)}] "
                f"seed={subset_seed} subset={subset_id}",
                flush=True,
            )
            continue
        run_dir = model_dir(args.lds_root, subset_seed, subset_id)
        if not (run_dir / "TRAINING_COMPLETE").is_file():
            raise FileNotFoundError(run_dir / "TRAINING_COMPLETE")
        final = run_dir / "final"
        if not final.is_dir():
            raise FileNotFoundError(final)
        _load_lora(pipeline, final)
        for query_position, record in enumerate(records, start=1):
            query_id = int(record["query_index"])
            key = (subset_seed, subset_id, query_id)
            if key in existing:
                continue
            losses = _simple_losses(pipeline, probes[query_id], args.batch_size)
            candidate_image, candidate_states = _sample_candidate(
                pipeline, record, args.probe_count, args.guidance_scale, timesteps)
            reference = references[query_id]
            noise = _noise_deviation(
                pipeline, reference, args.guidance_scale, args.noise_batch_size)
            row = {
                "subset_seed": subset_seed,
                "subset_id": subset_id,
                "query_index": query_id,
                "probe_count": len(losses),
                "endpoint_difference": _mse(reference["image"], candidate_image),
                "trajectory_difference": _mse(reference["states"], candidate_states),
                "simple_loss_50x1": float(losses.mean()),
                "noise_trajectory": float(noise.mean()),
                "simple_loss_std": float(losses.std()),
                "noise_trajectory_std": float(noise.std()),
            }
            existing[key] = {name: str(value) for name, value in row.items()}
            write_csv_atomic(
                shard_path, [existing[item] for item in sorted(existing)], TRUE_F_FIELDS)
            print(
                f"[true-f] model={model_position}/{len(selected_tasks)} "
                f"seed={subset_seed} subset={subset_id} "
                f"query={query_position}/{len(records)} Q{query_id} "
                f"end={row['endpoint_difference']:.8f} "
                f"traj={row['trajectory_difference']:.8f} "
                f"loss={row['simple_loss_50x1']:.8f} "
                f"noise={row['noise_trajectory']:.8f}",
                flush=True,
            )
        ordered = [existing[key] for key in sorted(existing)]
        write_csv_atomic(shard_path, ordered, TRUE_F_FIELDS)
        elapsed = time.monotonic() - started
        rate = model_position / elapsed if elapsed else 0.0
        eta = (len(selected_tasks) - model_position) / rate if rate else float("inf")
        print(
            f"[model complete] {model_position}/{len(selected_tasks)} "
            f"elapsed={elapsed / 3600:.2f}h eta={eta / 3600:.2f}h",
            flush=True,
        )
    del probes, references, pipeline
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[shard complete] {shard_path}", flush=True)


def merge_mode(args: argparse.Namespace) -> None:
    paths = sorted((args.output_dir / "shards").glob("true_f_shard_*_of_*.csv"))
    if not paths:
        raise FileNotFoundError(args.output_dir / "shards")
    rows = merge_true_f_rows(paths, args.query_ids)
    output = args.output_dir / "true_f_four_targets.csv"
    write_csv_atomic(output, rows, TRUE_F_FIELDS)
    print(f"[merge complete] rows={len(rows)} expected={EXPECTED_MODELS * len(args.query_ids)}")
    print(f"[saved] {output}")


def lds_mode(args: argparse.Namespace) -> None:
    true_f_path = args.output_dir / "true_f_four_targets.csv"
    true_f_rows = read_csv(true_f_path)
    if not true_f_rows:
        raise FileNotFoundError(true_f_path)
    artifacts = {
        "pollute": load_artifact(
            args.endpoint_root / "endpoint_pollute_timestamp_sum_squared.npz"
        ),
        "tracin": load_artifact(args.endpoint_root / "endpoint_tracin.npz"),
        "retrac": load_artifact(args.retrac_path),
    }
    candidate_sets = {
        name: tuple(map(int, artifact["indices"]))
        for name, artifact in artifacts.items()
    }
    first = next(iter(candidate_sets.values()))
    if any(set(values) != set(first) for values in candidate_sets.values()):
        raise ValueError("attribution artifacts use different candidate universes")
    candidate_indices = np.asarray(artifacts["pollute"]["indices"], dtype=np.int64)
    for artifact in artifacts.values():
        if not np.array_equal(artifact["indices"], candidate_indices):
            raise ValueError("attribution artifacts use different candidate order")
    membership, ordered_tasks = membership_matrix(args.lds_root, candidate_indices)
    rows = score_lds(
        true_f_rows=true_f_rows,
        membership=membership,
        ordered_tasks=ordered_tasks,
        artifacts=artifacts,
        query_ids=args.query_ids,
    )
    fields = list(rows[0])
    write_csv_atomic(args.output_dir / "five_method_lds_per_query.csv", rows, fields)

    all_rows = [row for row in rows if row["scope"] == "all_192"]
    lines = [
        "ARTBENCH FIVE-METHOD LDS — FOUR TARGETS",
        f"queries={len(args.query_ids)} subsets=192 (3 independent banks x 64)",
        "targets: endpoint difference, trajectory difference, shared-MC simple loss 50x1,",
        "         predicted-noise deviation on full-data DDIM reference states",
        "prediction: -(membership @ oriented attribution); larger LDS is better",
    ]
    summary: list[dict[str, Any]] = []
    for target_name in TARGETS:
        lines.extend([
            "", target_name.upper(), "-" * 84,
            f"{'METHOD':30s} {'TRANSFORM':13s} {'N':>4s} {'MEAN':>10s} {'STD':>10s} {'MEDIAN':>10s}",
        ])
        for method, *_ in METHODS:
            for transform in TRANSFORMS:
                selected = [
                    float(row["spearman"])
                    for row in all_rows
                    if row["target"] == target_name
                    and row["method"] == method and row["transform"] == transform
                ]
                if not selected:
                    raise RuntimeError(f"missing LDS: {target_name} {method} {transform}")
                summary.append(
                    {
                        "target": target_name,
                        "method": method,
                        "transform": transform,
                        "queries": len(selected),
                        "subsets_per_query": 192,
                        "lds_mean": statistics.fmean(selected),
                        "lds_std": statistics.pstdev(selected),
                        "lds_median": statistics.median(selected),
                        "lds_min": min(selected),
                        "lds_max": max(selected),
                    }
                )
                lines.append(
                    f"{method:30s} {transform:13s} {len(selected):4d} "
                    f"{100 * statistics.fmean(selected):+9.3f}% "
                    f"{100 * statistics.pstdev(selected):9.3f}% "
                    f"{100 * statistics.median(selected):+9.3f}%"
                )
    write_csv_atomic(
        args.output_dir / "five_method_lds_summary.csv", summary, list(summary[0])
    )
    report = "\n".join(lines) + "\n"
    (args.output_dir / "five_method_lds_report.txt").write_text(
        report, encoding="utf-8"
    )
    (args.output_dir / "evaluation_manifest.json").write_text(
        json.dumps(
            {
                "query_ids": args.query_ids,
                "subset_banks": list(SUBSET_BANKS),
                "subsets_per_bank": SUBSETS_PER_BANK,
                "probe_count": args.probe_count,
                "timesteps": expected_timesteps(1000, args.probe_count),
                "targets": list(TARGETS),
                "simple_loss_mc": "50 deterministic probes shared across all subset models per query",
                "noise_trajectory": "predicted-noise MSE on full-data DDIM reference states",
                "prediction": "-(membership @ oriented attribution)",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("\n" + report, end="")
    print(f"[saved] {args.output_dir / 'five_method_lds_report.txt'}")


def parse_args() -> argparse.Namespace:
    config = ArtBenchConfig()
    project = config.index_root.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("true-f", "merge", "lds"))
    parser.add_argument(
        "--lds-root", type=Path, default=project / "outputs/artbench2-lds"
    )
    parser.add_argument(
        "--query-manifest",
        type=Path,
        default=project / "outputs/artbench2-queries-60/manifest.json",
    )
    parser.add_argument("--query-ids", type=parse_ints, default=list(range(60)))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project / "outputs/artbench2-lds/evaluation/four_targets_q0_59",
    )
    parser.add_argument("--model", default="lambdalabs/miniSD-diffusers")
    parser.add_argument(
        "--full-lora-dir", type=Path,
        default=project / "outputs/artbench2-lora/final")
    parser.add_argument("--probe-count", type=int, default=50)
    parser.add_argument("--probe-seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--noise-batch-size", type=int, default=5)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--model-shard-index", type=int, default=0)
    parser.add_argument("--model-shard-count", type=int, default=1)
    parser.add_argument(
        "--endpoint-root",
        type=Path,
        default=project / "outputs/attribution/endpoint-q60-t50-full-linear-hooks",
    )
    parser.add_argument(
        "--retrac-path",
        type=Path,
        default=project
        / "outputs/attribution/retrac-q60-t50-full-linear-hooks/retrac.npz",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.probe_count < 1 or args.batch_size < 1 or args.noise_batch_size < 1:
        raise ValueError("probe count and batch sizes must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "true-f":
        evaluate_true_f(args)
    elif args.mode == "merge":
        merge_mode(args)
    else:
        lds_mode(args)


if __name__ == "__main__":
    main()
