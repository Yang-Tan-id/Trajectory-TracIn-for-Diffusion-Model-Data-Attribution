"""Stable-Diffusion/ArtBench runtime shared by attribution methods."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer

from ..config import label_to_prompt
from ..train_lora import create_lora_layers, set_attention_dropout
from .autograd import batched_loss_gradients
from .projection import stable_seed


@dataclass(frozen=True)
class QueryRecord:
    query_id: int
    image_path: Path
    prompt: str
    seed: int


class AttributionArtBenchDataset(Dataset):
    """The selected training split in stable index-file order.

    Official D-TRAK gradient extraction disables random horizontal flips even
    when training used them, so attribution preprocessing is deterministic.
    """

    def __init__(
        self,
        metadata_csv: Path,
        image_root: Path,
        index_path: Path,
        tokenizer: CLIPTokenizer,
        resolution: int = 256,
    ) -> None:
        frame = pd.read_csv(metadata_csv)
        with index_path.open("rb") as handle:
            indices = list(pickle.load(handle))
        self.metadata_indices = [int(index) for index in indices]
        self.frame = frame.loc[indices].copy()
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    resolution, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, position: int) -> dict[str, object]:
        row = self.frame.iloc[position]
        path = self.image_root / row["label"] / row["name"]
        with Image.open(path) as image:
            pixels = self.transform(image.convert("RGB"))
        tokens = self.tokenizer(
            label_to_prompt(row["label"]),
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]
        return {
            "pixel_values": pixels,
            "input_ids": tokens,
            "position": position,
            "metadata_index": self.metadata_indices[position],
            "label": row["label"],
        }


@dataclass
class AttributionRuntime:
    tokenizer: CLIPTokenizer
    scheduler: DDPMScheduler
    text_encoder: CLIPTextModel
    vae: AutoencoderKL
    unet: UNet2DConditionModel
    lora_layers: AttnProcsLayers
    optimizer: torch.optim.AdamW
    device: torch.device
    weight_dtype: torch.dtype

    @property
    def parameters(self) -> list[torch.nn.Parameter]:
        return list(self.lora_layers.parameters())


def build_runtime(
    *,
    model_name: str,
    rank: int,
    device: torch.device,
    mixed_precision: str,
    learning_rate: float,
    weight_decay: float,
) -> AttributionRuntime:
    if mixed_precision == "fp16" and device.type == "cuda":
        weight_dtype = torch.float16
    elif mixed_precision == "bf16" and device.type == "cuda":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(
        model_name, subfolder="text_encoder", torch_dtype=weight_dtype
    )
    vae = AutoencoderKL.from_pretrained(
        model_name, subfolder="vae", torch_dtype=weight_dtype
    )
    unet = UNet2DConditionModel.from_pretrained(
        model_name, subfolder="unet", torch_dtype=weight_dtype
    )
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    vae.to(device).eval()
    text_encoder.to(device).eval()
    unet.to(device).eval()
    lora_layers = create_lora_layers(unet, rank)
    lora_layers.to(device)
    set_attention_dropout(unet, 0.0)
    unet.eval()
    optimizer = torch.optim.AdamW(
        lora_layers.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    return AttributionRuntime(
        tokenizer=tokenizer,
        scheduler=scheduler,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        lora_layers=lora_layers,
        optimizer=optimizer,
        device=device,
        weight_dtype=weight_dtype,
    )


def list_checkpoints(root: Path) -> list[Path]:
    def epoch_number(path: Path) -> int:
        try:
            return int(path.name.rsplit("-", 1)[-1])
        except ValueError as error:
            raise ValueError(f"malformed checkpoint directory: {path}") from error

    checkpoints = sorted(root.glob("checkpoint-epoch-*"), key=epoch_number)
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint-epoch-* directories found in {root}")
    return checkpoints


def _find_checkpoint_file(checkpoint: Path, stems: Sequence[str]) -> Path:
    for stem in stems:
        candidates = sorted(checkpoint.glob(f"{stem}*"))
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"none of {list(stems)} found in {checkpoint}")


def load_raw_checkpoint(runtime: AttributionRuntime, checkpoint: Path) -> float:
    """Load raw LoRA and AdamW state, returning the checkpoint learning rate."""

    model_path = _find_checkpoint_file(checkpoint, ("pytorch_model", "model"))
    if model_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        model_state = load_file(str(model_path), device="cpu")
    else:
        model_state = torch.load(model_path, map_location="cpu")
    runtime.lora_layers.load_state_dict(model_state, strict=True)
    optimizer_path = _find_checkpoint_file(checkpoint, ("optimizer",))
    runtime.optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
    for state in runtime.optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(runtime.device)
    runtime.unet.eval()
    rates = {float(group["lr"]) for group in runtime.optimizer.param_groups}
    if len(rates) != 1:
        raise ValueError(f"expected one checkpoint LR, found {sorted(rates)}")
    return rates.pop()


def load_queries(manifest_path: Path) -> list[QueryRecord]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for query_id, item in enumerate(payload):
        image_path = Path(item["file"])
        if not image_path.is_absolute():
            image_path = manifest_path.parent / image_path
        records.append(
            QueryRecord(
                query_id=query_id,
                image_path=image_path,
                prompt=str(item["prompt"]),
                seed=int(item["seed"]),
            )
        )
    return records


@torch.no_grad()
def encode_image(runtime: AttributionRuntime, image: Image.Image | torch.Tensor) -> torch.Tensor:
    if isinstance(image, Image.Image):
        transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        pixels = transform(image.convert("RGB"))
    else:
        pixels = image
    pixels = pixels.unsqueeze(0).to(runtime.device, dtype=runtime.weight_dtype)
    # Match the official D-TRAK ArtBench gradient script: VAE posterior mode.
    return runtime.vae.encode(pixels).latent_dist.mode() * runtime.vae.config.scaling_factor


@torch.no_grad()
def encode_prompt(runtime: AttributionRuntime, prompt: str | torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(prompt):
        input_ids = prompt.unsqueeze(0).to(runtime.device)
    else:
        input_ids = runtime.tokenizer(
            prompt,
            max_length=runtime.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(runtime.device)
    return runtime.text_encoder(input_ids)[0]


def deterministic_noise(
    reference: torch.Tensor, *seed_parts: object
) -> torch.Tensor:
    generator = torch.Generator(device=reference.device)
    generator.manual_seed(stable_seed(*seed_parts))
    return torch.randn(
        reference.shape,
        generator=generator,
        device=reference.device,
        dtype=reference.dtype,
    )


def simple_loss(
    runtime: AttributionRuntime,
    latent: torch.Tensor,
    hidden_states: torch.Tensor,
    timestep: int,
    noise: torch.Tensor,
) -> torch.Tensor:
    timesteps = torch.full(
        (latent.shape[0],), timestep, device=runtime.device, dtype=torch.long
    )
    noisy = runtime.scheduler.add_noise(latent, noise, timesteps)
    prediction = runtime.unet(noisy, timesteps, hidden_states).sample
    if runtime.scheduler.config.prediction_type == "epsilon":
        target = noise
    elif runtime.scheduler.config.prediction_type == "v_prediction":
        target = runtime.scheduler.get_velocity(latent, noise, timesteps)
    else:
        raise ValueError(
            f"unsupported prediction type {runtime.scheduler.config.prediction_type}"
        )
    return F.mse_loss(prediction.float(), target.float(), reduction="mean")


def simple_losses(
    runtime: AttributionRuntime,
    latent: torch.Tensor,
    hidden_states: torch.Tensor,
    timesteps: Sequence[int],
    noises: torch.Tensor,
) -> torch.Tensor:
    """Return one diffusion loss per supplied timestep/noise pair."""

    count = len(timesteps)
    latent_batch = latent.expand(count, *latent.shape[1:])
    hidden_batch = hidden_states.expand(count, *hidden_states.shape[1:])
    timestep_tensor = torch.as_tensor(timesteps, device=runtime.device, dtype=torch.long)
    noisy = runtime.scheduler.add_noise(latent_batch, noises, timestep_tensor)
    prediction = runtime.unet(noisy, timestep_tensor, hidden_batch).sample
    if runtime.scheduler.config.prediction_type == "epsilon":
        target = noises
    elif runtime.scheduler.config.prediction_type == "v_prediction":
        target = runtime.scheduler.get_velocity(latent_batch, noises, timestep_tensor)
    else:
        raise ValueError(
            f"unsupported prediction type {runtime.scheduler.config.prediction_type}"
        )
    return F.mse_loss(
        prediction.float(), target.float(), reduction="none"
    ).mean(dim=(1, 2, 3))


def gradients(
    loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    *,
    retain_graph: bool = False,
) -> list[torch.Tensor]:
    values = torch.autograd.grad(
        loss, parameters, allow_unused=True, retain_graph=retain_graph
    )
    return [
        torch.zeros_like(parameter) if value is None else value.detach()
        for parameter, value in zip(parameters, values)
    ]


def mean_simple_gradient(
    runtime: AttributionRuntime,
    latent: torch.Tensor,
    hidden_states: torch.Tensor,
    timesteps: Sequence[int],
    *,
    seed_parts: Sequence[object],
    full_l2_each: bool = False,
    timestep_batch_size: int = 8,
) -> list[torch.Tensor]:
    if timestep_batch_size < 1:
        raise ValueError("timestep_batch_size must be positive")
    parameters = runtime.parameters
    total = [torch.zeros_like(parameter, dtype=torch.float32) for parameter in parameters]
    for start in range(0, len(timesteps), timestep_batch_size):
        current_timesteps = [
            int(value) for value in timesteps[start : start + timestep_batch_size]
        ]
        noises = torch.cat(
            [
                deterministic_noise(latent, *seed_parts, timestep)
                for timestep in current_timesteps
            ],
            dim=0,
        )
        losses = simple_losses(runtime, latent, hidden_states, current_timesteps, noises)
        if full_l2_each:
            _, current = next(
                batched_loss_gradients(
                    losses,
                    parameters,
                    list(range(len(current_timesteps))),
                    chunk_size=len(current_timesteps),
                )
            )
            norms = torch.sqrt(
                sum(
                    gradient.float()
                    .reshape(len(current_timesteps), -1)
                    .square()
                    .sum(dim=1)
                    for gradient in current
                )
            ).clamp_min(1e-12)
            for destination, value in zip(total, current):
                view_shape = (len(current_timesteps),) + (1,) * (value.ndim - 1)
                normalized = value.float() / norms.view(view_shape)
                destination.add_(normalized.sum(dim=0), alpha=1.0 / len(timesteps))
        else:
            current = gradients(losses.mean(), parameters)
            weight = len(current_timesteps) / len(timesteps)
            for destination, value in zip(total, current):
                destination.add_(value.float(), alpha=weight)
    return total


def select_timesteps(count: int, total: int = 1000) -> list[int]:
    if count < 1 or count > total:
        raise ValueError(f"timestep count must be in [1, {total}]")
    if count == 50 and total == 1000:
        # Exact model-evaluation schedule used by the CelebA DDIM-50 runner.
        # These are 50 independent endpoint re-noising evaluations; they are
        # not states chained from a saved reverse-generation trajectory.
        return list(range(999, -1, -20))
    return np.linspace(total - 1, 0, count, dtype=np.int32).tolist()
