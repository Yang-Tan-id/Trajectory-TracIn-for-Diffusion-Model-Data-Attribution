"""I/O and validation for fixed Stable-Diffusion generation trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class QueryTrajectory:
    """The U-Net input and CFG-combined output at every DDIM model call."""

    latents: np.ndarray
    predicted_noise: np.ndarray
    timesteps: np.ndarray


def ddim_stride_timesteps(
    inference_steps: int, *, training_timesteps: int = 1000
) -> np.ndarray:
    """Return the historical ``999, 979, ..., 19`` DDIM-50 schedule."""

    if inference_steps < 1:
        raise ValueError("inference_steps must be positive")
    if training_timesteps % inference_steps:
        raise ValueError(
            "training_timesteps must be divisible by inference_steps for the "
            "fixed-stride trajectory"
        )
    stride = training_timesteps // inference_steps
    return np.arange(
        training_timesteps - 1,
        -1,
        -stride,
        dtype=np.int64,
    )[:inference_steps]


def save_query_trajectory(
    path: Path,
    *,
    latents: np.ndarray,
    predicted_noise: np.ndarray,
    timesteps: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            latents=np.asarray(latents, dtype=np.float16),
            predicted_noise=np.asarray(predicted_noise, dtype=np.float16),
            timesteps=np.asarray(timesteps, dtype=np.int64),
        )
    temporary.replace(path)


def load_query_trajectory(path: Path, *, expected_steps: int) -> QueryTrajectory:
    if not path.is_file():
        raise FileNotFoundError(
            f"query trajectory not found: {path}. Regenerate the query with the "
            "updated sd_artbench.generate command."
        )
    with np.load(path, allow_pickle=False) as payload:
        required = {"latents", "predicted_noise", "timesteps"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"{path}: missing trajectory arrays {sorted(missing)}")
        latents = np.asarray(payload["latents"])
        predicted_noise = np.asarray(payload["predicted_noise"])
        timesteps = np.asarray(payload["timesteps"], dtype=np.int64)
    if latents.shape != predicted_noise.shape:
        raise ValueError(
            f"{path}: latent shape {latents.shape} != predicted-noise shape "
            f"{predicted_noise.shape}"
        )
    if latents.ndim != 4:
        raise ValueError(f"{path}: expected [step,C,H,W], got {latents.shape}")
    if len(timesteps) != latents.shape[0]:
        raise ValueError(f"{path}: timestep count does not match trajectory tensors")
    if len(timesteps) != expected_steps:
        raise ValueError(
            f"{path}: contains {len(timesteps)} steps, requested {expected_steps}"
        )
    if np.any(np.diff(timesteps) >= 0):
        raise ValueError(f"{path}: trajectory timesteps must be strictly descending")
    return QueryTrajectory(latents, predicted_noise, timesteps)
