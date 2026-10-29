"""Shared math and deterministic sampling helpers for predicted-noise KL."""

from __future__ import annotations

import numpy as np


def uniform_timesteps(total_timesteps: int, count: int) -> np.ndarray:
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    if not 1 <= count <= total_timesteps:
        raise ValueError("count must be in [1, total_timesteps]")
    values = np.linspace(total_timesteps - 1, 0, count, dtype=np.int32)
    if len(np.unique(values)) != count:
        raise ValueError("uniform timestep selection contains duplicates")
    return values


def query_probe_seed(base_seed: int, query_id: int) -> int:
    if query_id < 0:
        raise ValueError("query_id must be nonnegative")
    return int(base_seed) + 100003 * int(query_id)


def predicted_noise_kl_numpy(
    subset_prediction: np.ndarray, full_prediction: np.ndarray
) -> float:
    subset = np.asarray(subset_prediction, dtype=np.float64)
    full = np.asarray(full_prediction, dtype=np.float64)
    if subset.shape != full.shape:
        raise ValueError(f"prediction shapes differ: {subset.shape} != {full.shape}")
    if subset.size == 0:
        raise ValueError("predictions must not be empty")
    return float(0.5 * np.mean((subset - full) ** 2))
