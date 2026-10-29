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


def fixed_beta_vlb_weights_numpy(betas: np.ndarray) -> np.ndarray:
    """Return epsilon-MSE weights for a fixed-beta DDPM reverse process.

    The model-dependent Gaussian KL/decoder-mean term is

        beta_t^2 / (2 sigma_t^2 alpha_t (1 - alpha_bar_t)) * MSE,

    and the sampler in this repository uses sigma_t^2 = beta_t.
    Model-independent variance and prior constants are intentionally omitted;
    they do not affect LDS rank correlation across subset checkpoints.
    """
    beta_values = np.asarray(betas, dtype=np.float64)
    if beta_values.ndim != 1 or beta_values.size == 0:
        raise ValueError("betas must be a nonempty one-dimensional array")
    if np.any(beta_values <= 0.0) or np.any(beta_values >= 1.0):
        raise ValueError("all betas must lie strictly between zero and one")
    alphas = 1.0 - beta_values
    alphas_cumprod = np.cumprod(alphas)
    return beta_values / (2.0 * alphas * (1.0 - alphas_cumprod))


def stratified_vlb_nll_numpy(
    per_timestep_mse: np.ndarray,
    timestep_values: np.ndarray,
    betas: np.ndarray,
) -> float:
    """Estimate the model-dependent per-coordinate DDPM VLB-NLL sum."""
    losses = np.asarray(per_timestep_mse, dtype=np.float64)
    timesteps = np.asarray(timestep_values, dtype=np.int64)
    beta_values = np.asarray(betas, dtype=np.float64)
    if losses.ndim != 1 or timesteps.ndim != 1 or losses.shape != timesteps.shape:
        raise ValueError("losses and timesteps must be matching 1D arrays")
    if losses.size == 0:
        raise ValueError("at least one timestep is required")
    if np.any(timesteps < 0) or np.any(timesteps >= beta_values.size):
        raise ValueError("timestep outside beta schedule")
    weights = fixed_beta_vlb_weights_numpy(beta_values)
    # Uniform stratified quadrature for the full T-term VLB sum.
    return float(beta_values.size * np.mean(weights[timesteps] * losses))
