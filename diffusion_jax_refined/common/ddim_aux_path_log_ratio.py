"""Coefficients for a positive-variance auxiliary kernel on a DDIM path."""

from __future__ import annotations

import numpy as np


def ddim_aux_coefficients(
    alphas_cumprod: np.ndarray,
    timesteps: np.ndarray,
    *,
    eta: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(t, s, b0, b, v)`` for positive-variance DDIM transitions.

    ``timesteps`` contains saved reference states in descending order.  The
    unrecorded final transition to the deterministic endpoint is excluded.
    """
    abar = np.asarray(alphas_cumprod, dtype=np.float64).reshape(-1)
    steps = np.asarray(timesteps, dtype=np.int64).reshape(-1)
    if len(steps) < 2:
        raise ValueError("at least two saved DDIM timesteps are required")
    if not np.all(steps[:-1] > steps[1:]):
        raise ValueError("DDIM timesteps must be strictly descending")
    if not 0.0 < float(eta) <= 1.0:
        raise ValueError("eta must be in (0, 1]")
    if steps.min() < 0 or steps.max() >= len(abar):
        raise ValueError("timestep lies outside alphas_cumprod")

    t_values = steps[:-1]
    s_values = steps[1:]
    abar_t = abar[t_values]
    abar_s = abar[s_values]
    a = np.sqrt(abar_s / abar_t)
    base_variance = (
        (1.0 - abar_s) / (1.0 - abar_t)
        * (1.0 - abar_t / abar_s)
    )
    variance = float(eta) ** 2 * base_variance
    if np.any(~np.isfinite(variance)) or np.any(variance <= 0.0):
        raise ValueError("auxiliary DDIM variance must be finite and positive")
    remaining = 1.0 - abar_s - variance
    if np.any(remaining < -1e-10):
        raise ValueError("auxiliary variance exceeds 1 - alpha_bar_s")
    remaining = np.maximum(remaining, 0.0)
    common = a * np.sqrt(1.0 - abar_t)
    b0 = np.sqrt(1.0 - abar_s) - common
    b = np.sqrt(remaining) - common
    return t_values, s_values, b0, b, variance


def path_log_ratio_numpy(
    eps_ref: np.ndarray,
    eps_target: np.ndarray,
    b0: np.ndarray,
    b: np.ndarray,
    variance: np.ndarray,
) -> tuple[float, float, float]:
    """Return normalized total, squared, and cross path-log-ratio terms."""
    reference = np.asarray(eps_ref, dtype=np.float64)
    target = np.asarray(eps_target, dtype=np.float64)
    if reference.shape != target.shape or reference.shape[0] != len(b):
        raise ValueError("prediction and coefficient shapes do not match")
    delta = target - reference
    axes = tuple(range(1, delta.ndim))
    squared_error = np.mean(delta**2, axis=axes)
    cross_inner = np.mean(reference * delta, axis=axes)
    squared = np.sum((b**2 / (2.0 * variance)) * squared_error)
    cross = np.sum(-(b * (b0 - b) / variance) * cross_inner)
    return float(squared + cross), float(squared), float(cross)
