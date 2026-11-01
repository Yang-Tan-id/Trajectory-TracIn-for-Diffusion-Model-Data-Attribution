"""Probe-pair construction shared by simple-loss LDS evaluators."""

from __future__ import annotations

from collections.abc import Sequence


def manual_simple_loss_probe_pairs(
    timesteps: Sequence[int],
    noise_seeds: Sequence[int],
    *,
    paired: bool,
) -> tuple[tuple[int, int], ...]:
    """Return paired probes or the explicitly requested Cartesian grid."""
    timestep_values = tuple(int(value) for value in timesteps)
    seed_values = tuple(int(value) for value in noise_seeds)
    if not timestep_values or not seed_values:
        raise ValueError("manual simple-loss probes must be nonempty")
    if paired:
        if len(timestep_values) != len(seed_values):
            raise ValueError(
                "paired simple-loss timesteps/noise seeds must have equal length"
            )
        return tuple(zip(timestep_values, seed_values, strict=True))
    return tuple(
        (timestep, seed)
        for timestep in timestep_values
        for seed in seed_values
    )
