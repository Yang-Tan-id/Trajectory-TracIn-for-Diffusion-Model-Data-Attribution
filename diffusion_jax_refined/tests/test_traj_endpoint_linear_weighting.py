from __future__ import annotations

import numpy as np


def test_endpoint_linear_weights_are_normalized_per_checkpoint(monkeypatch) -> None:
    from diffusion_jax_refined.common.stage_artifact_runner import (
        _apply_traj_timestep_weighting,
    )

    monkeypatch.setenv("TRACIN_SCORE_TIMESTEP_WEIGHTING", "endpoint_linear")
    monkeypatch.setenv("TRACIN_SCORE_TIMESTEPS_TOTAL", "1000")
    weights = np.asarray([0.5, 0.5, 0.25, 0.25], dtype=np.float64)
    checkpoints = np.asarray([0, 0, 1, 1], dtype=np.int64)
    timesteps = np.asarray([999, 0, 900, 100], dtype=np.int64)

    actual = _apply_traj_timestep_weighting(weights, checkpoints, timesteps)

    np.testing.assert_allclose(actual[:2], [1.0 / 1001.0, 1000.0 / 1001.0])
    np.testing.assert_allclose(actual[2:], [0.05, 0.45])
    np.testing.assert_allclose(actual[:2].sum(), weights[:2].sum())
    np.testing.assert_allclose(actual[2:].sum(), weights[2:].sum())
