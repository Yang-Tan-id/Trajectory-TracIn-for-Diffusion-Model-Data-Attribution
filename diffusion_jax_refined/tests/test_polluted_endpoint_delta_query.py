from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from diffusion_jax_refined.legacy_jax.traj_tracin.algorithm import (
    normalize_query_objective_name,
    query_objective_uses_next_checkpoint,
    query_scalar,
)


class _LinearEpsAdapter:
    @staticmethod
    def eps_apply(_model, params, xt, _t, _cond):
        return params * xt


class PollutedEndpointDeltaQueryTest(unittest.TestCase):
    def test_objectives_are_next_checkpoint_targets(self) -> None:
        for objective in (
            "trajectory_polluted_endpoint_next_delta_projection",
            "trajectory_polluted_endpoint_next_delta_projection_normalized",
        ):
            self.assertEqual(normalize_query_objective_name(objective), objective)
            self.assertTrue(query_objective_uses_next_checkpoint(objective))

    def test_delta_is_stop_gradient_and_normalization_only_changes_direction_scale(self) -> None:
        adapter = _LinearEpsAdapter()
        xt = jnp.asarray([[[[1.0, 2.0]]]], dtype=jnp.float32)
        t = jnp.asarray([7], dtype=jnp.int32)

        def gradient(objective: str) -> float:
            return float(
                jax.grad(
                    lambda p: query_scalar(
                        adapter,
                        None,
                        p,
                        jnp.asarray(3.0),
                        jnp.asarray(0.0),
                        xt,
                        t,
                        None,
                        objective,
                    )
                )(jnp.asarray(1.0))
            )

        raw = gradient("trajectory_polluted_endpoint_next_delta_projection")
        normalized = gradient(
            "trajectory_polluted_endpoint_next_delta_projection_normalized"
        )
        # delta=(3-1)*xt.  The normalized objective must preserve its sign and
        # differ only by the fixed output-space L2 norm of that delta.
        delta_norm = float(np.linalg.norm(2.0 * np.asarray(xt)))
        self.assertGreater(raw, 0.0)
        self.assertGreater(normalized, 0.0)
        self.assertAlmostEqual(raw / normalized, delta_norm, places=5)


if __name__ == "__main__":
    unittest.main()
