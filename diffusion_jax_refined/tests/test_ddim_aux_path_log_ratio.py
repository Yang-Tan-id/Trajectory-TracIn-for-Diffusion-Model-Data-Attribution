from __future__ import annotations

import unittest

import numpy as np

from diffusion_jax_refined.common.ddim_aux_path_log_ratio import (
    ddim_aux_coefficients,
    path_log_ratio_numpy,
)


class DdimAuxPathLogRatioTest(unittest.TestCase):
    def test_coefficients_are_positive_and_exclude_final_transition(self) -> None:
        betas = np.linspace(1e-4, 2e-2, 1000)
        abar = np.cumprod(1.0 - betas)
        timesteps = np.asarray([999, 799, 599, 399, 199, 19])
        t_values, s_values, b0, b, variance = ddim_aux_coefficients(
            abar, timesteps, eta=1.0
        )
        np.testing.assert_array_equal(t_values, timesteps[:-1])
        np.testing.assert_array_equal(s_values, timesteps[1:])
        self.assertEqual(len(variance), len(timesteps) - 1)
        self.assertTrue(np.all(variance > 0.0))
        self.assertTrue(np.all(variance <= 1.0 - abar[s_values] + 1e-12))
        self.assertTrue(np.all(np.isfinite(b0)))
        self.assertTrue(np.all(np.isfinite(b)))

    def test_expansion_matches_direct_gaussian_residual_difference(self) -> None:
        rng = np.random.default_rng(7)
        eps_ref = rng.normal(size=(3, 2, 4, 4, 1))
        eps_target = eps_ref + rng.normal(scale=0.2, size=eps_ref.shape)
        b0 = np.asarray([-0.3, -0.2, -0.1])
        b = np.asarray([-0.4, -0.25, -0.15])
        variance = np.asarray([0.2, 0.15, 0.1])
        total, squared, cross = path_log_ratio_numpy(
            eps_ref, eps_target, b0, b, variance
        )
        delta = eps_target - eps_ref
        shape = (3, 1, 1, 1, 1)
        ref_residual = (b0 - b).reshape(shape) * eps_ref
        target_residual = ref_residual - b.reshape(shape) * delta
        axes = tuple(range(1, eps_ref.ndim))
        direct = np.sum(
            (
                np.mean(target_residual**2, axis=axes)
                - np.mean(ref_residual**2, axis=axes)
            )
            / (2.0 * variance)
        )
        self.assertAlmostEqual(total, direct, places=12)
        self.assertAlmostEqual(total, squared + cross, places=12)

    def test_identical_models_have_zero_log_ratio(self) -> None:
        eps_ref = np.ones((2, 1, 2, 2, 1))
        result = path_log_ratio_numpy(
            eps_ref,
            eps_ref.copy(),
            np.asarray([-0.2, -0.1]),
            np.asarray([-0.3, -0.15]),
            np.asarray([0.1, 0.05]),
        )
        self.assertEqual(result, (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
