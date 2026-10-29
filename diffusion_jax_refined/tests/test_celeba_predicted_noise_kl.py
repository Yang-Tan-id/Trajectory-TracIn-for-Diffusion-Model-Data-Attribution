from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from diffusion_jax_refined.common.predicted_noise_kl import (
    fixed_beta_vlb_weights_numpy,
    predicted_noise_kl_numpy,
    query_probe_seed,
    stratified_vlb_nll_numpy,
    uniform_timesteps,
)


class TestPredictedNoiseKl(unittest.TestCase):
    def test_fifty_uniform_timesteps_cover_full_training_range(self):
        values = uniform_timesteps(1000, 50)
        self.assertEqual(len(values), 50)
        self.assertEqual(int(values[0]), 999)
        self.assertEqual(int(values[-1]), 0)
        self.assertEqual(len(np.unique(values)), 50)
        self.assertTrue(np.all(np.diff(values) < 0))

    def test_fixed_variance_gaussian_kl_proxy(self):
        full = np.zeros((2, 3), dtype=np.float32)
        subset = np.full((2, 3), 2.0, dtype=np.float32)
        self.assertAlmostEqual(predicted_noise_kl_numpy(subset, full), 2.0)
        self.assertEqual(predicted_noise_kl_numpy(full, full), 0.0)

    def test_query_probe_seeds_are_deterministic_and_distinct(self):
        seeds = [query_probe_seed(2026, query_id) for query_id in range(10)]
        self.assertEqual(seeds[0], 2026)
        self.assertEqual(len(set(seeds)), 10)
        self.assertEqual(seeds, [query_probe_seed(2026, q) for q in range(10)])

    def test_fixed_beta_vlb_weights_match_closed_form(self):
        betas = np.asarray([0.1, 0.2], dtype=np.float64)
        weights = fixed_beta_vlb_weights_numpy(betas)
        alphas = 1.0 - betas
        abar = np.cumprod(alphas)
        expected = betas / (2.0 * alphas * (1.0 - abar))
        np.testing.assert_allclose(weights, expected)

    def test_stratified_vlb_uses_schedule_weights(self):
        betas = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        timesteps = np.asarray([3, 1], dtype=np.int32)
        losses = np.asarray([2.0, 5.0], dtype=np.float64)
        weights = fixed_beta_vlb_weights_numpy(betas)
        expected = len(betas) * np.mean(weights[timesteps] * losses)
        self.assertAlmostEqual(
            stratified_vlb_nll_numpy(losses, timesteps, betas), expected
        )

    def test_forward_probe_code_preserves_explicit_batch_axis(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "legacy_jax"
            / "LDS"
            / "DM_cifar_lds.py"
        ).read_text()
        probe_start = source.index("def _build_predicted_noise_kl_probes")
        probe_stop = source.index("def _predicted_noise_kl_objective", probe_start)
        probe_source = source[probe_start:probe_stop]
        self.assertIn("x0 = self.x0_ref", probe_source)
        self.assertNotIn("x0 = self.x0_ref[0]", probe_source)

    def test_celeba_exporter_can_select_one_cache_namespace(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "celeba"
            / "script"
            / "export_attr40_query100_lds_csv.py"
        ).read_text()
        self.assertIn('"--cache-namespace"', source)
        self.assertIn("query_root = query_root / args.cache_namespace", source)


if __name__ == "__main__":
    unittest.main()
