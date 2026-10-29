from __future__ import annotations

import unittest

import numpy as np

from diffusion_jax_refined.common.predicted_noise_kl import (
    predicted_noise_kl_numpy,
    query_probe_seed,
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


if __name__ == "__main__":
    unittest.main()
