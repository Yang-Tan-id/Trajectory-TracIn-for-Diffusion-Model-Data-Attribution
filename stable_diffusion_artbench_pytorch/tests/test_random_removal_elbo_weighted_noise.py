from __future__ import annotations

import unittest

import numpy as np

from sd_artbench.summarize_random_removal_elbo_weighted_noise import (
    aggregate_pairs,
    fixed_variance_elbo_weights_from_betas,
)


class RandomRemovalElboWeightedNoiseTest(unittest.TestCase):
    def test_weights_follow_supplied_scheduler_betas(self) -> None:
        betas = np.asarray([0.1, 0.2], dtype=np.float64)
        weights = fixed_variance_elbo_weights_from_betas(betas)
        expected0 = 0.1 / (2.0 * 0.9 * (1.0 - 0.9))
        expected1 = 0.2 / (2.0 * 0.8 * (1.0 - 0.9 * 0.8))
        np.testing.assert_allclose(weights, [expected0, expected1])

    def test_aggregate_filters_exact_query_model_cross_product(self) -> None:
        weights = np.asarray([1.0, 2.0, 4.0])
        rows = {}
        for query in (0, 1):
            for model in (0, 1):
                for step, timestep in enumerate((2, 1)):
                    rows[(query, model, step)] = {
                        "evaluation_query_index": query,
                        "random_model_index": model,
                        "step": step,
                        "timestep": timestep,
                        "predicted_noise_mse": float(1 + query + model + step),
                        "endpoint_mse": float(model + 1),
                        "trajectory_mse": float(model + 2),
                    }
        pairs, schedule = aggregate_pairs(
            rows, [0], [0, 1], weights, expected_steps=2
        )
        self.assertEqual(len(pairs), 2)
        self.assertEqual({row["evaluation_query_index"] for row in pairs}, {0})
        self.assertEqual(len(schedule), 2)
        self.assertAlmostEqual(
            pairs[0]["noise_deviation_elbo_weighted_mean"],
            (4.0 * 1.0 + 2.0 * 2.0) / 6.0,
        )

    def test_missing_step_fails(self) -> None:
        rows = {(0, 0, 0): {
            "evaluation_query_index": 0, "random_model_index": 0,
            "step": 0, "timestep": 1, "predicted_noise_mse": 1.0,
            "endpoint_mse": 1.0, "trajectory_mse": 1.0,
        }}
        with self.assertRaisesRegex(RuntimeError, "missing row"):
            aggregate_pairs(rows, [0], [0], np.ones(2), expected_steps=2)


if __name__ == "__main__":
    unittest.main()
