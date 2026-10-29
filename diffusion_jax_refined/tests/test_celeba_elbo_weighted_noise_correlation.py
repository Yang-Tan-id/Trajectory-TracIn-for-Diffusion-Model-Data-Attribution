import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "celeba/script/summarize_random_removal_elbo_weighted_noise.py"
SPEC = importlib.util.spec_from_file_location("elbo_noise", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CelebaElboWeightedNoiseCorrelationTest(unittest.TestCase):
    def test_weight_matches_fixed_variance_ddpm_coefficient(self):
        weights = MODULE.fixed_variance_elbo_weights(1000, 1e-4, 0.02)
        self.assertEqual(weights.shape, (1000,))
        self.assertAlmostEqual(weights[0], 0.500050005, places=8)
        self.assertGreater(weights[0], weights[-1])

    def test_pair_aggregation_uses_all_steps_and_elbo_weights(self):
        weights = np.asarray([1.0, 2.0, 4.0])
        rows = {}
        for model_id, deviations in enumerate(((1.0, 2.0), (3.0, 5.0))):
            for step, (timestep, deviation) in enumerate(zip((2, 1), deviations)):
                rows[(0, model_id, step)] = {
                    "evaluation_query_id": 0,
                    "random_model_id": model_id,
                    "step": step,
                    "timestep": timestep,
                    "predicted_noise_mse": deviation,
                    "endpoint_mse": float(model_id + 1),
                    "trajectory_mse": float(model_id + 2),
                }
        pairs, schedule = MODULE.aggregate_pairs(
            rows, [0], [0, 1], weights, expected_steps=2
        )
        self.assertEqual(len(pairs), 2)
        self.assertEqual(len(schedule), 2)
        self.assertAlmostEqual(
            pairs[0]["noise_deviation_elbo_weighted_mean"],
            (4.0 * 1.0 + 2.0 * 2.0) / 6.0,
        )
        self.assertAlmostEqual(
            pairs[0]["noise_deviation_elbo_quadrature"],
            3.0 / 2.0 * (4.0 * 1.0 + 2.0 * 2.0),
        )

    def test_missing_pair_step_is_rejected(self):
        rows = {(0, 0, 0): {
            "evaluation_query_id": 0, "random_model_id": 0, "step": 0,
            "timestep": 2, "predicted_noise_mse": 1.0,
            "endpoint_mse": 1.0, "trajectory_mse": 1.0,
        }}
        with self.assertRaisesRegex(RuntimeError, "missing row"):
            MODULE.aggregate_pairs(rows, [0], [0], np.ones(3), expected_steps=2)


if __name__ == "__main__":
    unittest.main()
