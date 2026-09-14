from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class DasMcNormalized10x10Tests(unittest.TestCase):
    def test_scalar_trajectory_norm_formula(self):
        gradients = np.asarray([[3.0, 4.0], [0.0, 5.0]], dtype=np.float64)
        residuals = np.asarray([3.0, 4.0], dtype=np.float64)
        actual_gradient = gradients.sum(axis=0) / 2.0 / np.sqrt(np.square(gradients).sum())
        actual_residual = residuals.sum() / 2.0 / np.sqrt(np.square(residuals).sum())
        np.testing.assert_allclose(actual_gradient, np.asarray([0.3, 0.9]) / np.sqrt(2.0))
        self.assertAlmostEqual(actual_residual, 0.7)

    def test_algorithm_keeps_gradient_and_residual_separate(self):
        text = (ROOT / "legacy_jax" / "das" / "algorithm.py").read_text()
        self.assertIn("DAS_AGGREGATE_MC_NORMALIZED", text)
        self.assertIn("train_gradient_sum += phi64", text)
        self.assertIn("train_residual_sum += residual64", text)
        self.assertIn("gradient_denom[:, None]", text)
        self.assertIn("stage_residuals.append(aggregated_residuals)", text)
        self.assertIn("aggregated_gradients.T @ aggregated_gradients", text)

    def test_rtx_launchers_use_isolated_namespace(self):
        launcher_root = ROOT / "3dshapes" / "tacc" / "rtx_small"
        train = (launcher_root / "run_das_mc_normalized10x10_train_rtx_small.sh").read_text()
        query = (launcher_root / "run_das_mc_normalized10x10_query_score_rtx_small.sh").read_text()
        lds = (launcher_root / "run_das_mc_normalized10x10_lds_cached.sh").read_text()
        self.assertIn("DAS_AGGREGATE_MC_NORMALIZED=1", train)
        self.assertIn("das_mc_normalized10x10", train)
        self.assertIn("--aggregate-mc-normalized", query)
        self.assertIn("--artifact-namespace mc_normalized10x10", query)
        self.assertIn("--artifact-namespace mc_normalized10x10", lds)


if __name__ == "__main__":
    unittest.main()
