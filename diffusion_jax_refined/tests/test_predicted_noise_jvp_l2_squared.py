from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class PredictedNoiseJvpL2SquaredTests(unittest.TestCase):
    def test_squared_batched_dot_matches_explicit_rows(self):
        rng = np.random.default_rng(42)
        train = rng.normal(size=(7, 5)).astype(np.float32)
        query = rng.normal(size=(3, 5)).astype(np.float32)
        actual = np.square(train @ query.T).T
        expected = np.asarray(
            [[float(np.dot(train[i], query[q])) ** 2 for i in range(7)] for q in range(3)]
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_original_four_normalization_variants(self):
        train = np.asarray([[3.0, 4.0], [1.0, 0.0]], dtype=np.float64)
        query = np.asarray([[0.0, 2.0]], dtype=np.float64)
        dot = train @ query.T
        raw = np.square(dot)
        query_l2 = np.square(dot / np.linalg.norm(query, axis=1)[None, :])
        train_l2 = np.square(dot / np.linalg.norm(train, axis=1)[:, None])
        both_l2 = np.square(
            dot
            / np.linalg.norm(train, axis=1)[:, None]
            / np.linalg.norm(query, axis=1)[None, :]
        )
        np.testing.assert_allclose(raw[:, 0], [64.0, 0.0])
        np.testing.assert_allclose(query_l2[:, 0], [16.0, 0.0])
        np.testing.assert_allclose(train_l2[:, 0], [64.0 / 25.0, 0.0])
        np.testing.assert_allclose(both_l2[:, 0], [16.0 / 25.0, 0.0])

    def test_query_probe_is_scalar_and_raw_projected(self):
        text = (ROOT / "legacy_jax" / "traj_tracin" / "algorithm.py").read_text()
        self.assertIn('"trajectory_predicted_noise_probe"', text)
        self.assertIn("def predicted_noise_probe_key", text)
        self.assertNotIn("make_jax_key(", text)
        self.assertIn("jnp.sum(eps.astype(jnp.float32) * output_probe.astype(jnp.float32)) / normalizer", text)
        self.assertIn("stage_features.append(np.asarray(projector(one_grad), dtype=np.float32))", text)

    def test_rtx_pipeline_reuses_original_train_parts(self):
        driver = (ROOT / "3dshapes" / "script" / "run_predicted_noise_jvp_l2_squared.py").read_text()
        launcher = (
            ROOT
            / "3dshapes"
            / "tacc"
            / "rtx_small"
            / "run_predicted_noise_jvp_l2_squared_rtx_small.sh"
        ).read_text()
        self.assertIn('/ "traj_tracin"', driver)
        self.assertIn('f"ckpt_{ckpt_i:04d}.npz"', driver)
        self.assertIn('"score_query_normalized"', driver)
        self.assertIn('"score_train_l2_normalized"', driver)
        self.assertIn('"score_query_train_l2_normalized"', driver)
        self.assertIn("weight_squared = float(weight) ** 2", driver)
        self.assertIn("trajectory_predicted_noise_probe", driver)
        self.assertIn('f"run_{run_id}"', driver)
        self.assertIn("--cleanup-query-artifacts", launcher)
        self.assertIn('--run-id "${SLURM_JOB_ID}"', launcher)
        self.assertIn("--shard-count 2", launcher)
        self.assertIn("--score-schemes predicted_noise_jvp_l2_squared", launcher)


if __name__ == "__main__":
    unittest.main()
