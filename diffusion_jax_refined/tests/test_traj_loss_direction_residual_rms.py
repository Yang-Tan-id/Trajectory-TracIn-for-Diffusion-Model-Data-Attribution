from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ALGORITHM = ROOT / "legacy_jax" / "traj_tracin" / "algorithm.py"
LAUNCHER = (
    ROOT
    / "3dshapes"
    / "tacc"
    / "rtx_small"
    / "run_traj_tracin_loss_direction_residual_rms_train_rtx_small.sh"
)
PIPELINE = (
    ROOT
    / "3dshapes"
    / "tacc"
    / "rtx_small"
    / "run_loss_direction_residual_rms_pipeline_rtx_small.sh"
)
SCORER = ROOT / "3dshapes" / "script" / "run_expected_residual_jacobian_scores.py"


class LossDirectionResidualRmsTest(unittest.TestCase):
    def test_feature_keeps_rms_and_gradient_direction(self) -> None:
        gradient = np.asarray([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
        residual_rms = np.asarray([2.0, 7.0], dtype=np.float32)
        eps = 1e-8

        feature = gradient / np.maximum(
            np.linalg.norm(gradient, axis=1, keepdims=True), eps
        )
        feature *= residual_rms[:, None]

        np.testing.assert_allclose(feature[0], np.asarray([1.2, 1.6]), rtol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(feature[0])), 2.0, places=6)
        np.testing.assert_array_equal(feature[1], np.zeros(2, dtype=np.float32))

    def test_engine_reuses_source_gradient_and_matching_rng(self) -> None:
        source = ALGORITHM.read_text()
        self.assertIn("TRAJ_TRACIN_TRAIN_REUSE_GRADIENT_RESIDUAL_RMS", source)
        self.assertIn("TRAJ_TRACIN_SOURCE_TRAIN_ARTIFACT", source)
        self.assertIn("source_features", source)
        self.assertIn("unit_source[start:end] * rms[:, None]", source)
        self.assertIn("700_000 * (ckpt_i + 1)", source)
        self.assertIn("10_000 * snap_id", source)
        self.assertIn("residual_rms=residual_rms_terms", source)
        self.assertIn(
            "unit_projected_expected_loss_gradient_times_matching_mc_residual_rms",
            source,
        )

    def test_rtx_launcher_is_resumable_and_forward_only(self) -> None:
        source = LAUNCHER.read_text()
        self.assertIn("#SBATCH -p rtx-small", source)
        self.assertIn("TRAJ_TRACIN_TRAIN_REUSE_GRADIENT_RESIDUAL_RMS=1", source)
        self.assertIn("TRAJ_TRACIN_SOURCE_TRAIN_ARTIFACT", source)
        self.assertIn("TRAJ_TRACIN_CKPT_SHARD_COUNT=2", source)
        self.assertIn("TRAJ_TRACIN_SKIP_STAGE_MERGE=1", source)
        self.assertIn("No train backward", source)
        self.assertIn("merged duplicate intentionally omitted", source)

    def test_pipeline_uses_four_predicted_noise_query_probes(self) -> None:
        source = PIPELINE.read_text()
        self.assertIn("for probe_index in 0 1 2 3", source)
        self.assertIn("--predicted-num-probes 4", source)
        self.assertIn("loss_direction_residual_rms_original_f", source)
        self.assertIn("loss_direction_residual_rms_predicted_noise", source)

        scorer = SCORER.read_text()
        self.assertIn("load_predicted_query_probes", scorer)
        self.assertIn("/ float(args.predicted_num_probes)", scorer)
        self.assertIn("jnp.square(dots)", scorer)


if __name__ == "__main__":
    unittest.main()
