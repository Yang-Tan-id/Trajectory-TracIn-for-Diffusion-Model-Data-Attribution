from __future__ import annotations

from pathlib import Path
import importlib.util
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
SIGNED_SCORE_PIPELINE = (
    ROOT
    / "3dshapes"
    / "tacc"
    / "rtx_small"
    / "run_loss_direction_residual_rms_signed_score_rtx_small.sh"
)
ORIGINAL_F_FOUR_NORM_PIPELINE = (
    ROOT
    / "3dshapes"
    / "tacc"
    / "rtx_small"
    / "run_loss_direction_residual_rms_original_f_four_norm_q0_rtx_small.sh"
)
ORIGINAL_F_TIMESTAMP_CROSSFIT = (
    ROOT
    / "3dshapes"
    / "tacc"
    / "rtx_small"
    / "run_original_f_timestamp_sign_crossfit_bad_queries_rtx_small.sh"
)
ORIGINAL_F_CHECKPOINT_CROSSFIT = (
    ROOT
    / "3dshapes"
    / "tacc"
    / "rtx_small"
    / "run_original_f_checkpoint_sign_crossfit_bad_queries_rtx_small.sh"
)
F_NEXT_SQUARED_PIPELINE = (
    ROOT
    / "3dshapes"
    / "tacc"
    / "rtx_small"
    / "run_f_next_dot_squared_score_rtx_small.sh"
)
SCORER = ROOT / "3dshapes" / "script" / "run_expected_residual_jacobian_scores.py"
LDS_DRIVER = ROOT / "3dshapes" / "script" / "run_traj_tracin_lds_cached.py"
ENSEMBLE_DRIVER = (
    ROOT / "3dshapes" / "script" / "materialize_f_next_linear_square_ensemble.py"
)
ENSEMBLE_LAUNCHER = (
    ROOT
    / "3dshapes"
    / "tacc"
    / "rtx_small"
    / "run_f_next_linear_square_z50_lds_rtx_small.sh"
)
SHARED_ORTHOGONAL_PIPELINE = (
    ROOT
    / "3dshapes"
    / "tacc"
    / "rtx_small"
    / "run_predicted_noise_shared_orthogonal_probe4_pipeline_rtx_small.sh"
)


class LossDirectionResidualRmsTest(unittest.TestCase):
    def test_shared_orthogonal_probes_are_fixed_across_trajectory(self) -> None:
        algorithm = ALGORITHM.read_text()
        query_driver = (
            ROOT / "3dshapes" / "script" / "run_traj_tracin_queries_and_scores.py"
        ).read_text()
        pipeline = SHARED_ORTHOGONAL_PIPELINE.read_text()

        self.assertIn("def shared_orthogonal_predicted_noise_probes(", algorithm)
        self.assertIn('jnp.linalg.qr(gaussian, mode="reduced")', algorithm)
        self.assertIn("shared_orthogonal_probe_bank = None", algorithm)
        self.assertIn("jnp.broadcast_to(", algorithm)
        self.assertIn('"--predicted-noise-probe-mode"', query_driver)
        self.assertIn("for probe_index in 0 1 2 3", pipeline)
        self.assertIn("--predicted-noise-probe-mode shared_orthogonal", pipeline)
        self.assertIn("--predicted-noise-probe-count", pipeline)
        self.assertIn("--expected-query-probe-mode shared_orthogonal", pipeline)
        self.assertIn("predicted_noise_shared_orthogonal_probe4_linear", pipeline)
        self.assertIn("--prediction-sign=1", pipeline)
        self.assertIn("--prediction-sign=-1", pipeline)

    def test_extended_fixed_bank_preserves_historical_four(self) -> None:
        algorithm = ALGORITHM.read_text()
        query_driver = (
            ROOT / "3dshapes" / "script" / "run_traj_tracin_queries_and_scores.py"
        ).read_text()
        independent_launcher = (
            ROOT
            / "3dshapes"
            / "tacc"
            / "rtx_small"
            / "run_predicted_noise_probe9_12_queries_rtx_small.sh"
        ).read_text()
        fixed_launcher = (
            ROOT
            / "3dshapes"
            / "tacc"
            / "rtx_small"
            / "run_predicted_noise_fixed_probe5_8_queries_rtx_small.sh"
        ).read_text()

        self.assertIn(
            "def extended_shared_orthogonal_predicted_noise_probes(", algorithm
        )
        self.assertIn("base = shared_orthogonal_predicted_noise_probes(", algorithm)
        self.assertIn("candidates - base_unit.T @ (base_unit @ candidates)", algorithm)
        self.assertIn('"shared_orthogonal_extended"', query_driver)
        self.assertIn("for probe_index in 8 9 10 11", independent_launcher)
        self.assertIn("for probe_index in 4 5 6 7", fixed_launcher)
        self.assertIn(
            "--predicted-noise-probe-mode shared_orthogonal_extended",
            fixed_launcher,
        )
        self.assertIn("--predicted-noise-probe-count 8", fixed_launcher)

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
        self.assertIn('"raw": predicted_raw', scorer)
        self.assertIn(
            'float(weight) * predicted_values[query_variant]', scorer
        )
        self.assertNotIn("float(weight) ** 2 * predicted_raw", scorer)
        self.assertNotIn('totals[("predicted", query_variant)] / predicted_weight', scorer)

    def test_signed_predicted_noise_score_reuses_saved_artifacts(self) -> None:
        scorer = SCORER.read_text()
        self.assertIn('"--predicted-contraction"', scorer)
        self.assertIn('choices=("squared", "signed")', scorer)
        self.assertIn("raw_values = dots", scorer)
        self.assertIn("query_l2_values = dots / query_norms[None, :]", scorer)

        launcher = SIGNED_SCORE_PIPELINE.read_text()
        self.assertNotIn("run_traj_tracin_queries_and_scores.py", launcher)
        self.assertNotIn("run_traj_tracin_loss_direction_residual_rms_train", launcher)
        self.assertIn("--predicted-contraction signed", launcher)
        self.assertIn(
            "loss_direction_residual_rms_predicted_noise_signed", launcher
        )

        lds_driver = LDS_DRIVER.read_text()
        self.assertIn(
            '"loss_direction_residual_rms_predicted_noise_signed": '
            '"traj_tracin_loss_direction_residual_rms_signed_predicted_noise"',
            lds_driver,
        )

    def test_original_f_four_l2_normalizations_are_materialized(self) -> None:
        scorer = SCORER.read_text()
        self.assertIn('"train_l2": train_l2_original', scorer)
        self.assertIn('"query_train_l2": query_train_l2_original', scorer)
        self.assertIn('"train_l2": "score_train_l2_normalized"', scorer)
        self.assertIn(
            '"query_train_l2": "score_query_train_l2_normalized"', scorer
        )
        self.assertIn('"--score-variants"', scorer)

        lds_driver = LDS_DRIVER.read_text()
        self.assertIn('FOUR_NORM_EXPECTED_SCHEMES', lds_driver)
        self.assertIn('"loss_direction_residual_rms_original_f"', lds_driver)

        launcher = ORIGINAL_F_FOUR_NORM_PIPELINE.read_text()
        self.assertIn("--skip-predicted", launcher)
        self.assertIn("TRAIN_NAMESPACE=traj_tracin", launcher)
        self.assertIn("--score-variants train_l2,query_train_l2", launcher)
        self.assertIn("--query-ids 0", launcher)
        self.assertIn("loss_direction_residual_rms_original_f", launcher)

    def test_original_f_timestamp_crossfit_reuses_direct_loss_train_parts(self) -> None:
        analyzer = (
            ROOT
            / "3dshapes"
            / "script"
            / "analyze_original_f_timestamp_sign_crossfit.py"
        ).read_text()
        launcher = ORIGINAL_F_TIMESTAMP_CROSSFIT.read_text()

        self.assertIn('payload.get(', analyzer)
        self.assertIn('"raw_projected_expected_loss_gradient"', analyzer)
        self.assertIn("[components saved]", analyzer)
        self.assertNotIn(
            "load_target_data(cache_group(eval_root), score_indices, targets=TARGETS)",
            analyzer,
        )
        self.assertIn("--train-namespace traj_tracin", launcher)
        self.assertIn(
            "--train-feature-semantics raw_projected_expected_loss_gradient",
            launcher,
        )
        self.assertNotIn("TRAJ_TRACIN_TRAIN_REUSE_GRADIENT_RESIDUAL_RMS", launcher)
        self.assertNotIn("--component-dir", launcher)
        self.assertNotIn(
            "run_traj_tracin_loss_direction_residual_rms_train_rtx_small.sh",
            launcher,
        )

    def test_original_f_checkpoint_crossfit_is_cached_and_structured(self) -> None:
        analyzer = (
            ROOT
            / "3dshapes"
            / "script"
            / "analyze_original_f_checkpoint_sign_crossfit.py"
        ).read_text()
        launcher = ORIGINAL_F_CHECKPOINT_CROSSFIT.read_text()

        self.assertIn('"single_change_point"', analyzer)
        self.assertIn('"five_bins"', analyzer)
        self.assertIn('"ten_bins"', analyzer)
        self.assertIn('"individual_coordinate"', analyzer)
        self.assertIn("crossfit_global_baseline_mean_percent", analyzer)
        self.assertIn("crossfit_beat_global_fraction", analyzer)
        self.assertIn("checkpoint_components.npz", analyzer)
        self.assertIn("[cached]", launcher)
        self.assertNotIn("01_train_datapoint_gradient.py", launcher)

    def test_original_train_f_next_squared_score_is_score_only(self) -> None:
        scorer = SCORER.read_text()
        self.assertIn('"--original-contraction"', scorer)
        self.assertIn("raw_original = jnp.square(dots)", scorer)
        self.assertIn('"--skip-predicted"', scorer)
        self.assertIn('"raw_projected_expected_loss_gradient"', scorer)

        launcher = F_NEXT_SQUARED_PIPELINE.read_text()
        self.assertIn("--train-namespace \"${TRAIN_NAMESPACE}\"", launcher)
        self.assertIn("TRAIN_NAMESPACE=traj_tracin", launcher)
        self.assertIn("--original-contraction squared", launcher)
        self.assertIn("--skip-predicted", launcher)
        self.assertNotIn("01_train_datapoint_gradient.py", launcher)
        self.assertNotIn("run_traj_tracin_queries_and_scores.py", launcher)
        self.assertIn("--score-schemes f_next_dot_squared", launcher)

    def test_linear_square_ensemble_standardizes_each_component(self) -> None:
        spec = importlib.util.spec_from_file_location("f_next_ensemble", ENSEMBLE_DRIVER)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        linear = np.asarray([1.0, 2.0, 8.0])
        squared = np.asarray([100.0, 200.0, 300.0])
        linear_z, _, _ = module.zscore(linear, 1e-12)
        squared_z, _, _ = module.zscore(squared, 1e-12)
        ensemble = 0.5 * linear_z + 0.5 * squared_z

        self.assertAlmostEqual(float(np.mean(linear_z)), 0.0)
        self.assertAlmostEqual(float(np.std(linear_z)), 1.0)
        self.assertAlmostEqual(float(np.mean(squared_z)), 0.0)
        np.testing.assert_allclose(ensemble, 0.5 * (linear_z + squared_z))

        launcher = ENSEMBLE_LAUNCHER.read_text()
        self.assertIn("--alpha 0.5", launcher)
        self.assertIn("--score-schemes f_next_linear_square_z50", launcher)

    def test_cached_lds_supports_positive_prediction_direction(self) -> None:
        source = LDS_DRIVER.read_text()
        self.assertIn('"--prediction-sign"', source)
        self.assertIn('choices=(-1.0, 1.0)', source)
        self.assertIn(
            'sign_tag = "p1" if args.prediction_sign > 0 else "m1"', source
        )
        self.assertIn(
            "sum_scores(kept, score_map, args.prediction_sign)", source
        )


if __name__ == "__main__":
    unittest.main()
