from pathlib import Path
import sys
import unittest
import itertools

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "3dshapes" / "script"))

from run_predicted_noise_jvp_l2_squared import (
    reduce_checkpoint_timestamp_means,
    reduce_final_probe_scores,
    reduce_probe_rms,
    reduce_probe_median_absolute,
    reduce_timestamp_checkpoint_sums,
)
from materialize_f_next_final_score_squared import square_final_scores
from analyze_predicted_noise_angle_oriented_scores import aggregate_oriented_queries
from analyze_predicted_noise_probe24_term_winners import select_nearest_probe_scores
from analyze_nearest_train_probe_predicted_noise_relation import selected_probe_indices
from analyze_nearest_train_probe_predicted_noise_relation import OUTPUT_SELECTIONS


class PredictedNoiseJvpL2SquaredTests(unittest.TestCase):
    def test_three_output_direction_selection_rules_exist(self):
        self.assertEqual(
            {
                "current_noise_direction": "cosine_to_current_predicted_noise",
                "next_noise_direction": "cosine_to_next_predicted_noise",
                "delta_noise_direction": "cosine_to_next_predicted_noise_delta",
            },
            OUTPUT_SELECTIONS,
        )

    def test_all_query_delta_launcher_uses_fixed_full_query_list(self):
        launcher = (
            ROOT
            / "3dshapes"
            / "tacc"
            / "rtx_small"
            / "run_delta_noise_direction_all_queries_rtx_small.sh"
        ).read_text()
        self.assertIn("0,1,2,3,4,5,6,7,8,9", launcher)
        self.assertIn("predicted_noise_output_next_original12", launcher)
        self.assertIn("predicted_noise_output_next_fresh12", launcher)
        self.assertIn("[phase 4/4]", launcher)

    def test_nearest_probe_indices_are_per_datapoint(self):
        scores = np.asarray(
            [[-0.9, 0.4, 0.2], [-0.1, -0.3, -0.2]], dtype=np.float32
        )
        np.testing.assert_array_equal(
            selected_probe_indices(scores, "nearest_direction"), [1, 0]
        )
        np.testing.assert_array_equal(
            selected_probe_indices(scores, "nearest_axis_signed"), [0, 1]
        )

    def test_per_datapoint_nearest_probe_reductions(self):
        scores = np.asarray(
            [[-0.9, 0.4, 0.2], [-0.1, -0.3, -0.2]], dtype=np.float32
        )
        reduced = select_nearest_probe_scores(scores)
        np.testing.assert_allclose(reduced["nearest_direction"], [0.4, -0.1])
        np.testing.assert_allclose(reduced["nearest_axis_signed"], [-0.9, -0.3])

    def test_next_checkpoint_output_alignment_contract(self):
        algorithm = (
            ROOT / "legacy_jax" / "traj_tracin" / "algorithm.py"
        ).read_text()
        analyzer = (
            ROOT
            / "3dshapes"
            / "script"
            / "analyze_predicted_noise_probe24_output_alignment.py"
        ).read_text()

        self.assertIn("TRAJ_TRACIN_PROBE_ALIGNMENT_NEXT_CHECKPOINT", algorithm)
        self.assertIn("next_checkpoint_delta_probe_cosines", algorithm)
        self.assertIn("delta_eps=eps(params[c+1],x_t)-eps(params[c],x_t)", algorithm)
        self.assertIn("cosine_to_next_predicted_noise_delta", analyzer)
        self.assertIn("mean_lds_vs_abs_delta_cosine_spearman_percent", analyzer)

    def test_angle_oriented_queries_aggregate_before_normalization(self):
        query = np.asarray(
            [
                [[1.0, 0.0]],
                [[0.0, 2.0]],
                [[-1.0, 1.0]],
            ],
            dtype=np.float32,
        )
        scalar = np.asarray([[2.0], [-3.0], [0.5]], dtype=np.float32)
        sign = aggregate_oriented_queries(query, scalar, (0, 1, 2), "angle_sign")
        weighted = aggregate_oriented_queries(
            query, scalar, (0, 1, 2), "angle_weighted"
        )
        np.testing.assert_allclose(sign, [[0.0, -1.0 / 3.0]])
        np.testing.assert_allclose(weighted, [[0.5, -11.0 / 6.0]])

    def test_probe_independence_audit_replays_exact_key_dimensions(self):
        audit = (
            ROOT
            / "3dshapes"
            / "script"
            / "analyze_predicted_noise_probe_independence.py"
        ).read_text()
        algorithm = (ROOT / "legacy_jax" / "traj_tracin" / "algorithm.py").read_text()

        for token in (
            "0x50524F42",
            "checkpoint_index",
            "timestep",
            "snapshot_position",
            "probe_index",
        ):
            self.assertIn(token, audit)
            self.assertIn(token, algorithm)
        self.assertIn('default=(1, 64, 64, 3)', audit)
        self.assertIn('global_gram += probes @ probes.T', audit)
        self.assertIn('dc_z[term_id] = probes.sum(axis=1) / term_norms', audit)
        self.assertIn('1.0 / math.sqrt(dimension * term_count)', audit)

    def test_squared_batched_dot_matches_explicit_rows(self):
        rng = np.random.default_rng(42)
        train = rng.normal(size=(7, 5)).astype(np.float32)
        query = rng.normal(size=(3, 5)).astype(np.float32)
        actual = np.square(train @ query.T).T
        expected = np.asarray(
            [[float(np.dot(train[i], query[q])) ** 2 for i in range(7)] for q in range(3)]
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_absolute_contraction_is_probe_sign_invariant(self):
        train = np.asarray([[1.0, 2.0], [-3.0, 1.0]], dtype=np.float64)
        query = np.asarray([[2.0, -1.0], [0.5, 4.0]], dtype=np.float64)
        score = np.abs(train @ query.T)
        flipped = np.abs(train @ (-query).T)
        np.testing.assert_allclose(score, flipped)

    def test_probe_rms_is_sign_invariant_and_reduces_before_trajectory_sum(self):
        probe_scores = np.asarray([3.0, -4.0, 0.0, 0.0], dtype=np.float64)
        mean_squares = np.mean(np.square(probe_scores))
        actual = reduce_probe_rms(mean_squares, eps=0.0)
        flipped = reduce_probe_rms(
            np.mean(np.square(-probe_scores)),
            eps=0.0,
        )
        self.assertAlmostEqual(float(actual), 2.5)
        self.assertAlmostEqual(float(actual), float(flipped))

    def test_probe_median_absolute_is_independently_sign_invariant(self):
        probe_values = np.asarray(
            [
                [[-100.0, 1.0]],
                [[2.0, -2.0]],
                [[3.0, 3.0]],
                [[4.0, -4.0]],
            ],
            dtype=np.float64,
        )
        expected = np.asarray([[3.5, 2.5]])
        np.testing.assert_allclose(
            reduce_probe_median_absolute(probe_values),
            expected,
        )
        probe_values[[0, 2]] *= -1.0
        np.testing.assert_allclose(
            reduce_probe_median_absolute(probe_values),
            expected,
        )

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

    def test_multiple_probes_average_squared_dots_not_gradients(self):
        train = np.asarray([[1.0, 2.0], [3.0, -1.0]], dtype=np.float64)
        probes = np.asarray(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 1.0], [1.0, -1.0]],
                [[2.0, -1.0], [-1.0, 2.0]],
                [[0.5, 0.5], [-0.5, 0.5]],
            ],
            dtype=np.float64,
        )
        actual = np.mean(
            np.stack([np.square(train @ query.T) for query in probes], axis=0),
            axis=0,
        )
        expected = sum(np.square(train @ query.T) for query in probes) / 4.0
        np.testing.assert_allclose(actual, expected)
        self.assertFalse(
            np.allclose(actual, np.square(train @ np.mean(probes, axis=0).T))
        )

    def test_final_probe_square_reductions_happen_after_trajectory_sum(self):
        # Axis 0 is probe. These are already-complete per-probe trajectory scores.
        probe_scores = np.asarray(
            [
                [[1.0, 2.0]],
                [[-1.0, 4.0]],
                [[3.0, -2.0]],
                [[-3.0, 0.0]],
            ],
            dtype=np.float64,
        )
        reduced = reduce_final_probe_scores(probe_scores)

        np.testing.assert_allclose(reduced["linear_mean"], [[0.0, 1.0]])
        np.testing.assert_allclose(reduced["square_then_mean"], [[5.0, 6.0]])
        np.testing.assert_allclose(reduced["mean_then_square"], [[0.0, 1.0]])
        self.assertFalse(
            np.allclose(
                reduced["square_then_mean"],
                reduced["mean_then_square"],
            )
        )

    def test_f_next_final_score_is_squared_per_datasample(self):
        scores = np.asarray([-3.0, -0.5, 0.0, 2.0], dtype=np.float64)
        np.testing.assert_allclose(
            square_final_scores(scores),
            [9.0, 0.25, 0.0, 4.0],
        )

    def test_f_next_final_square_materializes_all_normalizations(self):
        materializer = (
            ROOT / "3dshapes" / "script" / "materialize_f_next_final_score_squared.py"
        ).read_text()
        lds_driver = (
            ROOT / "3dshapes" / "script" / "run_traj_tracin_lds_cached.py"
        ).read_text()

        self.assertIn('("raw", "score")', materializer)
        self.assertIn('("query_l2", "score_query_normalized")', materializer)
        self.assertIn('("train_l2", "score_train_l2_normalized")', materializer)
        self.assertIn(
            '("query_train_l2", "score_query_train_l2_normalized")',
            materializer,
        )
        expected_set = lds_driver.split(
            "EXPECTED_RESIDUAL_JACOBIAN_SCHEMES = {", 1
        )[1].split("}", 1)[0]
        self.assertNotIn('"f_next_final_score_squared"', expected_set)

    def test_learning_rate_is_linear_and_the_term_sum_is_not_normalized(self):
        directional = np.asarray([2.0, 3.0], dtype=np.float64)
        learning_rates = np.asarray([0.1, 0.4], dtype=np.float64)

        actual = np.sum(learning_rates * np.square(directional))
        expected = 0.1 * (2.0**2) + 0.4 * (3.0**2)

        self.assertAlmostEqual(float(actual), expected)
        self.assertNotAlmostEqual(
            float(actual),
            float(np.sum(np.square(learning_rates) * np.square(directional))),
        )
        self.assertNotAlmostEqual(
            float(actual),
            float(actual / np.sum(learning_rates)),
        )

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
        self.assertIn("term_weight = float(weight)", driver)
        self.assertNotIn("weight_squared = float(weight) ** 2", driver)
        self.assertIn(
            "learning_rate_weighted_sum_of_squared_gradient_contractions", driver
        )
        self.assertIn("trajectory_predicted_noise_probe", driver)
        self.assertIn('f"run_{run_id}"', driver)
        self.assertIn("--cleanup-query-artifacts", launcher)
        self.assertIn('--run-id "${SLURM_JOB_ID}"', launcher)
        self.assertIn("--shard-count 2", launcher)
        self.assertIn("--score-schemes predicted_noise_jvp_l2_squared", launcher)

    def test_four_probe_rtx_pipeline_is_independent_and_reuses_train(self):
        launcher = (
            ROOT
            / "3dshapes"
            / "tacc"
            / "rtx_small"
            / "run_predicted_noise_jvp_l2_squared_probe4_rtx_small.sh"
        ).read_text()
        driver = (ROOT / "3dshapes" / "script" / "run_predicted_noise_jvp_l2_squared.py").read_text()
        self.assertIn("NUM_PROBES=4", launcher)
        self.assertIn("for probe_index in 0 1 2 3", launcher)
        self.assertIn('--predicted-noise-probe-index "${probe_index}"', launcher)
        self.assertIn('--num-probes "${NUM_PROBES}"', launcher)
        self.assertIn("predicted_noise_jvp_l2_squared_probe4", launcher)
        self.assertIn("for probe_index in range(args.num_probes)", driver)
        self.assertIn("/ float(args.num_probes)", driver)

    def test_signed_four_probe_score_reuses_original_train_and_saved_queries(self):
        driver = (
            ROOT / "3dshapes" / "script" / "run_predicted_noise_jvp_l2_squared.py"
        ).read_text()
        launcher = (
            ROOT
            / "3dshapes"
            / "tacc"
            / "rtx_small"
            / "run_predicted_noise_jvp_signed_probe4_score_rtx_small.sh"
        ).read_text()

        for contraction in ("squared", "signed", "final_post_square"):
            self.assertIn(f'"{contraction}"', driver)
        self.assertIn('SIGNED_NAMESPACE = "predicted_noise_jvp_signed"', driver)
        self.assertIn("query_namespace_pattern.format(probe_index=probe_index)", driver)
        self.assertIn("--contraction signed", launcher)
        self.assertIn(
            "loss_direction_residual_rms_predicted_noise_probe4_r{probe_index}",
            launcher,
        )
        self.assertNotIn("run_traj_tracin_queries_and_scores.py", launcher)
        self.assertIn("predicted_noise_jvp_signed_probe4", launcher)

    def test_final_post_square_launcher_emits_both_reductions(self):
        driver = (
            ROOT / "3dshapes" / "script" / "run_predicted_noise_jvp_l2_squared.py"
        ).read_text()
        launcher = (
            ROOT
            / "3dshapes"
            / "tacc"
            / "rtx_small"
            / "run_predicted_noise_probe4_final_post_square_score_rtx_small.sh"
        ).read_text()

        self.assertIn('"final_post_square"', driver)
        self.assertIn("reduce_final_probe_scores", driver)
        self.assertIn("--contraction final_post_square", launcher)
        self.assertIn("predicted_noise_jvp_final_square_then_mean_probe4", launcher)
        self.assertIn("predicted_noise_jvp_final_mean_then_square_probe4", launcher)
        self.assertIn("--prediction-sign=-1", launcher)

    def test_eight_probe_pipeline_reuses_first_four_and_emits_both_reductions(self):
        launcher = (
            ROOT
            / "3dshapes"
            / "tacc"
            / "rtx_small"
            / "run_predicted_noise_probe8_final_post_square_pipeline_rtx_small.sh"
        ).read_text()

        self.assertIn("NUM_PROBES=8", launcher)
        self.assertIn("for probe_index in 0 1 2 3 4 5 6 7", launcher)
        self.assertIn(
            "loss_direction_residual_rms_predicted_noise_probe4_r${probe_index}",
            launcher,
        )
        self.assertIn("--contraction final_post_square", launcher)
        self.assertIn("predicted_noise_jvp_final_square_then_mean_probe8", launcher)
        self.assertIn("predicted_noise_jvp_final_mean_then_square_probe8", launcher)
        self.assertIn("--prediction-sign=-1", launcher)

    def test_eight_probe_linear_mean_can_run_cached_lds_and_print_with_positive_sign(self):
        cached_lds = (
            ROOT / "3dshapes" / "script" / "run_traj_tracin_lds_cached.py"
        ).read_text()
        printer = (
            ROOT
            / "3dshapes"
            / "script"
            / "print_predicted_noise_probe4_final_post_square_lds.py"
        ).read_text()

        self.assertIn('"predicted_noise_jvp_final_linear_mean_probe8"', cached_lds)
        self.assertIn(
            '"traj_tracin_predicted_noise_jvp_final_linear_mean_probe8"', cached_lds
        )
        self.assertIn('choices=("linear", "square", "all")', printer)
        self.assertIn('"LINEAR MEAN (NO SQUARE)"', printer)
        self.assertIn('"p1",', printer)

    def test_twelve_probe_pipeline_reuses_first_eight_and_runs_all_reductions(self):
        launcher = (
            ROOT
            / "3dshapes"
            / "tacc"
            / "rtx_small"
            / "run_predicted_noise_probe12_final_post_square_pipeline_rtx_small.sh"
        ).read_text()
        cached_lds = (
            ROOT / "3dshapes" / "script" / "run_traj_tracin_lds_cached.py"
        ).read_text()
        printer = (
            ROOT
            / "3dshapes"
            / "script"
            / "print_predicted_noise_probe4_final_post_square_lds.py"
        ).read_text()

        self.assertIn("NUM_PROBES=12", launcher)
        self.assertIn("for probe_index in 0 1 2 3 4 5 6 7 8 9 10 11", launcher)
        self.assertIn("reuse 0-7 and generate missing 8-11", launcher)
        self.assertIn("predicted_noise_jvp_final_linear_mean_probe12", launcher)
        self.assertIn("predicted_noise_jvp_final_square_then_mean_probe12", launcher)
        self.assertIn("predicted_noise_jvp_final_mean_then_square_probe12", launcher)
        self.assertIn("--prediction-sign=1", launcher)
        self.assertIn("--prediction-sign=-1", launcher)
        self.assertIn(
            '"predicted_noise_jvp_final_linear_mean_probe12"', cached_lds
        )
        self.assertIn(
            '"traj_tracin_predicted_noise_jvp_final_mean_then_square_probe12"',
            cached_lds,
        )
        self.assertIn("choices=(4, 8, 12)", printer)

    def test_probe8_timestamp_grouped_checkpoint_square_pipeline(self):
        driver = (
            ROOT / "3dshapes" / "script" / "run_predicted_noise_jvp_l2_squared.py"
        ).read_text()
        launcher = (
            ROOT
            / "3dshapes"
            / "tacc"
            / "rtx_small"
            / "run_predicted_noise_probe8_timestamp_checkpoint_square_rtx_small.sh"
        ).read_text()
        cached_lds = (
            ROOT / "3dshapes" / "script" / "run_traj_tracin_lds_cached.py"
        ).read_text()

        self.assertIn('"timestamp_checkpoint_square"', driver)
        self.assertIn("checkpoint_lr = float(weight) * float(len(timestep_values))", driver)
        self.assertIn("reduce_timestamp_checkpoint_sums(values)", driver)
        self.assertIn("NUM_PROBES=8", launcher)
        self.assertIn("--contraction timestamp_checkpoint_square", launcher)
        self.assertIn(
            "predicted_noise_jvp_timestamp_checkpoint_sum_square_probe8",
            launcher,
        )
        self.assertIn("--prediction-sign=-1", launcher)
        self.assertIn(
            '"traj_tracin_predicted_noise_jvp_timestamp_checkpoint_sum_square_probe8"',
            cached_lds,
        )

    def test_timestamp_checkpoint_sums_square_then_average_timestamp_and_probe(self):
        checkpoint_sums = np.asarray(
            [
                [[[1.0, 2.0]], [[3.0, 4.0]]],
                [[[5.0, 6.0]], [[7.0, 8.0]]],
            ],
            dtype=np.float64,
        )
        actual = reduce_timestamp_checkpoint_sums(checkpoint_sums)
        expected = np.asarray(
            [[
                (1.0 + 9.0 + 25.0 + 49.0) / 4.0,
                (4.0 + 16.0 + 36.0 + 64.0) / 4.0,
            ]]
        )
        np.testing.assert_allclose(actual, expected)

    def test_checkpoint_timestamp_means_average_timestamps_before_square(self):
        probe_timestamp_scores = np.asarray(
            [
                [[[1.0, 2.0]], [[3.0, 4.0]]],
                [[[-2.0, 1.0]], [[2.0, 3.0]]],
            ],
            dtype=np.float64,
        )
        probe_timestamp_means = np.mean(probe_timestamp_scores, axis=1)
        actual = reduce_checkpoint_timestamp_means(probe_timestamp_means)
        expected = np.asarray([[(2.0**2 + 0.0**2) / 2.0, (3.0**2 + 2.0**2) / 2.0]])
        np.testing.assert_allclose(actual, expected)
        self.assertFalse(
            np.allclose(
                actual,
                np.mean(np.square(probe_timestamp_scores), axis=(0, 1)),
            )
        )

    def test_probe12_checkpoint_timestamp_mean_square_pipeline(self):
        launcher = (
            ROOT
            / "3dshapes"
            / "tacc"
            / "rtx_small"
            / "run_predicted_noise_probe12_checkpoint_timestamp_square_rtx_small.sh"
        ).read_text()
        cached_lds = (
            ROOT / "3dshapes" / "script" / "run_traj_tracin_lds_cached.py"
        ).read_text()

        self.assertIn("NUM_PROBES=12", launcher)
        self.assertIn("--contraction checkpoint_timestamp_sum_square", launcher)
        self.assertIn("checkpoint_timestamp_sum_square_probe12", launcher)
        self.assertIn("--prediction-sign=-1", launcher)
        self.assertIn(
            '"traj_tracin_predicted_noise_jvp_checkpoint_timestamp_sum_square_probe12"',
            cached_lds,
        )

    def test_probe8_checkpoint_timestamp_mean_square_pipeline(self):
        launcher = (
            ROOT
            / "3dshapes"
            / "tacc"
            / "rtx_small"
            / "run_predicted_noise_probe8_checkpoint_timestamp_square_rtx_small.sh"
        ).read_text()
        cached_lds = (
            ROOT / "3dshapes" / "script" / "run_traj_tracin_lds_cached.py"
        ).read_text()

        self.assertIn("NUM_PROBES=8", launcher)
        self.assertIn("--contraction checkpoint_timestamp_sum_square", launcher)
        self.assertIn("checkpoint_timestamp_sum_square_probe8", launcher)
        self.assertIn("--prediction-sign=-1", launcher)
        self.assertIn(
            '"traj_tracin_predicted_noise_jvp_checkpoint_timestamp_sum_square_probe8"',
            cached_lds,
        )

    def test_probe8_per_checkpoint_lds_pipeline_retains_checkpoint_axis(self):
        driver = (
            ROOT / "3dshapes" / "script" / "run_predicted_noise_jvp_l2_squared.py"
        ).read_text()
        analyzer = (
            ROOT
            / "3dshapes"
            / "script"
            / "analyze_predicted_noise_per_checkpoint_lds.py"
        ).read_text()
        launcher = (
            ROOT
            / "3dshapes"
            / "tacc"
            / "rtx_small"
            / "run_predicted_noise_probe8_per_checkpoint_lds_rtx_small.sh"
        ).read_text()

        self.assertIn('np.zeros((50, 10, 5000)', driver)
        self.assertIn('"per_checkpoint_scores.npz"', driver)
        self.assertIn('checkpoint_learning_rate_applied=np.asarray(False)', driver)
        self.assertIn('checkpoint_unweighted_score', driver)
        self.assertIn('--retain-checkpoint-scores', launcher)
        self.assertIn('NUM_PROBES=8', launcher)
        self.assertIn('per_query_checkpoint_lds.csv', analyzer)
        self.assertIn('checkpoint_summary.csv', analyzer)
        self.assertIn('values[:, query_id, :] @ kept_matrix.T', analyzer)

    def test_probe12_per_checkpoint_lds_pipeline_reuses_generic_analyzer(self):
        launcher = (
            ROOT
            / "3dshapes"
            / "tacc"
            / "rtx_small"
            / "run_predicted_noise_probe12_per_checkpoint_lds_rtx_small.sh"
        ).read_text()

        self.assertIn('NUM_PROBES=12', launcher)
        self.assertIn('--retain-checkpoint-scores', launcher)
        self.assertIn('analyze_predicted_noise_per_checkpoint_lds.py', launcher)
        self.assertIn('--num-probes "${NUM_PROBES}"', launcher)
        self.assertIn('predicted_noise_probe12_per_checkpoint_lds', launcher)

    def test_product_square_per_checkpoint_supports_probe8_and_probe12(self):
        driver = (
            ROOT / "3dshapes" / "script" / "run_predicted_noise_jvp_l2_squared.py"
        ).read_text()
        analyzer = (
            ROOT
            / "3dshapes"
            / "script"
            / "analyze_predicted_noise_per_checkpoint_lds.py"
        ).read_text()
        launcher = (
            ROOT
            / "3dshapes"
            / "tacc"
            / "rtx_small"
            / "run_predicted_noise_product_square_per_checkpoint_lds_rtx_small.sh"
        ).read_text()

        self.assertIn('args.contraction == "squared"', driver)
        self.assertIn('values / float(len(timesteps))', driver)
        self.assertIn('choices=("checkpoint_timestamp_sum_square", "squared")', analyzer)
        self.assertIn('termwise_product_square', analyzer)
        self.assertIn('export NUM_PROBES="${NUM_PROBES:-8}"', launcher)
        self.assertIn('--contraction squared --retain-checkpoint-scores', launcher)

    def test_shared_orthogonal_probe4_timestamp_checkpoint_square_pipeline(self):
        launcher = (
            ROOT
            / "3dshapes"
            / "tacc"
            / "rtx_small"
            / "run_predicted_noise_shared_orthogonal_probe4_timestamp_checkpoint_square_rtx_small.sh"
        ).read_text()
        cached_lds = (
            ROOT / "3dshapes" / "script" / "run_traj_tracin_lds_cached.py"
        ).read_text()
        printer = (
            ROOT
            / "3dshapes"
            / "script"
            / "print_predicted_noise_timestamp_checkpoint_square_lds.py"
        ).read_text()

        self.assertIn("NUM_PROBES=4", launcher)
        self.assertIn(
            "QUERY_PATTERN='predicted_noise_shared_orthogonal_probe4_r{probe_index}'",
            launcher,
        )
        self.assertIn("--expected-query-probe-mode shared_orthogonal", launcher)
        self.assertIn("--contraction timestamp_checkpoint_square", launcher)
        self.assertIn("NAMESPACE_SUFFIX=orthogonal_shared", launcher)
        self.assertIn(
            "predicted_noise_shared_orthogonal_probe4_timestamp_checkpoint_sum_square",
            launcher,
        )
        self.assertIn(
            '"traj_tracin_predicted_noise_jvp_timestamp_checkpoint_sum_square_probe4_orthogonal_shared"',
            cached_lds,
        )
        self.assertIn('parser.add_argument("--namespace-suffix"', printer)

    def test_shared_orthogonal_probe4_termwise_square_pipeline(self):
        launcher = (
            ROOT
            / "3dshapes"
            / "tacc"
            / "rtx_small"
            / "run_predicted_noise_shared_orthogonal_probe4_termwise_square_rtx_small.sh"
        ).read_text()
        cached_lds = (
            ROOT / "3dshapes" / "script" / "run_traj_tracin_lds_cached.py"
        ).read_text()
        printer = (
            ROOT
            / "3dshapes"
            / "script"
            / "print_predicted_noise_timestamp_checkpoint_square_lds.py"
        ).read_text()

        self.assertIn("NUM_PROBES=4", launcher)
        self.assertIn("--contraction squared", launcher)
        self.assertIn("--expected-query-probe-mode shared_orthogonal", launcher)
        self.assertIn("NAMESPACE_SUFFIX=orthogonal_shared", launcher)
        self.assertIn(
            "predicted_noise_shared_orthogonal_probe4_termwise_square",
            launcher,
        )
        self.assertIn("--reduction termwise_square", launcher)
        self.assertIn(
            '"traj_tracin_predicted_noise_jvp_l2_squared_probe4_orthogonal_shared"',
            cached_lds,
        )
        self.assertIn('"timestamp_checkpoint_square"', printer)
        self.assertIn('"termwise_square"', printer)

    def test_probe8_termwise_square_pipeline_reuses_saved_queries(self):
        launcher = (
            ROOT
            / "3dshapes"
            / "tacc"
            / "rtx_small"
            / "run_predicted_noise_probe8_termwise_square_rtx_small.sh"
        ).read_text()
        cached_lds = (
            ROOT / "3dshapes" / "script" / "run_traj_tracin_lds_cached.py"
        ).read_text()

        self.assertIn("NUM_PROBES=8", launcher)
        self.assertIn("--contraction squared", launcher)
        self.assertIn(
            "loss_direction_residual_rms_predicted_noise_probe4_r{probe_index}",
            launcher,
        )
        self.assertNotIn("run_traj_tracin_queries_and_scores.py", launcher)
        self.assertIn("predicted_noise_jvp_l2_squared_probe8", launcher)
        self.assertIn("--reduction termwise_square", launcher)
        self.assertIn(
            '"traj_tracin_predicted_noise_jvp_l2_squared_probe8"', cached_lds
        )

    def test_cumulative_probe_banks_emit_linear_and_termwise_square(self):
        launcher = (
            ROOT
            / "3dshapes"
            / "tacc"
            / "rtx_small"
            / "run_predicted_noise_bank_linear_termwise_square_rtx_small.sh"
        ).read_text()
        cached_lds = (
            ROOT / "3dshapes" / "script" / "run_traj_tracin_lds_cached.py"
        ).read_text()
        printer = (
            ROOT
            / "3dshapes"
            / "script"
            / "print_predicted_noise_timestamp_checkpoint_square_lds.py"
        ).read_text()

        self.assertIn("independent12)", launcher)
        self.assertIn("fixed8)", launcher)
        self.assertIn("run_score signed linear", launcher)
        self.assertIn("run_score squared termwise_square", launcher)
        self.assertIn('local label="$2"', launcher)
        self.assertIn('local run_id="${SLURM_JOB_ID}_${label}"', launcher)
        self.assertIn("--prediction-sign=1", launcher)
        self.assertIn("--prediction-sign=-1", launcher)
        self.assertIn("predicted_noise_jvp_signed_probe12", cached_lds)
        self.assertIn("predicted_noise_jvp_l2_squared_probe12", cached_lds)
        self.assertIn("predicted_noise_shared_orthogonal_probe8_linear", cached_lds)
        self.assertIn(
            "predicted_noise_shared_orthogonal_probe8_termwise_square", cached_lds
        )
        self.assertIn('"linear": "traj_tracin_predicted_noise_jvp_signed"', printer)

    def test_probe8_choose4_analysis_enumerates_all_subsets(self):
        path = (
            ROOT
            / "3dshapes"
            / "script"
            / "analyze_predicted_noise_probe8_choose4.py"
        )
        text = path.read_text()

        self.assertEqual(len(list(itertools.combinations(range(8), 4))), 70)
        self.assertIn("itertools.combinations(range(8), 4)", text)
        self.assertIn('default=1.0', text)
        self.assertIn('"sums_score_query_train_l2_normalized"', text)
        self.assertIn('"per_query.csv"', text)
        self.assertIn('"ten_query_means.csv"', text)

    def test_probe8_all_subset_sizes_analysis_covers_255_nonempty_subsets(self):
        path = (
            ROOT
            / "3dshapes"
            / "script"
            / "analyze_predicted_noise_probe8_all_subset_sizes.py"
        )
        text = path.read_text()

        self.assertEqual(sum(len(list(itertools.combinations(range(8), k))) for k in range(1, 9)), 255)
        self.assertIn("for subset_size in range(1, num_probes + 1)", text)
        self.assertIn('"subset_size_distribution.csv"', text)
        self.assertIn('"both_l2_counterfactual_by_subset_size.png"', text)
        self.assertIn("ax.errorbar(", text)

    def test_probe12_all_subset_sizes_launcher_covers_4095_subsets(self):
        analyzer = (
            ROOT
            / "3dshapes"
            / "script"
            / "analyze_predicted_noise_probe8_all_subset_sizes.py"
        ).read_text()
        launcher = (
            ROOT
            / "3dshapes"
            / "tacc"
            / "rtx_small"
            / "run_predicted_noise_probe12_all_subset_sizes_rtx_small.sh"
        ).read_text()

        self.assertEqual(
            sum(
                len(list(itertools.combinations(range(12), k)))
                for k in range(1, 13)
            ),
            4095,
        )
        self.assertIn("--num-probes", analyzer)
        self.assertIn("load_probe_scores(shard_dir, args.num_probes)", analyzer)
        self.assertIn("all 4095 nonempty subsets", launcher)
        self.assertIn("--num-probes 12", launcher)
        self.assertIn('RUN_ID="${SOURCE_RUN_ID:-3502474}"', launcher)

    def test_probe12_termwise_square_all_subset_launcher_retains_probe_axis(self):
        launcher = (
            ROOT
            / "3dshapes"
            / "tacc"
            / "rtx_small"
            / "run_predicted_noise_probe12_termwise_square_all_subsets_rtx_small.sh"
        ).read_text()
        analyzer = (
            ROOT
            / "3dshapes"
            / "script"
            / "analyze_predicted_noise_probe8_all_subset_sizes.py"
        ).read_text()
        self.assertIn("--contraction termwise_squared_per_probe", launcher)
        self.assertIn("--prediction-sign=-1", launcher)
        self.assertIn("probe12_all_subset_sizes_termwise_square", launcher)
        self.assertIn("--score-namespace", analyzer)
        self.assertIn("all_probe_subsets(args.num_probes)", analyzer)
        self.assertIn('NUM_PROBES=12', launcher)
        self.assertIn("all 4095 nonempty subsets", launcher)

    def test_probe24_term_winner_analysis_compares_next_update_and_train_directions(self):
        analyzer = (
            ROOT
            / "3dshapes"
            / "script"
            / "analyze_predicted_noise_probe24_term_winners.py"
        ).read_text()
        launcher = (
            ROOT
            / "3dshapes"
            / "tacc"
            / "rtx_small"
            / "run_predicted_noise_probe24_term_winner_analysis_rtx_small.sh"
        ).read_text()
        self.assertIn("ORIGINAL_PATTERN", analyzer)
        self.assertIn("FRESH_PATTERN", analyzer)
        self.assertIn("NEXT_PATTERN", analyzer)
        self.assertIn("winner = int(np.argmax(joint))", analyzer)
        self.assertIn('"cosine_to_next_update"', analyzer)
        self.assertIn('"mean_train_gradient_cosine"', analyzer)
        self.assertIn('"all_probe_term_lds.csv"', analyzer)
        self.assertIn('export QUERY_IDS="${QUERY_IDS:-2,3}"', launcher)

    def test_probe12_sign_flip_analysis_covers_all_4096_assignments(self):
        analyzer = (
            ROOT
            / "3dshapes"
            / "script"
            / "analyze_predicted_noise_probe12_sign_flips.py"
        ).read_text()
        launcher = (
            ROOT
            / "3dshapes"
            / "tacc"
            / "rtx_small"
            / "run_predicted_noise_probe12_sign_flips_rtx_small.sh"
        ).read_text()

        self.assertEqual(1 << 12, 4096)
        self.assertIn("def sign_matrix(num_probes: int)", analyzer)
        self.assertIn("means + means[::-1]", analyzer)
        self.assertIn("all_plus_percentile", analyzer)
        self.assertIn("both_l2_counterfactual_sign_histogram.png", analyzer)
        self.assertIn("exhaustive 4096 sign assignments", launcher)
        self.assertIn('RUN_ID="${SOURCE_RUN_ID:-3503519}"', launcher)


if __name__ == "__main__":
    unittest.main()
