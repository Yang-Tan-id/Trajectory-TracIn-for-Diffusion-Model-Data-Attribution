from pathlib import Path
import sys
import unittest
import itertools

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "3dshapes" / "script"))

from run_predicted_noise_jvp_l2_squared import reduce_final_probe_scores
from materialize_f_next_final_score_squared import square_final_scores


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

        self.assertIn('choices=("squared", "signed", "final_post_square")', driver)
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


if __name__ == "__main__":
    unittest.main()
