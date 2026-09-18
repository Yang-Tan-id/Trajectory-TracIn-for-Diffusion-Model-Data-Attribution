from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DasGenerationTrajectoryQueryTests(unittest.TestCase):
    def test_query_kernel_uses_saved_xt_without_q_sample(self):
        source = (ROOT / "legacy_jax" / "das" / "algorithm.py").read_text()
        start = source.index("def make_projected_trajectory_query_grad_fn(")
        end = source.index("def make_projected_mc_average_loss_grad_fn(", start)
        kernel = source[start:end]
        self.assertIn("adapter.eps_apply(model, p, x_t, t, cond)", kernel)
        self.assertNotIn("q_sample(", kernel)
        self.assertIn('query_input_mode == "generation_trajectory"', source)
        self.assertIn("trajectory_states_by_timestep[t_value]", source)
        self.assertIn("query_input_mode=np.asarray(query_input_mode)", source)

    def test_driver_can_reuse_standard_train_with_isolated_query_namespace(self):
        source = (ROOT / "3dshapes" / "script" / "run_das_queries_and_scores.py").read_text()
        self.assertIn("--train-artifact-namespace", source)
        self.assertIn("--query-input-mode", source)
        self.assertIn("DAS_QUERY_INPUT_MODE=args.query_input_mode", source)
        self.assertIn("/ train_das_name", source)

    def test_pipeline_runs_only_query_score_and_cached_lds(self):
        source = (
            ROOT
            / "3dshapes"
            / "tacc"
            / "rtx_small"
            / "run_das_generation_trajectory100x1_pipeline_rtx_small.sh"
        ).read_text()
        self.assertIn("--artifact-namespace generation_trajectory100x1", source)
        self.assertIn('--train-artifact-namespace ""', source)
        self.assertIn("--query-input-mode generation_trajectory", source)
        self.assertIn("--num-mc-noise 1", source)
        self.assertIn("script/run_das_lds_cached.py", source)
        self.assertNotIn("01_train_datapoint_gradient.py", source)


if __name__ == "__main__":
    unittest.main()
