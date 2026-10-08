from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DTrakThreeObjectivesTest(unittest.TestCase):
    def test_kernel_has_three_matched_output_functions_and_explicit_grid(self) -> None:
        source = (ROOT / "legacy_jax/dtrak/algorithm.py").read_text()
        self.assertIn('output_function: str = "simple_loss"', source)
        self.assertIn('if output_function == "square":', source)
        self.assertIn('if output_function == "average":', source)
        self.assertIn("jnp.linspace(0, T - 1, count)", source)
        self.assertIn("jnp.sum(jnp.square(pred)", source)

    def test_score_batches_all_queries_into_one_solve_per_objective(self) -> None:
        source = (ROOT / "3dshapes/script/score_dtrak_three_objectives_100x1.py").read_text()
        self.assertIn("rhs = np.stack(query_features, axis=1)", source)
        self.assertIn("solved = np.linalg.solve(gram[0], rhs)", source)
        self.assertIn("scores = (train[0] @ solved).T", source)

    def test_rtx_job_is_resumable_and_not_node_pinned(self) -> None:
        source = (
            ROOT
            / "3dshapes/tacc/rtx_small/run_dtrak_three_objectives_100x1_q0_99_rtx_small.sh"
        ).read_text()
        self.assertNotIn("--nodelist", source)
        self.assertIn('if [[ -f "$artifact" ]]', source)
        self.assertIn("DTRAK_TRAIN_EXPECTATION_SAMPLES=100", source)
        self.assertIn("DTRAK_QUERY_EXPECTATION_SAMPLES=100", source)
        self.assertIn("run_train_objective simple_loss", source)
        self.assertIn("run_train_objective square", source)
        self.assertIn("run_train_objective average", source)


if __name__ == "__main__":
    unittest.main()
