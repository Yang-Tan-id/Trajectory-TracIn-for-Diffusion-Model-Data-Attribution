from __future__ import annotations

import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCORER = (
    REPO_ROOT
    / "diffusion_jax_refined/3dshapes/script/score_endpoint_pollute_10x1_timestamp_square.py"
)
LAUNCHER = (
    REPO_ROOT
    / "diffusion_jax_refined/3dshapes/tacc/rtx_small/"
    "run_endpoint_pollute_10x1_per_timestamp_square_rtx_small.sh"
)
LDS_RUNNER = (
    REPO_ROOT
    / "diffusion_jax_refined/3dshapes/script/run_traj_tracin_lds_cached.py"
)
COMBINER = (
    REPO_ROOT
    / "diffusion_jax_refined/3dshapes/script/"
    "combine_endpoint_pollute_10x1_timestamp_square_first7.py"
)


class EndpointPolluteTimestampSquareTest(unittest.TestCase):
    def test_scorer_contract(self) -> None:
        text = SCORER.read_text()
        tree = ast.parse(text)
        score_root = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "score_root"
        )
        self.assertTrue(
            any(isinstance(statement, ast.Return) for statement in score_root.body),
            "score_root must return its constructed output path",
        )
        self.assertIn("float(weight) * contribution", text)
        self.assertIn("np.square(np.asarray(jax.device_get(value)", text)
        self.assertLess(
            text.index("float(weight) * contribution"),
            text.index("np.square(np.asarray(jax.device_get(value)"),
        )
        self.assertIn("optimizer_history_features", text)
        self.assertIn("train_term + history_term[None, :]", text)
        self.assertIn("query_outputs_complete", text)

    def test_launcher_uses_real_slurm_tasks_and_p1(self) -> None:
        text = LAUNCHER.read_text()
        self.assertIn("ibrun -n 1 -o", text)
        self.assertIn('${PREDICTION_SIGN:-1}', text)
        self.assertIn("summary.txt", text)
        self.assertNotIn("--nodelist", text)
        self.assertIn('${SLURM_JOB_ID:-${RUN_TAG:-manual}}', text)

    def test_all_timestamps_are_registered_for_lds(self) -> None:
        text = LDS_RUNNER.read_text()
        self.assertIn(
            "for _timestep in (0, 111, 222, 333, 444, 555, 666, 777, 888, 999)",
            text,
        )
        self.assertIn("timestamp_aware_square_t{_timestep:03d}_q0_99", text)

    def test_first7_combiner_sums_cached_square_scores(self) -> None:
        text = COMBINER.read_text()
        ast.parse(text)
        self.assertIn("TIMESTAMPS = (0, 111, 222, 333, 444, 555, 666)", text)
        self.assertIn("total += current_scores", text)
        self.assertIn(
            '"formula": "sum_first7(per_timestamp_checkpoint_sum_squared)"',
            text,
        )


if __name__ == "__main__":
    unittest.main()
