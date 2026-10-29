import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYZE = ROOT / "celeba/script/analyze_random_removal_trajectory_change_profile.py"
SUMMARY = ROOT / "celeba/script/summarize_random_removal_trajectory_change_profile.py"
RUNNER = ROOT / "celeba/script/run_random_trajectory_change_profile_q0_9_4gpu.sh"


class CelebaRandomTrajectoryChangeProfileTest(unittest.TestCase):
    def test_scripts_parse(self):
        ast.parse(ANALYZE.read_text(encoding="utf-8"))
        ast.parse(SUMMARY.read_text(encoding="utf-8"))

    def test_saves_true_per_step_state_mse(self):
        source = ANALYZE.read_text(encoding="utf-8")
        self.assertIn("per_step, trajectory_mse", source)
        self.assertIn('"trajectory_state_mse": float(value)', source)
        self.assertIn("timesteps_to_save=timesteps", source)

    def test_summary_reports_concentration(self):
        source = SUMMARY.read_text(encoding="utf-8")
        self.assertIn("equal_pair_share", source)
        self.assertIn("FIVE 10-STEP SEGMENTS", source)
        self.assertIn("CUMULATIVE CHANGE FROM THE NOISY START", source)

    def test_runner_uses_all_models_and_summarizes(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("seq -s, 0 69", source)
        self.assertIn("summarize_random_removal_trajectory_change_profile.py", source)
        self.assertIn('query_groups=("0,1,2" "3,4,5" "6,7" "8,9")', source)


if __name__ == "__main__":
    unittest.main()
