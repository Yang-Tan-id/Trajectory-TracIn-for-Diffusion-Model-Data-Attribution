import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "celeba/script/analyze_random_removal_noise_correlation.py"
RUNNER = ROOT / "celeba/script/run_random_cross_model_noise_q0_9_4gpu.sh"


class CelebaRandomCrossModelNoiseTest(unittest.TestCase):
    def test_script_parses(self):
        ast.parse(SCRIPT.read_text(encoding="utf-8"))

    def test_cross_product_and_same_input_noise(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("for model_id in args.model_ids", source)
        self.assertIn('"random_model_id"', source)
        self.assertIn("evaluator.xt_ref", source)
        self.assertIn("candidate_params", source)
        self.assertIn("base_params", source)
        self.assertIn("jax.vmap(one)(reference_xt, timesteps)", source)

    def test_full_50_step_endpoint_trajectory_and_noise(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("len(timesteps) != 50", source)
        self.assertIn("endpoint_pixel_metrics", source)
        self.assertIn("paired_mse(reference, candidate)", source)
        self.assertIn("predicted_noise_mse", source)

    def test_resume_is_per_query_model_pair(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("completed_pairs", source)
        self.assertIn("cross_model_per_query_timestep.csv", source)

    def test_runner_uses_four_gpus_and_all_70_models(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("GPU_IDS:-4,5,6,7", source)
        self.assertIn("seq -s, 0 69", source)
        self.assertIn('query_groups=("0,1,2" "3,4,5" "6,7" "8,9")', source)


if __name__ == "__main__":
    unittest.main()
