import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "celeba/script/analyze_random_removal_simple_loss_correlation.py"
RUNNER = ROOT / "celeba/script/run_random_cross_model_simple_loss_50x1_q0_9_4gpu.sh"


class CelebaRandomSimpleLossCorrelationTest(unittest.TestCase):
    def test_script_parses(self):
        ast.parse(SCRIPT.read_text(encoding="utf-8"))

    def test_exact_50_by_1_contract(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("np.linspace(999, 0, 50", source)
        self.assertIn("jax.random.split(root_key, 50)", source)
        self.assertIn("jax.vmap(one)(timesteps, rng_keys)", source)
        self.assertIn('"mc_definition": "50 timesteps x 1 noise each"', source)

    def test_uses_existing_endpoint_metrics_without_resampling(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("load_endpoint_metrics", source)
        self.assertNotIn("_sample_model_space_trajectory", source)

    def test_outputs_signed_absolute_and_per_timestep(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("simple_loss_signed_delta", source)
        self.assertIn("simple_loss_absolute_delta", source)
        self.assertIn("simple_loss_50x1_per_timestep.csv", source)

    def test_runner_uses_four_gpus_and_all_models(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("GPU_IDS:-4,5,6,7", source)
        self.assertIn("seq -s, 0 69", source)
        self.assertIn('query_groups=("0,1,2" "3,4,5" "6,7" "8,9")', source)


if __name__ == "__main__":
    unittest.main()
