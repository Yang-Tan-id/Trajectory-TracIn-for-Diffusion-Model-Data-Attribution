import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "diffusion_jax_refined/celeba/script/evaluate_top1000_q0_69_generate.py"
PERCEPTUAL = ROOT / "diffusion_jax_refined/celeba/script/evaluate_top1000_q0_69_perceptual.py"
RUNNER = ROOT / "diffusion_jax_refined/celeba/script/run_top1000_q0_69_generate_4gpu.sh"


class CelebaTop1000MetricsTest(unittest.TestCase):
    def test_scripts_parse(self):
        ast.parse(GENERATOR.read_text(encoding="utf-8"))
        ast.parse(PERCEPTUAL.read_text(encoding="utf-8"))

    def test_five_methods_are_identical_in_both_stages(self):
        generator = GENERATOR.read_text(encoding="utf-8")
        perceptual = PERCEPTUAL.read_text(encoding="utf-8")
        methods = (
            "endpoint_pollute_adamw_timestamp_both_l2_full10k_pos",
            "endpoint_tracin_adamw_linear_raw_full10k_neg",
            "endpoint_tracin_adamw_linear_both_l2_full10k_neg",
            "retrac_adamw_query_train_l2_full10k_neg",
            "random_full10k_seeded",
        )
        for method in methods:
            self.assertIn(method, generator)
            self.assertIn(method, perceptual)

    def test_generator_uses_local_checkpoint_and_ten_step_resume(self):
        source = GENERATOR.read_text(encoding="utf-8")
        self.assertIn('run_dir / "model"', source)
        self.assertNotIn('metrics["removal_checkpoint"]', source)
        self.assertIn("endpoint_top1000_removal_q0_9", source)
        self.assertIn("matches_by_target", source)
        self.assertIn("METRICS_GENERATION_COMPLETE", source)
        self.assertIn("original_trajectory.npz", source)
        self.assertIn("per_step_mse", source)
        self.assertIn("endpoint_pixel_metrics", source)
        self.assertIn("display_image(reference)", source)

    def test_all_requested_metrics_are_emitted(self):
        source = PERCEPTUAL.read_text(encoding="utf-8")
        for metric in (
            "endpoint_mse",
            "endpoint_rmse",
            "trajectory_mse",
            "trajectory_rmse",
            "clip_distance",
            "lpips",
            "dino_distance",
        ):
            self.assertIn(metric, source)

    def test_runner_declares_four_gpu_default_and_70_completion(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("GPU_IDS:-0,5,6,7", source)
        self.assertIn('count" == "70"', source)


if __name__ == "__main__":
    unittest.main()
