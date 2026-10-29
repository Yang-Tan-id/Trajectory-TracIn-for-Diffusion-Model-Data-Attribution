from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src/sd_artbench/analyze_random_removal_simple_loss_correlation.py"
RUNNER = ROOT / "run_random_removal_simple_loss_q0_9_4gpu.sh"
MERGER = ROOT / "src/sd_artbench/merge_random_removal_simple_loss.py"


class RandomRemovalSimpleLossCorrelationTest(unittest.TestCase):
    def test_script_parses(self) -> None:
        ast.parse(SCRIPT.read_text(encoding="utf-8"))
        ast.parse(MERGER.read_text(encoding="utf-8"))

    def test_uses_query_image_forward_noise_and_training_target(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("pipeline.vae.encode", source)
        self.assertIn("scheduler.add_noise", source)
        self.assertIn('scheduler.config.prediction_type == "epsilon"', source)
        self.assertIn("removed_losses - full_losses", source)

    def test_shared_probes_are_reused_across_models(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        build = source.index("probes = build_shared_probes")
        model_loop = source.index("for model in args.model_indices", build)
        self.assertLess(build, model_loop)
        self.assertIn("simple_losses(pipeline, probes", source)

    def test_reuses_existing_output_metrics(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("load_output_metrics", source)
        self.assertIn("endpoint_mse", source)
        self.assertIn("trajectory_mse", source)

    def test_runner_uses_four_gpus_and_sixty_models(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("GPU_IDS:-4,5,6,7", source)
        self.assertIn("seq -s, 0 59", source)
        self.assertIn('query_groups=("0,1,2" "3,4,5" "6,7" "8,9")', source)
        self.assertIn("sd_artbench.merge_random_removal_simple_loss", source)


if __name__ == "__main__":
    unittest.main()
