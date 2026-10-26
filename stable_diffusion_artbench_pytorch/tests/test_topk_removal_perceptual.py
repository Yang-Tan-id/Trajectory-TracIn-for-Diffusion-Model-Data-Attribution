import ast
import itertools
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "src/sd_artbench/evaluate_topk_removal_perceptual.py"
)


class PerceptualEvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SCRIPT.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_four_methods_and_metrics_are_present(self):
        for value in (
            "endpoint_pollute_query_train_l2",
            "endpoint_tracin_raw",
            "endpoint_tracin_query_train_l2",
            "retrac_query_train_l2",
            "pixel_l2",
            "clip_distance",
            "lpips",
            "dino_distance",
        ):
            self.assertIn(value, self.source)

    def test_reuses_saved_trajectory_images(self):
        self.assertIn('"full-data.png"', self.source)
        self.assertIn('f"{method}.png"', self.source)
        self.assertNotIn("StableDiffusionPipeline", self.source)

    def test_exact_label_swap(self):
        function = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "exact_label_swap"
        )
        namespace = {"Sequence": list, "itertools": itertools}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(SCRIPT), "exec"), namespace)
        result = namespace["exact_label_swap"]([4.0, 3.0], [1.0, 2.0])
        self.assertEqual(result["wins"], 2)
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["p_one_sided"], 0.25)
        self.assertEqual(result["p_two_sided"], 0.5)

    def test_outputs_are_declared(self):
        for filename in (
            "perceptual_metrics.csv",
            "perceptual_label_swap.csv",
            "perceptual_report.txt",
            "perceptual_manifest.json",
        ):
            self.assertIn(filename, self.source)


if __name__ == "__main__":
    unittest.main()
