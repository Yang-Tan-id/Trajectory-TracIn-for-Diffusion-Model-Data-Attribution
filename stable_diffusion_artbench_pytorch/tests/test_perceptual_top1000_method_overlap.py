import ast
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "src/sd_artbench/analyze_perceptual_top1000_method_overlap.py"
)


class PerceptualTop1000OverlapTest(unittest.TestCase):
    def test_metric_directions_and_six_methods(self):
        source = SCRIPT.read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn('("pixel_rmse", "Pixel-RMSE", -1)', source)
        self.assertIn('("clip_cosine", "CLIP", +1)', source)
        self.assertIn('("lpips", "LPIPS", -1)', source)
        self.assertIn('("dino_cosine", "DINO", +1)', source)
        self.assertIn("from .plot_das_top10_with_five_methods import METHODS, selections", source)

    def test_all5000_cache_and_outputs(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('pixel_lpips_all5000_q{query:03d}.npz', source)
        self.assertIn('perceptual_top1000_indices.npz', source)
        self.assertIn('per_query_overlap.csv', source)
        self.assertIn('summary.csv', source)
        self.assertIn('report.txt', source)


if __name__ == "__main__":
    unittest.main()
