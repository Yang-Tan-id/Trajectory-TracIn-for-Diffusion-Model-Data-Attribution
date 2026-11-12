import ast
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "src/sd_artbench/plot_das_top10_with_five_methods.py"
)


class DasTop10PlotTest(unittest.TestCase):
    def test_six_rows_and_directions(self):
        source = SCRIPT.read_text(encoding="utf-8")
        ast.parse(source)
        for label in ("PINT", "TracInCP", "GAS", "ReTrac", "Random", "DAS"):
            self.assertIn(f'("{label}"', source)
        self.assertIn('("PINT", "pollute", "query_train_l2", +1)', source)
        self.assertIn('("TracInCP", "tracin", "raw", -1)', source)
        self.assertIn('("GAS", "tracin", "query_train_l2", -1)', source)
        self.assertIn('("ReTrac", "retrac", None, -1)', source)

    def test_uses_lambda1000_das_and_seeded_random(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('scores/lambda_1000/scores.npz', source)
        self.assertIn('np.random.SeedSequence([args.random_seed, query])', source)
        self.assertIn('default=2026', source)

    def test_includes_full_data_output_as_separate_reference(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('"--endpoint-image-root"', source)
        self.assertIn('"full-data.png"', source)
        self.assertIn('output_label = "Output"', source)
        self.assertIn('output_gap = 40', source)


if __name__ == "__main__":
    unittest.main()
