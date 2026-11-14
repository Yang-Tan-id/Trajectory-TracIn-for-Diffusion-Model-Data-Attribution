import ast
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "src/sd_artbench/plot_das_vs_five_method_output_grid.py"
)


class DasVsFiveMethodOutputGridTest(unittest.TestCase):
    def test_seven_columns_and_expected_files(self):
        source = SCRIPT.read_text(encoding="utf-8")
        ast.parse(source)
        for label in ("Output", "PINT", "TracInCP", "GAS", "ReTrac", "Random", "DAS"):
            self.assertIn(f'"{label}"', source)
        self.assertIn('("das_lambda_1000.png",)', source)
        self.assertIn('"endpoint_pollute_query_train_l2.png"', source)
        self.assertIn('"endpoint_tracin_raw.png"', source)
        self.assertIn('"endpoint_tracin_query_train_l2.png"', source)
        self.assertIn('"retrac_query_train_l2.png"', source)

    def test_defaults_to_q0_q27_and_large_output_gap(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("default=tuple(range(28))", source)
        self.assertIn("default=44", source)
        self.assertIn("args.output_gap < 2 * args.gap", source)
        self.assertIn('print("[row labels] omitted")', source)
        self.assertIn('print("[seeds] omitted")', source)


if __name__ == "__main__":
    unittest.main()
