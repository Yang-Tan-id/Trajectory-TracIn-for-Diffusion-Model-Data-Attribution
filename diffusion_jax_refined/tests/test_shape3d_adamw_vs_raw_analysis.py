from __future__ import annotations

import ast
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / (
    "diffusion_jax_refined/3dshapes/script/"
    "analyze_adamw_vs_raw_simple_noise.py"
)
SOURCE = SCRIPT.read_text()


class Shape3DAdamWVsRawAnalysisTest(unittest.TestCase):
    def test_five_method_pairs_and_directions(self) -> None:
        self.assertEqual(SOURCE.count("Method(\n"), 5)
        self.assertIn('label="PINT / RAW"', SOURCE)
        self.assertIn('label="PINT / BOTH-L2"', SOURCE)
        self.assertIn('label="TRACIN / RAW"', SOURCE)
        self.assertIn('label="TRACIN / BOTH-L2"', SOURCE)
        self.assertIn('label="RETRAC / BOTH-L2"', SOURCE)
        self.assertEqual(SOURCE.count('raw_sign="p1"'), 5)
        self.assertEqual(SOURCE.count('adamw_sign="p1"'), 2)
        self.assertEqual(SOURCE.count('adamw_sign="m1"'), 3)

    def test_namespaces_and_variants(self) -> None:
        self.assertIn("recreate_raw_polluted_endpoint", SOURCE)
        self.assertIn("recreate_adamw_full_polluted_endpoint", SOURCE)
        self.assertIn("endpoint_tracin_train100x1_query100x1_q0_99", SOURCE)
        self.assertIn("paper_retrac_exact4_endpoint100x1_q0_99", SOURCE)
        self.assertEqual(SOURCE.count('variant="query_train_l2"'), 3)

    def test_script_syntax_and_statistical_contract(self) -> None:
        ast.parse(SOURCE)
        self.assertIn("def paired_label_swap(", SOURCE)
        self.assertIn(
            'parser.add_argument("--permutations", type=int, default=1_000_000)',
            SOURCE,
        )
        self.assertIn("difference = adamw - raw", SOURCE)


if __name__ == "__main__":
    unittest.main()
