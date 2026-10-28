from __future__ import annotations

import ast
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / (
    "diffusion_jax_refined/celeba/script/"
    "analyze_endpoint_full10k_cross_method_overlap.py"
)
SOURCE = SCRIPT.read_text()


class CelebAPintTracinOverlapDirectionsTest(unittest.TestCase):
    def test_syntax_and_direction_contract(self) -> None:
        ast.parse(SOURCE)
        self.assertIn('if transform == "raw_gradient":', SOURCE)
        self.assertIn("tracin_ranking = tracin_scores", SOURCE)
        self.assertIn("tracin_ranking = -tracin_scores", SOURCE)
        self.assertIn("pollute_ranking = pollute_scores", SOURCE)

    def test_full10k_top1000_four_variants(self) -> None:
        self.assertIn('default=",".join(str(value) for value in range(72))', SOURCE)
        self.assertIn('parser.add_argument("--topk", type=int, default=1000)', SOURCE)
        self.assertIn('expected = np.arange(10_000', SOURCE)
        self.assertEqual(SOURCE.count('("raw", "score")'), 1)
        self.assertIn('("query_train_l2", "score_query_train_l2_normalized")', SOURCE)

    def test_report_documents_both_directions(self) -> None:
        self.assertIn(
            '"descending and AdamW ranks ascending."', SOURCE
        )
        self.assertIn('"overlap_fraction": overlap / args.topk', SOURCE)
        self.assertIn("ranking_spearman", SOURCE)


if __name__ == "__main__":
    unittest.main()
