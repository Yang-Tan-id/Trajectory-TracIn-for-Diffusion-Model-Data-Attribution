from __future__ import annotations

import ast
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / (
    "diffusion_jax_refined/celeba/script/"
    "analyze_endpoint_tracin_raw_vs_adamw_overlap.py"
)
SOURCE = SCRIPT.read_text()


class CelebATracinRawVsAdamWOverlapTest(unittest.TestCase):
    def test_syntax_and_full10k_contract(self) -> None:
        ast.parse(SOURCE)
        self.assertIn('default=",".join(map(str, range(72)))', SOURCE)
        self.assertIn('parser.add_argument("--topk", type=int, default=1000)', SOURCE)
        self.assertIn('expected = np.arange(10_000', SOURCE)
        self.assertIn('for transform in ("raw_gradient", "adamw_full")', SOURCE)
        self.assertIn(
            '"--method", choices=("tracin", "pint"), default="tracin"', SOURCE
        )

    def test_ranking_directions_and_variants(self) -> None:
        self.assertIn("raw_oriented = raw_scores", SOURCE)
        self.assertIn('if args.method == "tracin":', SOURCE)
        self.assertIn("adamw_oriented = -adamw_scores", SOURCE)
        self.assertIn("adamw_oriented = adamw_scores", SOURCE)
        self.assertIn("endpoint_pollute_{transform}", SOURCE)
        self.assertIn("timestamp_sum_squared_", SOURCE)
        self.assertIn('("raw", "score")', SOURCE)
        self.assertIn('("query_l2", "score_query_normalized")', SOURCE)
        self.assertIn('("train_l2", "score_train_l2_normalized")', SOURCE)
        self.assertIn(
            '("query_train_l2", "score_query_train_l2_normalized")', SOURCE
        )

    def test_outputs_overlap_and_rank_agreement(self) -> None:
        self.assertIn('"overlap_fraction": overlap / args.topk', SOURCE)
        self.assertIn('"jaccard": overlap / (2 * args.topk - overlap)', SOURCE)
        self.assertIn('"ranking_spearman": spearman(', SOURCE)
        self.assertIn('"overlap_report.txt"', SOURCE)


if __name__ == "__main__":
    unittest.main()
