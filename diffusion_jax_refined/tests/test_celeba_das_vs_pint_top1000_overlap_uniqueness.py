from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "diffusion_jax_refined/celeba/script/"
    / "analyze_das_vs_pint_top1000_overlap_uniqueness.py"
)


class CelebaDASVsPINTOverlapUniquenessTest(unittest.TestCase):
    def test_full10k_q0_q59_contract(self) -> None:
        text = SCRIPT.read_text()
        self.assertIn("default=list(range(60))", text)
        self.assertIn('parser.add_argument("--topk", type=int, default=1000)', text)
        self.assertIn('"das_50x1_mc1_full10k_ema"', text)
        self.assertIn('("raw", "query_train_l2")', text)

    def test_reports_overlap_and_cross_query_uniqueness(self) -> None:
        text = SCRIPT.read_text()
        for field in (
            "overlap_fraction",
            "jaccard",
            "union_coverage",
            "singleton_count",
            "mean_pairwise_overlap_fraction",
        ):
            self.assertIn(field, text)


if __name__ == "__main__":
    unittest.main()
