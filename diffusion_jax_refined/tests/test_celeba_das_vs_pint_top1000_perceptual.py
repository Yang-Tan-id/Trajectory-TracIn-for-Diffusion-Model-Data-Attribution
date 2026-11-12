from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "diffusion_jax_refined/celeba/script/"
    / "analyze_das_vs_pint_top1000_endpoint_perceptual.py"
)


class CelebaDASVsPINTTop1000PerceptualTest(unittest.TestCase):
    def test_fixed_score_contracts_and_no_removal(self) -> None:
        text = SCRIPT.read_text()
        self.assertIn('"das_50x1_mc1_full10k_ema"', text)
        self.assertIn('"lambda_200"', text)
        self.assertIn('"adamw_full"', text)
        self.assertIn('("raw", "query_train_l2")', text)
        self.assertNotIn("run_endpoint_top500_removal", text)

    def test_full_query_and_metric_contract(self) -> None:
        text = SCRIPT.read_text()
        self.assertIn("default=list(range(100))", text)
        self.assertIn('parser.add_argument("--topk", type=int, default=1000)', text)
        for metric in ("clip_cosine", "dino_cosine", "pixel_rmse", "lpips"):
            self.assertIn(metric, text)


if __name__ == "__main__":
    unittest.main()
