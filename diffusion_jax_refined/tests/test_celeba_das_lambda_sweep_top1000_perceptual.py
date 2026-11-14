from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "diffusion_jax_refined"
    / "celeba"
    / "script"
    / "analyze_das_lambda_sweep_top1000_endpoint_perceptual.py"
)


class CelebaDASLambdaSweepTop1000PerceptualTest(unittest.TestCase):
    def test_contract_is_full10k_sm_lambda_sweep(self) -> None:
        text = SCRIPT.read_text()
        self.assertIn("das_50x1_mc1_full10k_ema_sherman_morrison", text)
        self.assertIn("default=list(range(100))", text)
        self.assertIn('"--lambdas",', text)
        self.assertIn('f"lambda_{lambda_tag(damping)}"', text)
        self.assertIn("selections[(query_id, damping)]", text)

    def test_outputs_resumable_per_query_and_summary_files(self) -> None:
        text = SCRIPT.read_text()
        self.assertIn('f"pixel_lpips_q{query_id:03d}.npz"', text)
        self.assertIn('output_dir / "per_query.csv"', text)
        self.assertIn('output_dir / "summary.csv"', text)
        self.assertIn('output_dir / "report.txt"', text)
        self.assertIn("BEST MEAN SIMILARITY", text)


if __name__ == "__main__":
    unittest.main()
