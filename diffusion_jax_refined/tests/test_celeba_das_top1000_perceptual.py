from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
CELEBA = ROOT / "diffusion_jax_refined" / "celeba"


class CelebaDASTop1000PerceptualTest(unittest.TestCase):
    def test_removal_worker_has_fixed_full10k_das_contract(self) -> None:
        text = (CELEBA / "script" / "run_das_ema_top1000_removal.py").read_text()
        self.assertIn('"das_ema_lambda200_full10k_pos"', text)
        self.assertIn('"das_50x1_mc1_full10k_ema"', text)
        self.assertIn('/ "lambda_200"', text)
        self.assertIn("lambda selected once by mean Q0-Q99 simple_loss_50x1 LDS", text)

    def test_runner_covers_q0_q69_and_validates_before_training(self) -> None:
        text = (
            CELEBA
            / "script"
            / "run_das_ema_lambda200_full10k_top1000_q0_69_4gpu.sh"
        ).read_text()
        self.assertIn("$(seq -s, 0 69)", text)
        self.assertIn('output_group="das_top1000_removal_q0_69"', text)
        self.assertIn('topk=1000', text)
        self.assertIn('--stage validate', text)
        self.assertIn('--stage all', text)

    def test_generation_and_perceptual_reports_include_das(self) -> None:
        for name in (
            "evaluate_top1000_q0_69_generate.py",
            "evaluate_top1000_q0_69_perceptual.py",
        ):
            text = (CELEBA / "script" / name).read_text()
            self.assertIn('"das_ema_lambda200_full10k_pos"', text)
            self.assertIn('"DAS-EMA-L200"', text)


if __name__ == "__main__":
    unittest.main()
