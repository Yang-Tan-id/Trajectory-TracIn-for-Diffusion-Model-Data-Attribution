from __future__ import annotations

import unittest
from pathlib import Path


class TestCelebaSimpleLoss50x1FiveMethodJob(unittest.TestCase):
    def setUp(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self.job = (
            root
            / "celeba/script/run_simple_loss_50x1_five_method_lds_q0_99_4gpu.sh"
        ).read_text()
        self.evaluator = (
            root / "celeba/script/evaluate_five_method_simple_loss_50x1.py"
        ).read_text()
        self.probe_helpers = (
            root / "common/predicted_noise_kl.py"
        ).read_text()

    def test_original_fifty_probe_definition_is_preserved(self) -> None:
        self.assertIn(
            "np.linspace(total_timesteps - 1, 0, count, dtype=np.int32)",
            self.probe_helpers,
        )
        self.assertIn("100003 * int(query_id)", self.probe_helpers)
        self.assertIn("probe_count=50", self.job)
        self.assertIn("probe_seed=2026", self.job)
        self.assertIn('target="simple_loss_50x1"', self.job)

    def test_only_five_method_variants_are_scored(self) -> None:
        self.assertIn('method_configs("celeba")', self.evaluator)
        self.assertIn('"raw": "score"', self.evaluator)
        self.assertIn(
            '"query_train_l2": "score_query_train_l2_normalized"',
            self.evaluator,
        )
        self.assertNotIn('"query_l2":', self.evaluator)
        self.assertNotIn('"train_l2":', self.evaluator)

    def test_report_replaces_ten_step_simple_loss(self) -> None:
        export_line = next(
            line for line in self.job.splitlines() if line.strip().startswith("--targets ")
        )
        self.assertIn("simple_loss_50x1", export_line)
        self.assertNotIn(",simple_loss,", export_line)
        self.assertIn("celeba_five_method_lds_q0_99_all.txt", self.job)


if __name__ == "__main__":
    unittest.main()
