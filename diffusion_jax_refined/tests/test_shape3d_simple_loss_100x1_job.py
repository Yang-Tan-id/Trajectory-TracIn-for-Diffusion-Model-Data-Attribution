from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Shape3DSimpleLoss100x1JobTest(unittest.TestCase):
    def test_job_is_scoped_and_uses_paired_shared_probes(self) -> None:
        path = (
            ROOT
            / "3dshapes/tacc/rtx_small/run_simple_loss_100x1_five_method_lds_q0_99.sh"
        )
        source = path.read_text()
        self.assertIn("--target-functions simple_loss", source)
        self.assertIn("--simple-loss-num-mc 100", source)
        self.assertIn("--pair-simple-loss-probes", source)
        self.assertIn("--target-cache-namespace", source)
        self.assertIn("--gpus 0,1,2,3", source)
        self.assertIn("evaluate_five_method_simple_loss_100x1.py", source)
        self.assertIn("shape3d_five_method_lds_q0_99_all.txt", source)

    def test_selected_evaluator_has_only_requested_variants(self) -> None:
        source = (
            ROOT
            / "3dshapes/script/evaluate_five_method_simple_loss_100x1.py"
        ).read_text()
        self.assertIn('"raw": "score"', source)
        self.assertIn(
            '"query_train_l2": "score_query_train_l2_normalized"', source
        )
        self.assertNotIn('"query_l2":', source)
        self.assertNotIn('"train_l2":', source)


if __name__ == "__main__":
    unittest.main()
