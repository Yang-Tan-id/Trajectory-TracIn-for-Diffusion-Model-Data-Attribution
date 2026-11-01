from __future__ import annotations

import unittest
from pathlib import Path

from diffusion_jax_refined.common.simple_loss_probes import (
    manual_simple_loss_probe_pairs,
)


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
        self.assertIn("--gpus 0,1", source)
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

    def test_paired_probe_builder_produces_100_not_10000(self) -> None:
        timesteps = tuple(range(999, 899, -1))
        seeds = tuple(range(100000, 100100))
        paired = manual_simple_loss_probe_pairs(
            timesteps, seeds, paired=True
        )
        self.assertEqual(len(paired), 100)
        self.assertEqual(paired[0], (999, 100000))
        self.assertEqual(paired[-1], (900, 100099))

        cartesian = manual_simple_loss_probe_pairs(
            timesteps, seeds, paired=False
        )
        self.assertEqual(len(cartesian), 10000)


if __name__ == "__main__":
    unittest.main()
