from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Shape3DSimpleLoss10x1JobTest(unittest.TestCase):
    def test_launcher_uses_full_range_uniform_10x1_contract(self) -> None:
        source = (
            ROOT
            / "3dshapes/tacc/rtx_small/run_simple_loss_10x1_five_method_lds_q0_99.sh"
        ).read_text()
        self.assertIn("uniform_timesteps(1000,10)", source)
        self.assertIn("full valid range t=0..999", source)
        self.assertIn("--simple-loss-num-mc 10", source)
        self.assertIn("--uniform-split-simple-loss-probes", source)
        self.assertIn("--model-major-simple-loss", source)
        self.assertIn("--expected-probes 10", source)
        self.assertIn("--target-name simple_loss_10x1", source)
        self.assertIn("SIMPLE_LOSS_EVAL_CHUNK_SIZE:-10", source)
        self.assertIn("simple_loss_10x1_t0_999_uniform_split_seed0", source)
        self.assertEqual(source.count("--transforms both"), 2)
        self.assertEqual(source.count("--shape3d-pint-banks both"), 2)
        self.assertIn(
            "endpoint_contarfactual,traj_contarfactual,simple_loss_10x1,noise_trajectory",
            source,
        )
        self.assertIn(
            "shape3d_twelve_config_lds_all_targets_simple_loss_10x1_t0_999_q0_99.txt",
            source,
        )

    def test_exporter_accepts_10x1_target(self) -> None:
        source = (ROOT / "script/export_pint_lds_txt.py").read_text()
        self.assertIn('"simple_loss_10x1"', source)


if __name__ == "__main__":
    unittest.main()
