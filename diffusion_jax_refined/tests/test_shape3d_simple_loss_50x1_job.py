from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Shape3DSimpleLoss50x1JobTest(unittest.TestCase):
    def test_launcher_matches_celeba_probe_contract(self) -> None:
        source = (
            ROOT
            / "3dshapes/tacc/rtx_small/run_simple_loss_50x1_five_method_lds_q0_99.sh"
        ).read_text()
        self.assertIn("uniform_timesteps(1000,50)", source)
        self.assertIn("--simple-loss-num-mc 50", source)
        self.assertIn("--uniform-split-simple-loss-probes", source)
        self.assertIn("--model-major-simple-loss", source)
        self.assertIn("--expected-probes 50", source)
        self.assertIn("--target-name simple_loss_50x1", source)
        self.assertEqual(source.count("--transforms adamw"), 2)
        self.assertIn("simple_loss_50x1_uniform_split_seed0", source)

    def test_core_uses_jax_split_keys(self) -> None:
        source = (ROOT / "legacy_jax/LDS/DM_cifar_lds.py").read_text()
        self.assertIn("simple_loss_uniform_split_probes", source)
        self.assertIn('"uniform_timesteps_jax_split"', source)
        self.assertIn("jax.random.split(", source)


if __name__ == "__main__":
    unittest.main()
