from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Shape3DSimpleLoss50x1JobTest(unittest.TestCase):
    def test_launcher_uses_uniform_t0_500_probe_contract(self) -> None:
        source = (
            ROOT
            / "3dshapes/tacc/rtx_small/run_simple_loss_50x1_five_method_lds_q0_99.sh"
        ).read_text()
        self.assertIn("uniform_timesteps(501,50)", source)
        self.assertLess(source.index('cd "$repo"'), source.index("uniform_timesteps(501,50)"))
        self.assertIn('runtime_root="${SLURM_TMPDIR', source)
        self.assertIn("diffusion_jax_refined/3dshapes/result'", source)
        self.assertIn('ln -s "$source_repo/diffusion_jax_refined/3dshapes/result"', source)
        self.assertIn("LDS_TRUE_F_LOG_ROOT", source)
        self.assertIn("--simple-loss-num-mc 50", source)
        self.assertIn("--uniform-split-simple-loss-probes", source)
        self.assertIn("--model-major-simple-loss", source)
        self.assertIn("--expected-probes 50", source)
        self.assertIn("--target-name simple_loss_50x1", source)
        self.assertEqual(source.count("--transforms both"), 2)
        self.assertEqual(source.count("--shape3d-pint-banks both"), 2)
        self.assertIn(
            "endpoint_contarfactual,traj_contarfactual,simple_loss_50x1,noise_trajectory",
            source,
        )
        self.assertIn("simple_loss_50x1_t0_500_uniform_split_seed0", source)
        self.assertIn(
            "shape3d_twelve_config_lds_all_targets_simple_loss_50x1_t0_500_q0_99.txt",
            source,
        )

    def test_worker_failure_is_copied_to_outer_stderr(self) -> None:
        source = (ROOT / "3dshapes/script/run_lds_true_f.py").read_text()
        self.assertIn('os.environ.get(\n            "LDS_TRUE_F_LOG_ROOT"', source)
        self.assertIn("FAILED WORKER", source)
        self.assertIn("text[-12000:]", source)

    def test_core_uses_jax_split_keys(self) -> None:
        source = (ROOT / "legacy_jax/LDS/DM_cifar_lds.py").read_text()
        self.assertIn("simple_loss_uniform_split_probes", source)
        self.assertIn('"uniform_timesteps_jax_split"', source)
        self.assertIn("jax.random.split(", source)
        self.assertIn("and not self.simple_loss_uniform_split_probes", source)

        model_major = (
            ROOT / "3dshapes/script/evaluate_simple_loss_model_major.py"
        ).read_text()
        self.assertIn(
            "simple_loss_pair_timesteps_noise=not args.uniform_split_probes",
            model_major,
        )

    def test_exporter_has_both_shape3d_pint_banks(self) -> None:
        source = (ROOT / "script/export_pint_lds_txt.py").read_text()
        self.assertIn('"PINT-10x1 / RAW"', source)
        self.assertIn('"PINT-100x1 / RAW"', source)
        self.assertIn("raw_namespace=None", source)
        self.assertIn(
            '"traj_tracin_recreate_raw_polluted_endpoint_delta_l2normalized_"',
            source,
        )
        self.assertIn('"timestamp_sum_squared_q0_99"', source)
        self.assertIn("--shape3d-pint-banks", source)


if __name__ == "__main__":
    unittest.main()
