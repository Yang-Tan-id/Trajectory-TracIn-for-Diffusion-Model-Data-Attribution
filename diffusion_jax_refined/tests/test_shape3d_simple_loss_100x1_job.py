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
        self.assertIn('SIMPLE_LOSS_EVAL_CHUNK_SIZE:-50', source)
        self.assertIn("--model-major-simple-loss", source)
        self.assertNotIn("LDS_MODEL_ROOT_OVERRIDE", source)
        self.assertNotIn("/dev/shm/shape3d_lds_", source)
        self.assertNotIn("rsync -aL", source)
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

    def test_shape3d_pint_scores_are_aligned_100x1(self) -> None:
        source = (
            ROOT.parent / "diffusion_jax_refined/script/export_pint_lds_txt.py"
        ).read_text()
        self.assertIn(
            '"timestamp_sum_squared_aligned100x1_q0_99"', source
        )
        self.assertIn(
            '"delta_l2normalized_timestamp_sum_squared_aligned100x1_q0_99"',
            source,
        )
        self.assertNotIn(
            '"timestamp_sum_squared_q0_99"\n        )\n        pint_adamw',
            source,
        )

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

    def test_model_major_runner_loads_checkpoint_outside_query_loop(self) -> None:
        source = (
            ROOT / "3dshapes/script/evaluate_simple_loss_model_major.py"
        ).read_text()
        checkpoint_loop = source.index(
            "for model_position, (model_id, subset_dir) in enumerate"
        )
        adapter_load = source.index(
            "target_adapter = evaluator._make_adapter(checkpoint)"
        )
        query_loop = source.index("for query, cache_path in missing")
        self.assertLess(checkpoint_loop, adapter_load)
        self.assertLess(adapter_load, query_loop)
        self.assertIn("evaluate_simple_loss_with_adapter", source)
        self.assertIn("cache_matches_probe_contract", source)
        self.assertIn('simple_loss_num_mc_terms", -1', source)
        self.assertIn("simple_loss_sampled_timesteps", source)
        self.assertIn("simple_loss_noise_key_seeds", source)


if __name__ == "__main__":
    unittest.main()
