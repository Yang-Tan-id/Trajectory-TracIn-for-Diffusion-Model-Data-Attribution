from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC = PACKAGE_ROOT / "src"
sys.path.insert(0, str(SRC))

from sd_artbench import artbench_random_top1000_60_plan as PLAN
from sd_artbench import topk_removal


RUNNER = PACKAGE_ROOT / "tacc/rtx_small/run_artbench_random_top1000_60_rtx_small.sh"
SLOT = PACKAGE_ROOT / "tacc/rtx_small/run_artbench_random_top1000_60_slot.sh"
SUBMITTER = PACKAGE_ROOT / "tacc/rtx_small/submit_artbench_random_top1000_60_rtx_small.sh"
SETUP = PACKAGE_ROOT / "tacc/rtx_small/setup_artbench_rtx_blackwell_env.sh"


class ArtBenchRandomTop100060RtxSmallTest(unittest.TestCase):
    def test_exact_random_task_plan_and_two_disjoint_shards(self) -> None:
        tasks = PLAN.build_tasks()
        self.assertEqual(len(tasks), 60)
        self.assertEqual(len(set(tasks)), 60)
        self.assertEqual(
            tasks,
            [(topk_removal.RANDOM_METHOD, query_id) for query_id in range(60)],
        )
        shards = [
            PLAN.build_tasks(shard_index=index, shard_count=2)
            for index in range(2)
        ]
        self.assertEqual([len(shard) for shard in shards], [30, 30])
        flattened = [task for shard in shards for task in shard]
        self.assertEqual(set(flattened), set(tasks))
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_random_top1000_is_reproducible_and_query_specific(self) -> None:
        indices = np.arange(5000, dtype=np.int64)
        first = topk_removal.random_scores(
            query_index=0, base_indices=indices, random_seed=2026
        )[0]
        repeated = topk_removal.random_scores(
            query_index=0, base_indices=indices, random_seed=2026
        )[0]
        other = topk_removal.random_scores(
            query_index=1, base_indices=indices, random_seed=2026
        )[0]
        np.testing.assert_array_equal(first, repeated)
        self.assertFalse(np.array_equal(first, other))
        removed, _, _ = topk_removal.select_topk(
            first, indices, top_k=1000, ranking_sign=1
        )
        self.assertEqual(len(removed), 1000)
        self.assertEqual(len(np.unique(removed)), 1000)

    def test_audit_accepts_one_complete_synthetic_shard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for method, query_id in PLAN.build_tasks(shard_index=0, shard_count=60):
                run_dir = root / method / f"query-{query_id:03d}" / "topk-1000"
                final_dir = run_dir / "model/final"
                final_dir.mkdir(parents=True)
                (run_dir / "TRAINING_COMPLETE").write_text("complete\n")
                (run_dir / "retained-indices.pkl").write_bytes(b"indices")
                (final_dir / "pytorch_lora_weights.safetensors").write_bytes(b"weights")
                (run_dir / "removal-manifest.json").write_text(
                    '{"status":"trained","top_k":1000,'
                    '"method":"random_full5000_seeded","random_seed":2026}\n'
                )
            PLAN.audit(root, top_k=1000, shard_index=0, shard_count=60)

    def test_shell_contracts_are_rtx_two_slot_resume_safe(self) -> None:
        for script in (RUNNER, SLOT, SUBMITTER, SETUP):
            result = subprocess.run(
                ["bash", "-n", str(script)], capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, result.stderr)
        runner = RUNNER.read_text()
        slot = SLOT.read_text()
        submitter = SUBMITTER.read_text()
        setup = SETUP.read_text()
        self.assertIn("#SBATCH -p rtx-small", runner)
        self.assertIn("#SBATCH -n 2", runner)
        self.assertNotIn("#SBATCH --gres", runner)
        self.assertIn("for slot in 0 1", runner)
        self.assertIn("--random-seed 2026", slot)
        self.assertIn("--stage all", slot)
        self.assertIn("--mode audit", runner)
        self.assertIn("for shard_index in 0 1", submitter)
        self.assertIn('"$shard_index" 2', submitter)
        self.assertNotIn("for shard_index in 0 1 2", submitter)
        self.assertNotIn("rsync", submitter)
        self.assertIn("sd_artbench_rtx", runner)
        self.assertIn("sd_artbench_rtx", submitter)
        self.assertIn('"sm_120" not in architectures', runner)
        self.assertIn("matrix @ matrix", runner)
        self.assertIn("torch==2.7.0 torchvision==0.22.0", setup)
        self.assertIn("https://download.pytorch.org/whl/cu128", setup)


if __name__ == "__main__":
    unittest.main()
