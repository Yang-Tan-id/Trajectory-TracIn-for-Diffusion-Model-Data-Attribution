from __future__ import annotations

import importlib.util
import subprocess
import unittest
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
PLAN_PATH = REPO / "diffusion_jax_refined/celeba/script/celeba_random_top1000_70_plan.py"
WORKER_PATH = REPO / "diffusion_jax_refined/celeba/script/run_endpoint_top500_removal.py"
RUNNER_PATH = REPO / "diffusion_jax_refined/celeba/tacc/rtx_small/run_celeba_random_top1000_q0_69_rtx_small.sh"
SLOT_PATH = REPO / "diffusion_jax_refined/celeba/tacc/rtx_small/run_celeba_random_top1000_q0_69_slot.sh"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CelebaRandomTop1000RtxSmallTest(unittest.TestCase):
    def test_plan_has_exactly_q0_through_q69(self) -> None:
        plan = load_module(PLAN_PATH, "celeba_random_top1000_plan_test")
        tasks = plan.build_tasks()
        self.assertEqual(len(tasks), 70)
        self.assertEqual(len(set(tasks)), 70)
        self.assertEqual({method for method, _ in tasks}, {"random_full10k_seeded"})
        self.assertEqual([query_id for _, query_id in tasks], list(range(70)))

    def test_random_selection_is_deterministic_unique_and_full10k(self) -> None:
        worker = load_module(WORKER_PATH, "celeba_random_top1000_worker_test")
        first = worker.select_random_removal(topk=1000, random_seed=20261025, query_id=0)
        repeat = worker.select_random_removal(topk=1000, random_seed=20261025, query_id=0)
        other = worker.select_random_removal(topk=1000, random_seed=20261025, query_id=1)
        np.testing.assert_array_equal(first, repeat)
        self.assertEqual(first.shape, (1000,))
        self.assertEqual(np.unique(first).size, 1000)
        self.assertGreaterEqual(int(first.min()), 0)
        self.assertLess(int(first.max()), 10_000)
        self.assertFalse(np.array_equal(first, other))
        self.assertTrue(np.any(first >= 5_000))

    def test_shell_scripts_are_valid_and_use_two_gpu_slots(self) -> None:
        for path in (RUNNER_PATH, SLOT_PATH):
            subprocess.run(["bash", "-n", str(path)], check=True)
        runner = RUNNER_PATH.read_text()
        slot = SLOT_PATH.read_text()
        self.assertIn("#SBATCH -p rtx-small", runner)
        self.assertIn("#SBATCH -n 2", runner)
        self.assertNotIn("#SBATCH --gres", runner)
        self.assertIn("for slot in 0 1", runner)
        self.assertIn("--random-seed", runner)
        self.assertIn("--random-seed", slot)
        self.assertIn("--topk 1000", slot)

    def test_random_job_is_separate_and_resumable(self) -> None:
        worker = WORKER_PATH.read_text()
        runner = RUNNER_PATH.read_text()
        self.assertIn('"random_full10k_seeded"', worker)
        self.assertIn("uniform_random_without_replacement", worker)
        self.assertIn("if target_checkpoint.is_file()", worker)
        self.assertIn("if metrics_path.is_file()", worker)
        self.assertIn("endpoint_random_top1000_removal_q0_69", runner)


if __name__ == "__main__":
    unittest.main()
