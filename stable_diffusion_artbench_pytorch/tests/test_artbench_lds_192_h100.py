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

from sd_artbench import artbench_lds_192_plan as plan


RUNNER = PACKAGE_ROOT / "tacc/h100/run_artbench_lds_192_h100.sh"
SLOT = PACKAGE_ROOT / "tacc/h100/run_artbench_lds_192_slot.sh"
SUBMITTER = PACKAGE_ROOT / "tacc/h100/submit_artbench_lds_192_h100.sh"


class ArtBenchLDS192H100Test(unittest.TestCase):
    def test_exact_task_plan(self) -> None:
        tasks = plan.build_tasks()
        self.assertEqual(len(tasks), 192)
        self.assertEqual(len(set(tasks)), 192)
        for subset_bank in (0, 1, 2):
            self.assertEqual(
                [subset for bank, subset in tasks if bank == subset_bank],
                list(range(64)),
            )

    def test_three_deterministic_distinct_seed_banks(self) -> None:
        banks = [plan.bank_subset_seeds(bank) for bank in (0, 1, 2)]
        self.assertTrue(all(values.shape == (64,) for values in banks))
        self.assertEqual(len(set(np.concatenate(banks).tolist())), 192)
        np.testing.assert_array_equal(banks[0], plan.bank_subset_seeds(0))

    def test_each_membership_is_exactly_half_and_stratified(self) -> None:
        base = np.arange(5000, dtype=np.int64)
        labels = np.asarray(["post_impressionism"] * 2500 + ["ukiyo_e"] * 2500)
        memberships = []
        for subset_bank in (0, 1, 2):
            seed, kept = plan.generate_subset_indices(
                base, labels, subset_bank=subset_bank, subset_id=7
            )
            self.assertGreaterEqual(seed, 0)
            self.assertEqual(len(kept), 2500)
            self.assertEqual(len(np.unique(kept)), 2500)
            selected = labels[kept]
            self.assertEqual(int(np.sum(selected == "post_impressionism")), 1250)
            self.assertEqual(int(np.sum(selected == "ukiyo_e")), 1250)
            memberships.append(kept)
        self.assertFalse(np.array_equal(memberships[0], memberships[1]))
        self.assertFalse(np.array_equal(memberships[1], memberships[2]))

    def test_train_contract_is_final_only_and_model_resumable(self) -> None:
        source = (SRC / "sd_artbench/artbench_lds_192_plan.py").read_text()
        self.assertIn('"--num-checkpoints",\n        "0"', source)
        self.assertIn('"--batch-size",\n        "8"', source)
        self.assertIn('"--gradient-accumulation-steps",\n        "8"', source)
        self.assertIn('marker = output_dir / "TRAINING_COMPLETE"', source)
        self.assertIn("if marker.is_file() and final_weights_exist(output_dir)", source)

    def test_h100_runner_uses_16_slots_and_24_hour_limit(self) -> None:
        runner = RUNNER.read_text()
        slot = SLOT.read_text()
        self.assertIn("#SBATCH -N 4", runner)
        self.assertIn("#SBATCH -n 16", runner)
        self.assertIn("#SBATCH --ntasks-per-node=4", runner)
        self.assertIn("#SBATCH -t 24:00:00", runner)
        self.assertIn("for slot in $(seq 0 15)", runner)
        self.assertIn("12 models per slot", runner)
        self.assertIn("task_index+=slot_count", slot)
        self.assertIn("--mode audit", runner)

    def test_submitter_never_contacts_school_server(self) -> None:
        source = SUBMITTER.read_text()
        self.assertNotIn("rsync", source)
        self.assertNotIn("ssh", source)
        self.assertIn("--mode validate", source)
        self.assertIn("sbatch --chdir", source)

    def test_shell_syntax(self) -> None:
        for script in (RUNNER, SLOT, SUBMITTER):
            result = subprocess.run(
                ["bash", "-n", str(script)], capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
