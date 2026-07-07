from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TACC = ROOT / "cifar2" / "tacc"


class TestTaccScriptsStatic(unittest.TestCase):
    def read(self, name: str) -> str:
        return (TACC / name).read_text()

    def test_h100_scripts_do_not_use_gres(self):
        for script in TACC.glob("*.sh"):
            with self.subTest(script=script.name):
                self.assertNotIn("--gres", script.read_text())

    def test_k8000_training_uses_four_nodes_and_sixteen_tasks(self):
        text = self.read("05_train_lds_k8000_h100.sh")
        self.assertRegex(text, r"#SBATCH\s+--nodes=4")
        self.assertRegex(text, r"#SBATCH\s+--ntasks-per-node=4")
        self.assertIn('LDS_K="${LDS_K:-8000}"', text)
        self.assertIn('ibrun -n 1 -o "${slot}"', text)

    def test_combined_and_per_seed_eval_use_one_node_four_tasks(self):
        for name in ("03_lds_eval_h100.sh", "04_lds_eval_by_seed_h100.sh", "06_eval_lds_k8000_h100.sh"):
            text = self.read(name)
            with self.subTest(script=name):
                self.assertRegex(text, r"#SBATCH\s+--nodes=1")
                self.assertRegex(text, r"#SBATCH\s+--ntasks-per-node=4")
                self.assertIn("MAX_PARALLEL_EVAL_TASKS", text)

    def test_scripts_activate_conda_and_unset_pythonpath(self):
        for name in ("01_train_lds_h100.sh", "03_lds_eval_h100.sh", "04_lds_eval_by_seed_h100.sh", "05_train_lds_k8000_h100.sh"):
            text = self.read(name)
            with self.subTest(script=name):
                self.assertIn("unset PYTHONPATH", text)
                self.assertIn('source "${SCRATCH}/miniforge3/etc/profile.d/conda.sh"', text)
                self.assertIn('conda activate "${CONDA_ENV_PATH}"', text)

    def test_scripts_fail_fast_and_refuse_overwrite(self):
        for name in ("01_train_lds_h100.sh", "02_sample_attribution_h100.sh", "03_lds_eval_h100.sh", "04_lds_eval_by_seed_h100.sh", "05_train_lds_k8000_h100.sh"):
            text = self.read(name)
            with self.subTest(script=name):
                self.assertIn("set -euo pipefail", text)
                self.assertRegex(text, r"Refusing to overwrite|ALLOW_OVERWRITE")

    def test_sample_attribution_shards_are_explicit(self):
        text = self.read("02_sample_attribution_h100.sh")
        self.assertIn('ATTR_SHARD="${ATTR_SHARD:?', text)
        self.assertIn('[[ "${ATTR_SHARD}" == "1" || "${ATTR_SHARD}" == "2" ]]', text)
        self.assertIn('if [[ "${ATTR_SHARD}" == "1" ]]', text)
        self.assertIn('if [[ "${ATTR_SHARD}" == "2" ]]; then', text)
        self.assertIn('validate_sample', text)

    def test_combined_eval_overwrite_guard_is_target_specific(self):
        text = self.read("03_lds_eval_h100.sh")
        self.assertIn('/lds/${algorithm}/${LDS_TARGET_FUNCTION}', text)
        self.assertIn('Refusing to overwrite ${out_dir}', text)
        self.assertNotIn('Refusing to overwrite ${eval_dir}', text)


if __name__ == "__main__":
    unittest.main()
