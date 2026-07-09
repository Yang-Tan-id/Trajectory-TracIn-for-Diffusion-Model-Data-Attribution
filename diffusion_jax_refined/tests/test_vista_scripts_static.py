from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VISTA = ROOT / "cifar2" / "vista"


class TestVistaScriptsStatic(unittest.TestCase):
    def read(self, name: str) -> str:
        return (VISTA / name).read_text()

    def test_train_lds_requests_gh_16_nodes_50_percent(self):
        text = self.read("00_train_lds_50pct_vista.sh")
        self.assertIn("#SBATCH --partition=gh", text)
        self.assertIn("#SBATCH --account=CCR25021", text)
        self.assertIn("#SBATCH --nodes=16", text)
        self.assertIn("#SBATCH --time=12:00:00", text)
        self.assertIn('LDS_M="${LDS_M:-50}"', text)
        self.assertIn('LDS_K="${LDS_K:-5000}"', text)
        self.assertIn('LDS_SEEDS="${LDS_SEEDS:-$(seq -s \' \' 1 16)}"', text)

    def test_sample_attr_requests_24_nodes_and_normalized_traj(self):
        text = self.read("01_sample_and_attribute_vista.sh")
        self.assertIn("#SBATCH --partition=gh", text)
        self.assertIn("#SBATCH --nodes=24", text)
        self.assertIn("#SBATCH --time=24:00:00", text)
        self.assertIn('INITIAL_SEED="${INITIAL_SEED:-24}"', text)
        self.assertIn('TRAJ_QUERY_OBJECTIVE="${TRAJ_QUERY_OBJECTIVE:-trajectory_noise_squared_deviation_normalized}"', text)
        self.assertIn('QUERIES=("horse" "automobile" "horse,automobile")', text)
        self.assertIn('TRAJ_RANGES=("1-2000" "2001-4000" "4001-6000" "6001-8000" "8001-10000")', text)
        self.assertIn('ENDPOINT_ALGORITHMS=("das" "dtrak" "end_tracin")', text)

    def test_eval_depends_on_prior_jobs_and_uses_simple_loss_mc_10(self):
        eval_text = self.read("02_eval_and_aggregate_vista.sh")
        submit_text = self.read("submit_vista_pipeline.sh")
        self.assertIn("#SBATCH --partition=gh", eval_text)
        self.assertIn("#SBATCH --nodes=16", eval_text)
        self.assertIn("#SBATCH --time=24:00:00", eval_text)
        self.assertIn("LDS_SIMPLE_LOSS_NUM_MC=10", eval_text)
        self.assertIn("aggregate_lds_by_seed.py", eval_text)
        self.assertIn("afterok:${train_job}:${attr_job}", submit_text)

    def test_vista_scripts_do_not_hardcode_other_tacc_roots(self):
        forbidden = ("/work2/", "/scratch/11227/", "/home1/", "/home2/")
        for script in VISTA.glob("*.sh"):
            text = script.read_text()
            with self.subTest(script=script.name):
                for needle in forbidden:
                    self.assertNotIn(needle, text)


if __name__ == "__main__":
    unittest.main()
