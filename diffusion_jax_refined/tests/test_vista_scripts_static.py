from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VISTA = ROOT / "cifar2" / "vista"


class TestVistaScriptsStatic(unittest.TestCase):
    def read(self, name: str) -> str:
        return (VISTA / name).read_text()

    def test_train_base_checkpoints_requests_gh_2_nodes_seed_67(self):
        text = self.read("00_train_four_models_vista.sh")
        self.assertIn("#SBATCH --partition=gh", text)
        self.assertIn("#SBATCH --account=CCR25021", text)
        self.assertIn("#SBATCH --nodes=2", text)
        self.assertIn("#SBATCH --time=02:00:00", text)
        self.assertIn("MODEL_MODES=(prompted_solo unprompted_solo)", text)
        self.assertNotIn("MODEL_MODES=(prompted_solo prompted_multi unprompted_solo unprompted_multi)", text)
        self.assertIn("share prompted_jax checkpoints", text)
        self.assertIn('TRAIN_SEED="${TRAIN_SEED:-67}"', (VISTA / "_vista_pipeline_lib.sh").read_text())

    def test_train_lds_requests_gh_64_nodes_50_percent_all_modes(self):
        text = self.read("01_train_lds_models_vista.sh")
        self.assertIn("#SBATCH --partition=gh", text)
        self.assertIn("#SBATCH --nodes=64", text)
        self.assertIn("#SBATCH --time=24:00:00", text)
        self.assertIn('LDS_M="${LDS_M:-50}"', text)
        self.assertIn('LDS_DATASET_PERCENTAGE="${LDS_DATASET_PERCENTAGE:-50}"', text)
        self.assertIn('LDS_SEEDS_TEXT="${LDS_SEEDS:-$(seq -s \' \' 1 16)}"', text)
        self.assertIn("MODEL_MODES=(prompted_solo prompted_multi unprompted_solo unprompted_multi)", text)

    def test_train_gradient_requests_8_nodes_2_checkpoint_families_4_algorithms(self):
        text = self.read("02_train_datapoint_gradients_vista.sh")
        self.assertIn("#SBATCH --partition=gh", text)
        self.assertIn("#SBATCH --nodes=8", text)
        self.assertIn("#SBATCH --time=48:00:00", text)
        self.assertIn("MODEL_MODES=(prompted_solo unprompted_solo)", text)
        self.assertNotIn("MODEL_MODES=(prompted_solo prompted_multi unprompted_solo unprompted_multi)", text)
        self.assertIn("ALGORITHMS=(dtrak das end_tracin traj_tracin)", text)

    def test_sample_score_eval_request_21_nodes_and_dependencies(self):
        sample_text = self.read("03_sample_query_gradients_vista.sh")
        score_text = self.read("04_score_vista.sh")
        eval_text = self.read("05_lds_eval_report_vista.sh")
        submit_text = self.read("submit_vista_pipeline.sh")
        self.assertIn("#SBATCH --nodes=21", sample_text)
        self.assertIn("#SBATCH --time=48:00:00", sample_text)
        self.assertIn("#SBATCH --nodes=21", score_text)
        self.assertIn("#SBATCH --time=24:00:00", score_text)
        self.assertIn("#SBATCH --nodes=21", eval_text)
        self.assertIn("#SBATCH --time=12:00:00", eval_text)
        self.assertIn("horse,automobile|96", sample_text)
        self.assertIn("LDS_SIMPLE_LOSS_NUM_MC=10", eval_text)
        self.assertIn("aggregate_lds_by_seed.py", eval_text)
        self.assertIn("summarize_lds_eval_report.py", eval_text)
        self.assertIn("afterok:${train_job}", submit_text)
        self.assertIn("afterok:${lds_job}:${train_grad_job}:${sample_qgrad_job}", submit_text)
        self.assertIn("afterok:${score_job}", submit_text)

    def test_vista_scripts_do_not_hardcode_other_tacc_roots(self):
        forbidden = ("/work2/", "/scratch/11227/", "/home1/", "/home2/")
        for script in VISTA.glob("*.sh"):
            text = script.read_text()
            with self.subTest(script=script.name):
                for needle in forbidden:
                    self.assertNotIn(needle, text)


if __name__ == "__main__":
    unittest.main()
