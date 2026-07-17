from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VISTA_ORIGINAL = ROOT / "cifar2" / "vista_original"


class TestVistaOriginalScriptsStatic(unittest.TestCase):
    def test_vista_original_folder_has_expected_jobs(self):
        expected = (
            "_vista_original_lib.sh",
            "00_train_base_models_vista.sh",
            "01_train_lds_models_vista.sh",
            "02_sample_and_original_attribution_vista.sh",
            "03_lds_eval_report_vista.sh",
            "submit_vista_original_pipeline.sh",
            "README.md",
        )
        for name in expected:
            with self.subTest(name=name):
                self.assertTrue((VISTA_ORIGINAL / name).is_file())

    def test_original_sample_attribution_job_uses_monolithic_runner(self):
        text = (VISTA_ORIGINAL / "02_sample_and_original_attribution_vista.sh").read_text()
        self.assertIn("#SBATCH --nodes=21", text)
        self.assertIn("#SBATCH --time=48:00:00", text)
        self.assertIn("bash scripts/00_sample_for_attribution.sh", text)
        self.assertIn("run_original_attribution_config.py", text)
        self.assertIn("dtrak das end_tracin traj_tracin", text)
        self.assertIn('UNPROMPTED_SAMPLE_MODEL_MODE="${mode}"', text)
        self.assertIn('UNPROMPTED_SCORE_MODEL_MODE="${mode}"', text)
        self.assertNotIn("01_train_datapoint_gradient.py", text)
        self.assertNotIn("02_query_gradient.py", text)
        self.assertNotIn("03_score.py", text)

    def test_original_submit_dependencies_are_compact(self):
        text = (VISTA_ORIGINAL / "submit_vista_original_pipeline.sh").read_text()
        self.assertIn("00_train_base_models_vista.sh", text)
        self.assertIn("01_train_lds_models_vista.sh", text)
        self.assertIn("02_sample_and_original_attribution_vista.sh", text)
        self.assertIn("03_lds_eval_report_vista.sh", text)
        self.assertIn("afterok:${train_job}", text)
        self.assertIn("afterok:${lds_job}:${sample_attr_job}", text)
        self.assertNotIn("02_train_datapoint_gradients_vista.sh", text)
        self.assertNotIn("04_score_vista.sh", text)

    def test_original_runner_cli_calls_run_algorithm_config(self):
        text = (ROOT / "common" / "run_original_attribution_config.py").read_text()
        self.assertIn("run_algorithm_config(args.config_path)", text)
        self.assertIn("common.algorithm_runner", text)


if __name__ == "__main__":
    unittest.main()
