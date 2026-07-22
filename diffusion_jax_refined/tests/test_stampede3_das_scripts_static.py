from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAMPEDE3_DAS = ROOT / "cifar2" / "stampede3_das"


class TestStampede3DasScriptsStatic(unittest.TestCase):
    def test_stampede3_das_folder_has_expected_jobs(self):
        expected = (
            "_stampede3_das_lib.sh",
            "00_train_base_models_stampede3.sh",
            "01_train_lds_models_stampede3.sh",
            "02_das_attribution_chunk_stampede3.sh",
            "02_das_attribution_array_stampede3.sh",
            "02a_das_attribution_stampede3.sh",
            "02b_das_attribution_stampede3.sh",
            "02c_das_attribution_stampede3.sh",
            "03_das_lds_eval_report_stampede3.sh",
            "submit_stampede3_das_pipeline.sh",
            "README.md",
        )
        for name in expected:
            with self.subTest(name=name):
                self.assertTrue((STAMPEDE3_DAS / name).is_file())

    def test_stampede3_jobs_use_h100_without_unsupported_gpu_directives(self):
        for path in STAMPEDE3_DAS.glob("*.sh"):
            with self.subTest(path=path.name):
                text = path.read_text()
                if path.name.startswith("_") or path.name.startswith("submit"):
                    continue
                self.assertIn("#SBATCH -p h100", text)
                self.assertNotIn("CCR25021", text)
                self.assertNotIn("#SBATCH -A", text)
                self.assertNotIn("--gres", text)
                self.assertNotIn("--gpus-per-task", text)

    def test_stampede3_train_uses_one_node_two_two_gpu_trainers(self):
        text = (STAMPEDE3_DAS / "00_train_base_models_stampede3.sh").read_text()
        self.assertIn("#SBATCH -N 1", text)
        self.assertIn("#SBATCH -n 4", text)
        self.assertIn("#SBATCH -t 02:00:00", text)
        self.assertIn("CUDA_VISIBLE_DEVICES=0,1", text)
        self.assertIn("CUDA_VISIBLE_DEVICES=2,3", text)
        self.assertIn("JAX_NUM_DEVICES=2", text)
        self.assertIn("prompted_solo", text)
        self.assertIn("unprompted_solo", text)

    def test_stampede3_lds_matches_requested_modes_and_scale(self):
        text = (STAMPEDE3_DAS / "01_train_lds_models_stampede3.sh").read_text()
        self.assertIn("#SBATCH -N 4", text)
        self.assertIn("#SBATCH -n 16", text)
        self.assertIn("#SBATCH -t 24:00:00", text)
        self.assertIn("MODEL_MODES=(prompted_solo unprompted_solo)", text)
        self.assertIn('LDS_SEEDS_TEXT="${LDS_SEEDS:-$(seq -s \' \' 0 7)}"', text)
        self.assertIn('LDS_M="${LDS_M:-64}"', text)
        self.assertIn('LDS_DATASET_PERCENTAGE="${LDS_DATASET_PERCENTAGE:-50}"', text)

    def test_stampede3_das_attribution_is_three_h100_chunks_with_full_sweep(self):
        text = (STAMPEDE3_DAS / "02_das_attribution_chunk_stampede3.sh").read_text()
        self.assertIn("#SBATCH -N 4", text)
        self.assertIn("#SBATCH -n 16", text)
        self.assertIn("#SBATCH -t 48:00:00", text)
        self.assertIn('ATTR_JOB_INDEX="${ATTR_JOB_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"', text)
        self.assertIn('ATTR_NUM_JOBS="${ATTR_NUM_JOBS:-3}"', text)
        self.assertIn('ATTR_CHUNK_SIZE="${ATTR_CHUNK_SIZE:-16}"', text)
        self.assertIn('UNPROMPTED_SEEDS_TEXT="${UNPROMPTED_INITIAL_SEEDS:-$(seq -s \' \' 0 23)}"', text)
        self.assertIn('PROMPTED_SEEDS_TEXT="${PROMPTED_INITIAL_SEEDS:-$(seq -s \' \' 0 7)}"', text)
        self.assertIn("0.01 0.02 0.05 0.1 0.2 0.5 1 2 5 10 20 50 100 200 500 1000 2000 5000 10000 20000 50000", text)
        self.assertIn("bash scripts/00_sample_for_attribution.sh", text)
        self.assertIn("DAS_DAMPING_OUTPUT_TAG", text)
        self.assertNotIn("dtrak", text)
        self.assertNotIn("traj_tracin", text)
        array_text = (STAMPEDE3_DAS / "02_das_attribution_array_stampede3.sh").read_text()
        self.assertIn("#SBATCH --array=0-2%1", array_text)

    def test_stampede3_eval_runs_sixteen_slots_three_queries_each(self):
        text = (STAMPEDE3_DAS / "03_das_lds_eval_report_stampede3.sh").read_text()
        self.assertIn("#SBATCH -N 4", text)
        self.assertIn("#SBATCH -n 16", text)
        self.assertIn("#SBATCH -t 24:00:00", text)
        self.assertIn("TARGETS=(simple_loss noise_trajectory)", text)
        self.assertIn("for slot in $(seq 0 15)", text)
        self.assertIn("idx += 16", text)
        self.assertIn("queries_per_gpu=3", text)
        self.assertIn("das_lambda_$(damping_tag", text)
        self.assertIn("das_unprompted/lambda_%s", text)
        self.assertIn("das/lambda_%s", text)
        self.assertIn('--algorithms "${EVAL_ALGORITHMS[@]}"', text)

    def test_stampede3_submit_uses_staged_submission_for_h100_limits(self):
        text = (STAMPEDE3_DAS / "submit_stampede3_das_pipeline.sh").read_text()
        self.assertIn("array elements toward the submit limit", text)
        self.assertIn("STAMPEDE3_ACCOUNT", text)
        self.assertIn("SBATCH_ACCOUNT_ARGS=(-A", text)
        self.assertIn('stage="${STAMPEDE3_SUBMIT_STAGE:-base_lds}"', text)
        self.assertIn("base_lds)", text)
        self.assertIn("attr)", text)
        self.assertIn("eval)", text)
        self.assertIn("00_train_base_models_stampede3.sh", text)
        self.assertIn("01_train_lds_models_stampede3.sh", text)
        self.assertIn("02_das_attribution_array_stampede3.sh", text)
        self.assertIn("03_das_lds_eval_report_stampede3.sh", text)
        self.assertIn("afterok:${train_job}", text)
        self.assertIn("afterok:${lds_job}:${attr_job}", text)
        self.assertIn("TRAIN_JOB_ID", text)
        self.assertIn("LDS_JOB_ID", text)
        self.assertIn("ATTR_JOB_ID", text)


if __name__ == "__main__":
    unittest.main()
