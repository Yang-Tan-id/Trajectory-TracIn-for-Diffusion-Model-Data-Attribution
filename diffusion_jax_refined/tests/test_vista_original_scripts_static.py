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
            "02_attribution_chunk_vista.sh",
            "02a_attribution_priority_vista.sh",
            "02b_attribution_priority_vista.sh",
            "02c_attribution_priority_vista.sh",
            "02d_attribution_priority_vista.sh",
            "02e_attribution_priority_vista.sh",
            "02f_attribution_priority_vista.sh",
            "02_sample_and_original_attribution_vista.sh",
            "03_lds_eval_report_vista.sh",
            "submit_vista_original_pipeline.sh",
            "README.md",
        )
        for name in expected:
            with self.subTest(name=name):
                self.assertTrue((VISTA_ORIGINAL / name).is_file())

    def test_original_lds_training_uses_two_real_checkpoint_families(self):
        text = (VISTA_ORIGINAL / "01_train_lds_models_vista.sh").read_text()
        self.assertIn("#SBATCH --nodes=16", text)
        self.assertIn("#SBATCH --time=24:00:00", text)
        self.assertIn("MODEL_MODES=(prompted_solo unprompted_solo)", text)
        self.assertIn('LDS_SEEDS_TEXT="${LDS_SEEDS:-$(seq -s \' \' 0 7)}"', text)
        self.assertIn('LDS_M="${LDS_M:-64}"', text)
        self.assertIn('LDS_DATASET_PERCENTAGE="${LDS_DATASET_PERCENTAGE:-50}"', text)
        self.assertNotIn("prompted_multi unprompted_solo unprompted_multi", text)

    def test_original_attribution_chunks_use_monolithic_runner_and_priority_order(self):
        text = (VISTA_ORIGINAL / "02_attribution_chunk_vista.sh").read_text()
        self.assertIn("#SBATCH --nodes=64", text)
        self.assertIn("#SBATCH --time=24:00:00", text)
        self.assertIn('ATTR_NUM_JOBS="${ATTR_NUM_JOBS:-6}"', text)
        self.assertIn('ATTR_CHUNK_SIZE="${ATTR_CHUNK_SIZE:-64}"', text)
        self.assertIn('PROMPTED_SEEDS_TEXT="${PROMPTED_INITIAL_SEEDS:-$(seq -s \' \' 0 7)}"', text)
        self.assertIn('UNPROMPTED_SEEDS_TEXT="${UNPROMPTED_INITIAL_SEEDS:-$(seq -s \' \' 0 23)}"', text)
        self.assertIn("TRAJ_RANGES=(1-2500 2501-5000 5001-7500 7501-10000)", text)
        self.assertIn("bash scripts/00_sample_for_attribution.sh", text)
        self.assertIn("run_original_attribution_config.py", text)
        self.assertIn("for algorithm in dtrak end_tracin", text)
        self.assertIn('ATTRIBUTION_SCORE_MODEL_MODE="${score_mode}"', text)
        self.assertIn('ATTRIBUTION_RANGES="${range}"', text)
        self.assertNotIn("01_train_datapoint_gradient.py", text)
        self.assertNotIn("02_query_gradient.py", text)
        self.assertNotIn("03_score.py", text)
        final_text = (VISTA_ORIGINAL / "02f_attribution_priority_vista.sh").read_text()
        self.assertIn("#SBATCH --nodes=16", final_text)

    def test_original_jobs_find_real_script_dir_under_slurm_spool(self):
        job_scripts = (
            "00_train_base_models_vista.sh",
            "01_train_lds_models_vista.sh",
            "02_attribution_chunk_vista.sh",
            "03_lds_eval_report_vista.sh",
        )
        for name in job_scripts:
            with self.subTest(name=name):
                text = (VISTA_ORIGINAL / name).read_text()
                self.assertIn('SCRIPT_DIR="${VISTA_ORIGINAL_DIR:-}"', text)
                self.assertIn('${SLURM_SUBMIT_DIR:-}/diffusion_jax_refined/cifar2/vista_original', text)
                self.assertIn('! -f "${SCRIPT_DIR}/_vista_original_lib.sh"', text)
                self.assertIn('source "${SCRIPT_DIR}/_vista_original_lib.sh"', text)

        for name in (
            "02a_attribution_priority_vista.sh",
            "02b_attribution_priority_vista.sh",
            "02c_attribution_priority_vista.sh",
            "02d_attribution_priority_vista.sh",
            "02e_attribution_priority_vista.sh",
            "02f_attribution_priority_vista.sh",
        ):
            with self.subTest(name=name):
                text = (VISTA_ORIGINAL / name).read_text()
                self.assertIn('SCRIPT_DIR="${VISTA_ORIGINAL_DIR:-}"', text)
                self.assertIn('${SLURM_SUBMIT_DIR:-}/diffusion_jax_refined/cifar2/vista_original', text)
                self.assertIn('02_attribution_chunk_vista.sh', text)

    def test_original_vista_lib_does_not_require_scratch_env(self):
        text = (VISTA_ORIGINAL / "_vista_original_lib.sh").read_text()
        self.assertIn('${SCRATCH:-}', text)
        self.assertNotIn('${SCRATCH}/conda-envs/trajectory-tracin}', text)

    def test_original_run_slot_prefers_srun_over_ibrun_mpirun(self):
        text = (VISTA_ORIGINAL / "_vista_original_lib.sh").read_text()
        self.assertIn('local backend="${VISTA_SLOT_BACKEND:-srun}"', text)
        self.assertIn("command -v srun", text)
        self.assertIn("--exclusive", text)
        self.assertIn("--mpi=none", text)
        self.assertIn('elif [[ "${backend}" == "ibrun" ]]', text)

    def test_original_submit_dependencies_are_compact(self):
        text = (VISTA_ORIGINAL / "submit_vista_original_pipeline.sh").read_text()
        self.assertIn('export VISTA_ORIGINAL_DIR="${SCRIPT_DIR}"', text)
        self.assertIn("00_train_base_models_vista.sh", text)
        self.assertIn("01_train_lds_models_vista.sh", text)
        for suffix in "abcdef":
            self.assertIn(f"02{suffix}_attribution_priority_vista.sh", text)
        self.assertIn("03_lds_eval_report_vista.sh", text)
        self.assertIn("afterok:${train_job}", text)
        self.assertIn("afterok:${lds_job}:${attr_jobs}", text)
        self.assertNotIn("02_train_datapoint_gradients_vista.sh", text)
        self.assertNotIn("04_score_vista.sh", text)

    def test_original_eval_uses_two_targets_and_48_query_nodes(self):
        text = (VISTA_ORIGINAL / "03_lds_eval_report_vista.sh").read_text()
        self.assertIn("#SBATCH --nodes=48", text)
        self.assertIn("#SBATCH --time=24:00:00", text)
        self.assertIn("TARGETS=(simple_loss noise_trajectory)", text)
        self.assertIn('LDS_SEEDS_TEXT="${LDS_SEEDS:-$(seq -s \' \' 0 7)}"', text)
        self.assertIn('LDS_M="${LDS_M:-64}"', text)
        self.assertIn('UNPROMPTED_SEEDS_TEXT="${UNPROMPTED_INITIAL_SEEDS:-$(seq -s \' \' 0 23)}"', text)
        self.assertIn("for query in horse automobile horse,automobile", text)
        self.assertIn("summarize_lds_eval_report.py", text)

    def test_original_eval_resolves_percentage_lds_model_dirs_by_actual_k(self):
        text = (VISTA_ORIGINAL / "03_lds_eval_report_vista.sh").read_text()
        self.assertIn('model_pattern="${model_root}/m_${LDS_M}_k_*_pct_${LDS_DATASET_PERCENTAGE}_subset_seed_${lds_seed}"', text)
        self.assertIn('mapfile -t model_matches < <(compgen -G "${model_pattern}" | sort)', text)
        self.assertIn('Expected exactly one LDS model dir for pattern', text)
        self.assertIn('--model-glob "m_${LDS_M}_k_*_pct_${LDS_DATASET_PERCENTAGE}_subset_seed_*"', text)
        self.assertNotIn('m_${LDS_M}_k_${LDS_K}_pct_${LDS_DATASET_PERCENTAGE}_subset_seed_${lds_seed}', text)

    def test_original_runner_cli_calls_run_algorithm_config(self):
        text = (ROOT / "common" / "run_original_attribution_config.py").read_text()
        self.assertIn("run_algorithm_config(args.config_path)", text)
        self.assertIn("common.algorithm_runner", text)


if __name__ == "__main__":
    unittest.main()
