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
            "02_das_attribution_task_stampede3.sh",
            "02_smoke_das_attribution_stampede3.sh",
            "02_smoke_das_attribution_rtx_small.sh",
            "02_full_das_attribution_rtx_small.sh",
            "02_das_attribution_array_stampede3.sh",
            "02_dtrak_endtracin_attribution_chunk_stampede3.sh",
            "02_dtrak_endtracin_attribution_stampede3.sh",
            "02_dtrak_endtracin_attribution_all_stampede3.sh",
            "02_dtrak_endtracin_attribution_task_stampede3.sh",
            "02_traj_tracin_one_query_rtx_small.sh",
            "02a_das_attribution_stampede3.sh",
            "02b_das_attribution_stampede3.sh",
            "02c_das_attribution_stampede3.sh",
            "03_das_lds_eval_report_stampede3.sh",
            "03_das_lds_eval_slot_stampede3.sh",
            "03_dtrak_endtracin_lds_eval_report_stampede3.sh",
            "03_dtrak_endtracin_lds_eval_report_rtx_small.sh",
            "03_dtrak_endtracin_lds_eval_slot_stampede3.sh",
            "03_retry_slot_aggregate_stampede3.sh",
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
                if path.name == "02_smoke_das_attribution_rtx_small.sh":
                    self.assertIn("#SBATCH -p rtx-small", text)
                    self.assertNotIn("--gres", text)
                    self.assertNotIn("--gpus-per-task", text)
                    continue
                if path.name == "02_full_das_attribution_rtx_small.sh":
                    self.assertIn("#SBATCH -p rtx-small", text)
                    self.assertNotIn("--gres", text)
                    self.assertNotIn("--gpus-per-task", text)
                    continue
                if path.name == "02_traj_tracin_one_query_rtx_small.sh":
                    self.assertIn("#SBATCH -p rtx-small", text)
                    self.assertNotIn("--gres", text)
                    self.assertNotIn("--gpus-per-task", text)
                    continue
                if path.name == "03_dtrak_endtracin_lds_eval_report_rtx_small.sh":
                    self.assertIn("#SBATCH -p rtx-small", text)
                    self.assertNotIn("--gres", text)
                    self.assertNotIn("--gpus-per-task", text)
                    continue
                if path.name.startswith("_") or path.name.startswith("submit") or "_task_" in path.name or "_slot_" in path.name:
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
        self.assertIn("export DAS_DAMPING_SWEEP DAS_DAMPING_SWEEP_VALUES", text)
        self.assertNotIn('DAS_DAMPING_SWEEP_VALUES="${DAS_DAMPING_SWEEP_VALUES}" \\\n      bash -c', text)
        self.assertIn("expected_score_file()", text)
        self.assertIn("score_file_matches()", text)
        self.assertIn("Missing DAS score artifact", text)
        self.assertIn('${lambda_dir}*/scores.npy', text)
        self.assertIn("scores.npy", text)
        self.assertIn('bash "${SCRIPT_DIR}/02_das_attribution_task_stampede3.sh"', text)
        self.assertNotIn("bash -c", text)
        task_text = (STAMPEDE3_DAS / "02_das_attribution_task_stampede3.sh").read_text()
        self.assertIn("bash scripts/00_sample_for_attribution.sh", task_text)
        self.assertIn("[das-sweep] lambdas=", task_text)
        self.assertNotIn("for lambda in", task_text)
        self.assertNotIn("DAS_DAMPING_OUTPUT_TAG", task_text)
        self.assertNotIn("dtrak", text)
        self.assertNotIn("traj_tracin", text)
        array_text = (STAMPEDE3_DAS / "02_das_attribution_array_stampede3.sh").read_text()
        self.assertIn("#SBATCH --array=0-2%1", array_text)

    def test_stampede3_dtrak_endtracin_attribution_uses_mc100_defaults(self):
        text = (STAMPEDE3_DAS / "02_dtrak_endtracin_attribution_chunk_stampede3.sh").read_text()
        self.assertIn("#SBATCH -N 4", text)
        self.assertIn("#SBATCH -n 16", text)
        self.assertIn("#SBATCH -t 48:00:00", text)
        self.assertIn('ATTR_ALGORITHMS_TEXT="${ATTR_ALGORITHMS:-dtrak end_tracin}"', text)
        self.assertIn('DTRAK_TRAIN_EXPECTATION_SAMPLES="${DTRAK_TRAIN_EXPECTATION_SAMPLES:-100}"', text)
        self.assertIn('DTRAK_QUERY_EXPECTATION_SAMPLES="${DTRAK_QUERY_EXPECTATION_SAMPLES:-100}"', text)
        self.assertIn('DTRAK_BATCH_SIZE="${DTRAK_BATCH_SIZE:-8}"', text)
        self.assertIn('END_TRACIN_ENDPOINT_MC_SAMPLES="${END_TRACIN_ENDPOINT_MC_SAMPLES:-100}"', text)
        self.assertIn('END_TRACIN_TRAIN_MC_SAMPLES="${END_TRACIN_TRAIN_MC_SAMPLES:-100}"', text)
        self.assertIn('END_TRACIN_SCORE_BATCH_SIZE="${END_TRACIN_SCORE_BATCH_SIZE:-8}"', text)
        self.assertIn("expected_score_file()", text)
        self.assertIn("02_dtrak_endtracin_attribution_task_stampede3.sh", text)
        task_text = (STAMPEDE3_DAS / "02_dtrak_endtracin_attribution_task_stampede3.sh").read_text()
        self.assertIn("bash scripts/00_sample_for_attribution.sh", task_text)
        self.assertIn("run_original_attribution_config.py", task_text)
        array_text = (STAMPEDE3_DAS / "02_dtrak_endtracin_attribution_stampede3.sh").read_text()
        self.assertIn("#SBATCH --array=0-5%1", array_text)
        all_text = (STAMPEDE3_DAS / "02_dtrak_endtracin_attribution_all_stampede3.sh").read_text()
        self.assertIn("#SBATCH -N 4", all_text)
        self.assertIn("#SBATCH -n 16", all_text)
        self.assertIn("#SBATCH -t 48:00:00", all_text)
        self.assertIn('ATTR_NUM_SLOTS="${ATTR_NUM_SLOTS:-16}"', all_text)
        self.assertIn('ATTR_ALGORITHMS_TEXT="${ATTR_ALGORITHMS:-dtrak end_tracin}"', all_text)
        self.assertIn('DTRAK_TRAIN_EXPECTATION_SAMPLES="${DTRAK_TRAIN_EXPECTATION_SAMPLES:-100}"', all_text)
        self.assertIn('DTRAK_QUERY_EXPECTATION_SAMPLES="${DTRAK_QUERY_EXPECTATION_SAMPLES:-100}"', all_text)
        self.assertIn('DTRAK_BATCH_SIZE="${DTRAK_BATCH_SIZE:-8}"', all_text)
        self.assertIn('END_TRACIN_ENDPOINT_MC_SAMPLES="${END_TRACIN_ENDPOINT_MC_SAMPLES:-100}"', all_text)
        self.assertIn('END_TRACIN_TRAIN_MC_SAMPLES="${END_TRACIN_TRAIN_MC_SAMPLES:-100}"', all_text)
        self.assertIn('END_TRACIN_SCORE_BATCH_SIZE="${END_TRACIN_SCORE_BATCH_SIZE:-8}"', all_text)
        self.assertIn("for ((i = slot; i < total_tasks; i += ATTR_NUM_SLOTS))", all_text)
        self.assertIn("02_dtrak_endtracin_attribution_task_stampede3.sh", all_text)

    def test_stampede3_smoke_attribution_uses_one_node_one_lambda_and_checks_score(self):
        text = (STAMPEDE3_DAS / "02_smoke_das_attribution_stampede3.sh").read_text()
        self.assertIn("#SBATCH -N 1", text)
        self.assertIn("#SBATCH -n 1", text)
        self.assertIn("#SBATCH -t 00:30:00", text)
        self.assertIn('DAS_DAMPING_SWEEP_VALUES="${DAS_DAMPING_SWEEP_VALUES:-0.01}"', text)
        self.assertIn('SMOKE_SCORE_INDEX_RANGES="${SMOKE_SCORE_INDEX_RANGES:-1-64}"', text)
        self.assertIn('SMOKE_DAS_PROJ_DIM="${SMOKE_DAS_PROJ_DIM:-512}"', text)
        self.assertIn('SMOKE_DAS_TIMESTEPS="${SMOKE_DAS_TIMESTEPS:-0}"', text)
        self.assertIn('SMOKE_DAS_NUM_MC_NOISE="${SMOKE_DAS_NUM_MC_NOISE:-1}"', text)
        self.assertIn('SCORE_INDEX_RANGES="${SMOKE_SCORE_INDEX_RANGES}"', text)
        self.assertIn('DAS_PROJ_DIM="${SMOKE_DAS_PROJ_DIM}"', text)
        self.assertIn('DAS_TIMESTEPS="${SMOKE_DAS_TIMESTEPS}"', text)
        self.assertIn('DAS_NUM_MC_NOISE="${SMOKE_DAS_NUM_MC_NOISE}"', text)
        self.assertIn("02_das_attribution_task_stampede3.sh", text)
        self.assertIn("Missing smoke DAS score artifact", text)
        self.assertIn('${score_lambda_dir}*/scores.npy', text)
        self.assertIn("Smoke DAS attribution complete.", text)
        rtx_text = (STAMPEDE3_DAS / "02_smoke_das_attribution_rtx_small.sh").read_text()
        self.assertIn("#SBATCH -p rtx-small", rtx_text)
        self.assertIn('exec bash "${SCRIPT_DIR}/02_smoke_das_attribution_stampede3.sh"', rtx_text)

    def test_stampede3_rtx_small_full_das_uses_one_node_one_gpu_formal_settings(self):
        text = (STAMPEDE3_DAS / "02_full_das_attribution_rtx_small.sh").read_text()
        self.assertIn("#SBATCH -p rtx-small", text)
        self.assertIn("#SBATCH -N 1", text)
        self.assertIn("#SBATCH -n 1", text)
        self.assertIn("#SBATCH -t 24:00:00", text)
        self.assertIn('STAMPEDE3_SLOT_BACKEND="${STAMPEDE3_SLOT_BACKEND:-local}"', text)
        self.assertIn('ATTR_NUM_JOBS="${ATTR_NUM_JOBS:-48}"', text)
        self.assertIn('ATTR_CHUNK_SIZE="${ATTR_CHUNK_SIZE:-1}"', text)
        self.assertIn("0.01 0.02 0.05 0.1 0.2 0.5 1 2 5 10 20 50 100 200 500 1000 2000 5000 10000 20000 50000", text)
        self.assertIn("unset SCORE_INDEX_RANGES ATTRIBUTION_RANGES DAS_PROJ_DIM DAS_TIMESTEPS DAS_NUM_MC_NOISE", text)
        self.assertIn('exec bash "${SCRIPT_DIR}/02_das_attribution_chunk_stampede3.sh"', text)

    def test_stampede3_traj_tracin_one_query_rtx_splits_four_ranges_and_skips_existing(self):
        text = (STAMPEDE3_DAS / "02_traj_tracin_one_query_rtx_small.sh").read_text()
        self.assertIn("#SBATCH -p rtx-small", text)
        self.assertIn("#SBATCH -N 1", text)
        self.assertIn("#SBATCH -n 1", text)
        self.assertIn("#SBATCH --array=0-3%4", text)
        self.assertIn("#SBATCH -t 24:00:00", text)
        self.assertIn('TRAJ_RANGES_TEXT="${TRAJ_RANGES:-1-2500 2501-5000 5001-7500 7501-10000}"', text)
        self.assertIn('TRAJ_SCORE_BATCH_SIZE="${TRAJ_SCORE_BATCH_SIZE:-8}"', text)
        self.assertIn('TRAJ_SNAPSHOT_CHUNK_SIZE="${TRAJ_SNAPSHOT_CHUNK_SIZE:-4}"', text)
        self.assertIn('score_file_for_range()', text)
        self.assertIn('[traj-skip] range=${range} existing=${score_file}', text)
        self.assertIn('SCORE_INDEX_RANGES="${range}"', text)
        self.assertIn('ATTRIBUTION_RANGES="${range}"', text)
        self.assertIn('range_index="${TRAJ_RANGE_INDEX:-${SLURM_ARRAY_TASK_ID}}"', text)
        self.assertIn('check_ranges=("${ranges[$range_index]}")', text)
        self.assertIn('ALGORITHM=traj_tracin', text)
        self.assertIn('02_dtrak_endtracin_attribution_task_stampede3.sh', text)
        self.assertIn('traj_tracin_%s/scores.npy', text)
        self.assertIn('traj_tracin_unprompted_%s/scores.npy', text)

    def test_stampede3_eval_runs_sixteen_slots_three_queries_each(self):
        text = (STAMPEDE3_DAS / "03_das_lds_eval_report_stampede3.sh").read_text()
        self.assertIn("#SBATCH -N 4", text)
        self.assertIn("#SBATCH -n 16", text)
        self.assertIn("#SBATCH -t 24:00:00", text)
        self.assertIn("TARGETS=(${LDS_TARGETS:-simple_loss noise_trajectory})", text)
        self.assertIn('SLOT_LIST="$(seq -s \' \' 0 15)"', text)
        self.assertIn("for slot in ${SLOT_LIST}", text)
        self.assertIn('EVAL_SLOT_ONLY', text)
        self.assertIn("queries_per_slot=3", text)
        self.assertIn('LDS_EVAL_DEVICE_MODE:-gpu_then_cpu', text)
        self.assertIn('EVAL_SLOT_SHARD_COUNT:-1', text)
        self.assertIn('EVAL_SLOT_SHARD_INDEX="${shard_index}"', text)
        self.assertIn('EVAL_SLOT_SHARD_COUNT="${shard_count}"', text)
        self.assertIn('slot_${slot}_shard_${shard_index}.log', text)
        self.assertIn("run_eval_slot_once()", text)
        self.assertIn("run_eval_slot_with_fallback()", text)
        self.assertIn('CUDA_VISIBLE_DEVICES="${gpu}"', text)
        self.assertIn('LDS_DEVICE="${LDS_GPU_DEVICE:-gpu}"', text)
        self.assertIn('JAX_PLATFORMS=cpu', text)
        self.assertIn('LDS_DEVICE=cpu', text)
        self.assertIn("GPU attempt failed; retrying missing work on CPU", text)
        self.assertIn('PRED_TAG="${PRED_TAG}"', text)
        self.assertIn("das_lambda_$(damping_tag", text)
        self.assertIn("export PROMPTED_SEEDS_TEXT UNPROMPTED_SEEDS_TEXT LDS_SEEDS_TEXT TARGETS_TEXT DAS_DAMPING_SWEEP_VALUES", text)
        self.assertNotIn('TARGETS_TEXT="${TARGETS[*]}" \\\n      DAS_DAMPING_SWEEP_VALUES="${DAS_DAMPING_SWEEP_VALUES}"', text)
        self.assertIn('bash "${SCRIPT_DIR}/03_das_lds_eval_slot_stampede3.sh"', text)
        self.assertNotIn("bash -c", text)
        slot_text = (STAMPEDE3_DAS / "03_das_lds_eval_slot_stampede3.sh").read_text()
        self.assertIn("query_specs_inner()", slot_text)
        self.assertIn("score_dir_for_das_lambda()", slot_text)
        self.assertIn("idx += 16", slot_text)
        self.assertIn("das_unprompted*/lambda_%s", slot_text)
        self.assertIn("das*/lambda_%s", slot_text)
        self.assertIn("Expected exactly one DAS score dir for pattern", slot_text)
        self.assertIn("eval_summary_for_call()", slot_text)
        self.assertIn("[lds-eval-skip]", slot_text)
        self.assertIn("lds_summary.json", slot_text)
        self.assertIn('--algorithms "${EVAL_ALGORITHMS[@]}"', text)

    def test_stampede3_dtrak_endtracin_eval_uses_simple_loss_grid(self):
        text = (STAMPEDE3_DAS / "03_dtrak_endtracin_lds_eval_report_stampede3.sh").read_text()
        self.assertIn("#SBATCH -N 4", text)
        self.assertIn("#SBATCH -n 16", text)
        self.assertIn("#SBATCH -t 24:00:00", text)
        self.assertIn("TARGETS=(${LDS_TARGETS:-simple_loss noise_trajectory})", text)
        self.assertIn("EVAL_ALGORITHMS=(${EVAL_ALGORITHMS:-dtrak end_tracin})", text)
        self.assertIn("serial_slots=${EVAL_SERIAL_SLOTS:-0}", text)
        self.assertIn("local_parallel_slots=${EVAL_LOCAL_PARALLEL_SLOTS:-0}", text)
        self.assertIn('if [[ "${EVAL_SERIAL_SLOTS:-0}" == "1" ]]', text)
        self.assertIn('if [[ "${EVAL_SERIAL_SLOTS:-0}" != "1" ]]', text)
        self.assertIn('local_parallel_slots="${EVAL_LOCAL_PARALLEL_SLOTS:-0}"', text)
        self.assertIn('local_parallel_slots > 0', text)
        self.assertIn('LDS_SIMPLE_LOSS_TIMESTEPS="${LDS_SIMPLE_LOSS_TIMESTEPS:-$(seq -s, 0 999)}"', text)
        self.assertIn('LDS_SIMPLE_LOSS_NOISE_SEEDS="${LDS_SIMPLE_LOSS_NOISE_SEEDS:-0}"', text)
        self.assertIn('LDS_SIMPLE_LOSS_NUM_MC="${LDS_SIMPLE_LOSS_NUM_MC:-1000}"', text)
        self.assertIn('FORCE_LDS_EVAL="${FORCE_LDS_EVAL:-0}"', text)
        self.assertIn("03_dtrak_endtracin_lds_eval_slot_stampede3.sh", text)
        self.assertIn('--algorithms "${EVAL_ALGORITHMS[@]}"', text)
        slot_text = (STAMPEDE3_DAS / "03_dtrak_endtracin_lds_eval_slot_stampede3.sh").read_text()
        self.assertIn("eval_summary_for_call()", slot_text)
        self.assertIn("[lds-eval-skip]", slot_text)
        self.assertIn("FORCE_LDS_EVAL", slot_text)
        self.assertIn("--algorithm \"${algorithm}\"", slot_text)

    def test_stampede3_dtrak_endtracin_rtx_eval_runs_four_local_slots(self):
        text = (STAMPEDE3_DAS / "03_dtrak_endtracin_lds_eval_report_rtx_small.sh").read_text()
        self.assertIn("#SBATCH -p rtx-small", text)
        self.assertIn("#SBATCH -N 1", text)
        self.assertIn("#SBATCH -n 1", text)
        self.assertIn("#SBATCH -t 12:00:00", text)
        self.assertIn('STAMPEDE3_SLOT_BACKEND="${STAMPEDE3_SLOT_BACKEND:-local}"', text)
        self.assertIn('EVAL_SERIAL_SLOTS="${EVAL_SERIAL_SLOTS:-0}"', text)
        self.assertIn('EVAL_LOCAL_PARALLEL_SLOTS="${EVAL_LOCAL_PARALLEL_SLOTS:-4}"', text)
        self.assertIn('EVAL_SLOT_SHARD_COUNT="${EVAL_SLOT_SHARD_COUNT:-1}"', text)
        self.assertIn('LDS_EVAL_DEVICE_MODE="${LDS_EVAL_DEVICE_MODE:-gpu_then_cpu}"', text)
        self.assertIn('exec bash "${SCRIPT_DIR}/03_dtrak_endtracin_lds_eval_report_stampede3.sh"', text)

    def test_stampede3_retry_slot_aggregate_uses_one_node_slot_only(self):
        text = (STAMPEDE3_DAS / "03_retry_slot_aggregate_stampede3.sh").read_text()
        self.assertIn("#SBATCH -p h100", text)
        self.assertIn("#SBATCH -N 1", text)
        self.assertIn("#SBATCH -n 4", text)
        self.assertIn('EVAL_SLOT_ONLY="${EVAL_SLOT_ONLY:-3}"', text)
        self.assertIn('EVAL_SLOT_SHARD_COUNT="${EVAL_SLOT_SHARD_COUNT:-4}"', text)
        self.assertIn('LDS_EVAL_DEVICE_MODE="${LDS_EVAL_DEVICE_MODE:-gpu_then_cpu}"', text)
        self.assertIn('exec bash "${SCRIPT_DIR}/03_das_lds_eval_report_stampede3.sh"', text)

    def test_stampede3_submit_uses_staged_submission_for_h100_limits(self):
        text = (STAMPEDE3_DAS / "submit_stampede3_das_pipeline.sh").read_text()
        self.assertIn("split into three explicit chunk jobs", text)
        self.assertIn("STAMPEDE3_ACCOUNT", text)
        self.assertIn("SBATCH_ACCOUNT_ARGS=(-A", text)
        self.assertIn('stage="${STAMPEDE3_SUBMIT_STAGE:-base_lds}"', text)
        self.assertIn("base_lds)", text)
        self.assertIn("attr)", text)
        self.assertIn("eval)", text)
        self.assertIn("00_train_base_models_stampede3.sh", text)
        self.assertIn("01_train_lds_models_stampede3.sh", text)
        self.assertIn("02a_das_attribution_stampede3.sh", text)
        self.assertIn("02b_das_attribution_stampede3.sh", text)
        self.assertIn("02c_das_attribution_stampede3.sh", text)
        self.assertIn("03_das_lds_eval_report_stampede3.sh", text)
        self.assertIn('dependency_args=(--dependency=afterok:${TRAIN_JOB_ID})', text)
        self.assertIn("afterok:${lds_job}:${attr_jobs}", text)
        self.assertIn("ATTR_JOB_IDS", text)
        self.assertNotIn("02 DAS attribution array 0-2", text)
        self.assertIn("LDS_JOB_ID", text)
        self.assertIn("ATTR_JOB_ID", text)


if __name__ == "__main__":
    unittest.main()
