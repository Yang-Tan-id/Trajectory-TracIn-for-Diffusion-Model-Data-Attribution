from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TACC = ROOT / "cifar2" / "tacc"
SCRIPTS = ROOT / "cifar2" / "scripts"


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

    def test_sample_attribution_can_fill_h100_allocation_with_independent_tasks(self):
        text = self.read("02_sample_attribution_h100.sh")
        self.assertRegex(text, r"#SBATCH\s+--nodes=4")
        self.assertRegex(text, r"#SBATCH\s+--ntasks-per-node=4")
        self.assertIn('MAX_PARALLEL_ATTR_TASKS="${MAX_PARALLEL_ATTR_TASKS:-${SLURM_NTASKS:-16}}"', text)
        self.assertIn('CUDA_VISIBLE_DEVICES="$((slot % 4))"', text)
        self.assertIn('ibrun -n 1 -o "${slot}"', text)
        self.assertIn('if (( ${#pids[@]} >= MAX_PARALLEL_ATTR_TASKS )); then', text)
        self.assertIn('ENDPOINT_ALGORITHMS_TEXT="${ENDPOINT_ALGORITHMS_TEXT:-das dtrak end_tracin}"', text)

    def test_attribution_only_wrappers_reuse_samples_and_split_h100_rtx(self):
        h100 = self.read("08_attribution_only_h100.sh")
        rtx = self.read("09_attribution_only_rtx_remaining.sh")
        common = self.read("attribution_only_common.sh")

        self.assertRegex(h100, r"#SBATCH\s+--partition=h100")
        self.assertRegex(h100, r"#SBATCH\s+--nodes=4")
        self.assertRegex(h100, r"#SBATCH\s+--ntasks-per-node=4")
        self.assertIn('ATTR_TASK_START="${ATTR_TASK_START:-1}"', h100)
        self.assertIn('ATTR_TASK_END="${ATTR_TASK_END:-16}"', h100)
        self.assertIn('source "${SCRIPT_DIR}/attribution_only_common.sh"', h100)

        self.assertRegex(rtx, r"#SBATCH\s+--partition=rtx-small")
        self.assertRegex(rtx, r"#SBATCH\s+--nodes=1")
        self.assertRegex(rtx, r"#SBATCH\s+--ntasks-per-node=2")
        self.assertIn('ATTR_TASK_START="${ATTR_TASK_START:-17}"', rtx)
        self.assertIn('ATTR_TASK_END="${ATTR_TASK_END:-18}"', rtx)
        self.assertIn('GPU_PER_NODE="${GPU_PER_NODE:-2}"', rtx)
        self.assertIn('source "${SCRIPT_DIR}/attribution_only_common.sh"', rtx)

        self.assertIn("validate_sample", common)
        self.assertIn("launch_attribution", common)
        self.assertIn('ATTR_ALGORITHMS="${ATTR_ALGORITHMS:-traj_tracin das}"', common)
        self.assertIn('read -r -a ATTR_ALGORITHM_LIST <<<"${ATTR_ALGORITHMS}"', common)
        self.assertNotIn("scripts/00_sample.sh", common)
        self.assertNotIn("04_lds_eval", common)
        self.assertNotIn("RUN_ENDPOINTS", common)
        self.assertNotIn("ENDPOINT_ALGORITHMS", common)

    def test_cifar2_das_uses_batched_gpu_path_by_default_without_large_result_io(self):
        text = (ROOT / "cifar2" / "dataset_config.py").read_text()
        self.assertIn('"use_batched_per_example_grads": os.environ.get("DAS_BATCHED", "1")', text)
        self.assertIn('"per_example_grad_batch_size": int(os.environ.get("DAS_GRAD_BATCH_SIZE", "8"))', text)
        self.assertIn('"DAS_SHERMAN_MORRISON_DENOMINATOR", "1"', text)
        das_text = (ROOT / "legacy_jax" / "DM_dataAttribution_algo_end_das.py").read_text()
        self.assertIn("compute_batched_das_term", das_text)
        self.assertIn("jax.vmap(phi_fn", das_text)
        self.assertIn("H_device = H_device + phi_batch.T @ phi_batch", das_text)
        self.assertIn("return np.square(raw)", das_text)

    def test_local_traj_l1_k8000_workflow_script_defaults(self):
        text = (SCRIPTS / "05_traj_l1_attribution_k8000_lds_eval.sh").read_text()
        self.assertIn('TRAJ_QUERY_OBJECTIVE="${TRAJ_QUERY_OBJECTIVE:-eps_deviation_l1_mean}"', text)
        self.assertIn('LDS_K="${LDS_K:-8000}"', text)
        self.assertIn('LDS_M="${LDS_M:-50}"', text)
        self.assertIn('RUN_SAMPLE="${RUN_SAMPLE:-1}"', text)
        self.assertIn('SAMPLE_SEEDS="${INITIAL_SEED}"', text)
        self.assertIn('bash scripts/00_sample.sh', text)
        self.assertIn('MAX_PARALLEL_SAMPLE_TASKS', text)
        self.assertIn('sample_complete()', text)
        self.assertIn('TRAJ_RANGES_TEXT="${TRAJ_RANGES_TEXT:-1-2500 2501-5000 5001-7500 7501-10000}"', text)
        self.assertIn('ALGORITHMS="traj_tracin"', text)
        self.assertIn('bash scripts/01_data_attribution.sh', text)
        self.assertIn('bash scripts/04_lds_eval.sh', text)
        self.assertIn('aggregate_lds_by_seed.py', text)
        self.assertIn('MAX_PARALLEL_ATTR_TASKS', text)
        self.assertIn('MAX_PARALLEL_EVAL_TASKS', text)
        self.assertNotIn('sbatch', text)
        self.assertNotIn('ibrun', text)
        self.assertNotIn('ALGORITHMS="das"', text)

    def test_combined_eval_overwrite_guard_is_target_specific(self):
        text = self.read("03_lds_eval_h100.sh")
        self.assertIn('/lds/${eval_algorithm}/${LDS_TARGET_FUNCTION}', text)
        self.assertIn('eval_algorithm="$(traj_algorithm_tag)"', text)
        self.assertIn('Refusing to overwrite ${out_dir}', text)
        self.assertNotIn('Refusing to overwrite ${eval_dir}', text)

    def test_per_seed_eval_runs_aggregate_after_success(self):
        text = self.read("04_lds_eval_by_seed_h100.sh")
        self.assertIn("aggregate_lds_by_seed.py", text)
        self.assertIn("--target-function", text)
        self.assertIn("--lds-k", text)
        self.assertIn("per-seed LDS evaluations completed", text)
        self.assertIn("aggregate.log", text)

    def test_per_seed_eval_supports_removed_prediction_subset(self):
        text = self.read("04_lds_eval_by_seed_h100.sh")
        self.assertIn('LDS_PREDICTION_SUBSET="${LDS_PREDICTION_SUBSET:-kept}"', text)
        self.assertIn('LDS_PREDICTION_SIGN="${LDS_PREDICTION_SIGN:--1}"', text)
        self.assertIn("prediction_tag()", text)
        self.assertIn('${PREDICTION_TAG}/$(basename "${model_dir}")', text)
        self.assertIn('--prediction-subset "${LDS_PREDICTION_SUBSET}"', text)
        self.assertIn('--prediction-sign "${LDS_PREDICTION_SIGN}"', text)
        self.assertIn('aggregate_m_${LDS_M}_k_${LDS_K}_${PREDICTION_TAG}', text)


if __name__ == "__main__":
    unittest.main()
