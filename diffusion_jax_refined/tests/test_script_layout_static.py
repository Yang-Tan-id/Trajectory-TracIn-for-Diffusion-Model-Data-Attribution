from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("cifar2", "cifar10", "artbench")
ALGORITHMS = ("das", "dtrak", "end_tracin", "journey_trak", "traj_tracin")


class TestScriptLayoutStatic(unittest.TestCase):
    def test_legacy_tacc_scripts_are_removed(self):
        self.assertFalse((ROOT / "tacc").exists())
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                self.assertTrue((ROOT / dataset / "tacc" / "h100" / "script_0.sh").is_file())
                self.assertTrue((ROOT / dataset / "tacc" / "vista" / "script_0.sh").is_file())
                self.assertTrue((ROOT / dataset / "tacc" / "h100" / "sample_for_attribution.sh").is_file())
                self.assertTrue((ROOT / dataset / "tacc" / "vista" / "sample_for_attribution.sh").is_file())
                self.assertTrue((ROOT / dataset / "tacc" / "h100" / "sample_query_gradient.sh").is_file())
                self.assertTrue((ROOT / dataset / "tacc" / "vista" / "sample_query_gradient.sh").is_file())
                self.assertTrue((ROOT / dataset / "tacc" / "h100" / "datapoint_gradients.sh").is_file())
                self.assertTrue((ROOT / dataset / "tacc" / "vista" / "datapoint_gradients.sh").is_file())

    def test_dataset_train_framework_has_four_modes_and_selector(self):
        expected = (
            "00_train_prompted_solo.sh",
            "00_train_prompted_multi.sh",
            "00_train_unprompted_solo.sh",
            "00_train_unprompted_multi.sh",
            "script_0.sh",
        )
        for dataset in DATASETS:
            for name in expected:
                with self.subTest(dataset=dataset, script=name):
                    self.assertTrue((ROOT / dataset / "scripts" / name).is_file())
            selector = (ROOT / dataset / "scripts" / "script_0.sh").read_text()
            self.assertIn("MODES_TEXT", selector)
            self.assertIn("TRAIN_MODES", selector)
            self.assertIn("for MODE in ${MODES_TEXT}", selector)
            self.assertIn("prompted_solo", selector)
            self.assertIn("prompted_multi", selector)
            self.assertIn("unprompted_solo", selector)
            self.assertIn("unprompted_multi", selector)

    def test_train_scripts_delegate_to_shared_train_shell_lib(self):
        train_scripts = (
            "00_train_prompted_solo.sh",
            "00_train_prompted_multi.sh",
            "00_train_unprompted_solo.sh",
            "00_train_unprompted_multi.sh",
        )
        for dataset in DATASETS:
            for name in train_scripts:
                with self.subTest(dataset=dataset, script=name):
                    text = (ROOT / dataset / "scripts" / name).read_text()
                    self.assertIn("../common/train_shell_lib.sh", text)

    def test_training_does_not_use_query_or_prompt_for_dataset_filtering(self):
        shell_text = (ROOT / "common" / "train_shell_lib.sh").read_text()
        self.assertNotIn("TRAIN_PROMPT_MODE", shell_text)
        self.assertNotIn("TRAIN_MODEL_TAG", shell_text)
        self.assertNotIn("TRAIN_PROMPTS", shell_text)
        self.assertNotIn("QUERY=", shell_text)

        training_text = (ROOT / "common" / "prompted_jax_training.py").read_text()
        self.assertIn('return getattr(cfg_module, default_attr, None)', training_text)
        self.assertIn('learning_rate=_optional_float("JAX_LEARNING_RATE", 1e-4)', training_text)
        self.assertIn('lr_schedule=os.environ.get("JAX_LR_SCHEDULE", "cosine_warmup")', training_text)
        self.assertIn('lr_warmup_ratio=_optional_float("JAX_LR_WARMUP_RATIO", 0.1)', training_text)
        self.assertIn('dm_learning_rate=_optional_float("JAX_LEARNING_RATE", 1e-4)', training_text)
        self.assertNotIn("1e-4 if unconditional else 2e-4", training_text)
        self.assertNotIn("TRAIN_PROMPT_MODE", training_text)
        self.assertNotIn("TRAIN_PROMPTS", training_text)
        self.assertNotIn("TRAIN_MODEL_TAG", training_text)
        self.assertNotIn("TRAIN_MODEL_KIND", training_text)
        self.assertNotIn("prompted_solo", training_text)
        self.assertNotIn("prompted_multi", training_text)

        for dataset in DATASETS:
            config_text = (ROOT / dataset / "dataset_config.py").read_text()
            self.assertIn('TRAIN_SEED = int(os.environ.get("TRAIN_SEED"', config_text)
            self.assertIn("PROMPTED_CKPT_STEM", config_text)
            self.assertIn('REFERENCE_CKPT = str(PROMPTED_JAX_MODEL_ROOT / f"{PROMPTED_CKPT_STEM}.ckpt")', config_text)
            self.assertNotIn('PROMPTED_JAX_MODEL_ROOT / "seed_42_epoch', config_text)

    def test_tacc_script_0_selects_dataset_mode_and_gpu_count(self):
        for dataset in DATASETS:
            for system in ("h100", "vista"):
                with self.subTest(dataset=dataset, system=system):
                    text = (ROOT / dataset / "tacc" / system / "script_0.sh").read_text()
                    self.assertIn('DATASET_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"', text)
                    self.assertIn('TRAIN_MODE="${TRAIN_MODE:-prompted_multi}"', text)
                    self.assertIn('TRAIN_MODES="${TRAIN_MODES:-${TRAIN_MODE}}"', text)
                    self.assertIn('GPU_IDS="${GPU_IDS:-0,1,2,3}"', text)
                    self.assertIn('JAX_NUM_DEVICES', text)
                    self.assertIn('${DATASET_ROOT}/scripts/script_0.sh', text)
                    self.assertNotIn("DATASET=", text)
                    self.assertNotIn("REFINE_ROOT", text)

    def test_datapoint_gradient_scripts_accept_optional_modes_and_algorithms(self):
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                script = ROOT / dataset / "scripts" / "01_datapoint_gradients.sh"
                self.assertTrue(script.is_file())
                text = script.read_text()
                self.assertIn('ALGORITHMS_TEXT="${ALGORITHMS:-${ALGO:-${ALGORITHM:-das}}}"', text)
                self.assertIn('ALGORITHMS_TEXT="${ALGORITHMS_TEXT//,/ }"', text)
                self.assertIn('TRAIN_MODES_TEXT="${TRAIN_MODES:-${TRAIN_MODE:-}}"', text)
                self.assertIn('DATAPOINT_GRADIENT_MODES', text)
                self.assertIn('model/${mode_label}/seed_${TRAIN_SEED:-42}_train_gradient', text)
                self.assertIn('DATAPOINT_MODEL_MODE="${mode_label}"', text)
                self.assertIn('bash "${ROOT}/scripts/script_0.sh" ${TRAIN_MODES_TEXT}', text)
                self.assertIn('"${PYTHON_BIN}" 01_train_datapoint_gradient.py', text)
                self.assertNotIn('bash "${ROOT}/scripts/01_data_attribution.sh"', text)
                self.assertNotIn('bash "${ROOT}/scripts/01_data_attribution_unprompted.sh"', text)

                for algorithm in ALGORITHMS:
                    algorithm_dir = ROOT / dataset / "data_attribution" / algorithm
                    for name in ("01_train_datapoint_gradient.py", "02_query_gradient.py", "03_score.py"):
                        stage_script = algorithm_dir / name
                        self.assertTrue(stage_script.is_file())
                        stage_text = stage_script.read_text()
                        if name == "03_score.py":
                            self.assertIn("run_score_combination_stage", stage_text)
                            self.assertNotIn("run_algorithm_config", stage_text)
                        elif name == "01_train_datapoint_gradient.py":
                            self.assertIn("run_train_datapoint_gradient_artifact", stage_text)
                            self.assertNotIn('run_stage_config(Path(__file__).with_name("CONFIG.py"), "train_datapoint_gradient")', stage_text)
                        else:
                            self.assertIn("run_query_gradient_artifact", stage_text)
                            self.assertNotIn('run_stage_config(Path(__file__).with_name("CONFIG.py"), "query_gradient")', stage_text)

        for dataset in DATASETS:
            for system in ("h100", "vista"):
                with self.subTest(dataset=dataset, system=system):
                    script = ROOT / dataset / "tacc" / system / "datapoint_gradients.sh"
                    self.assertTrue(script.is_file())
                    text = script.read_text()
                    self.assertIn('DATASET_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"', text)
                    self.assertIn('ALGORITHMS="${ALGORITHMS:-${ALGO:-${ALGORITHM:-das}}}"', text)
                    self.assertIn('GPU_IDS="${GPU_IDS:-0,1,2,3}"', text)
                    self.assertIn('${DATASET_ROOT}/scripts/01_datapoint_gradients.sh', text)
                    self.assertNotIn("DATASET=", text)
                    self.assertNotIn("REFINE_ROOT", text)

    def test_sample_for_attribution_scripts_select_model_mode_and_sample_root(self):
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                script = ROOT / dataset / "scripts" / "00_sample_for_attribution.sh"
                self.assertTrue(script.is_file())
                text = script.read_text()
                self.assertIn('SAMPLE_MODEL_MODE="${SAMPLE_MODEL_MODE:-prompted_solo}"', text)
                self.assertIn('SAMPLE_ROOT="${SAMPLE_ROOT:-${ROOT}/result/${EXPERIMENT_TAG:-experiment1}/sample}"', text)
                self.assertIn("UNPROMPTED=1", text)
                self.assertIn('"${PYTHON_BIN}" "${ROOT}/sampling/run_sampling.py"', text)

                config_text = (ROOT / dataset / "sampling" / "CONFIG.py").read_text()
                self.assertIn("SAMPLE_MODEL_MODE", config_text)
                self.assertIn("EXPERIMENT_TAG", config_text)
                self.assertIn('RESULT_ROOT / "sample"', config_text)
                self.assertIn("MODEL_TAG = SAMPLE_MODEL_MODE", config_text)

        for dataset in DATASETS:
            for system in ("h100", "vista"):
                with self.subTest(dataset=dataset, system=system):
                    script = ROOT / dataset / "tacc" / system / "sample_for_attribution.sh"
                    self.assertTrue(script.is_file())
                    text = script.read_text()
                    self.assertIn('DATASET_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"', text)
                    self.assertIn('${DATASET_ROOT}/scripts/00_sample_for_attribution.sh', text)
                    self.assertNotIn("DATASET=", text)
                    self.assertNotIn("REFINE_ROOT", text)

    def test_sample_query_gradient_is_one_job_per_model_seed_and_algorithm(self):
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                script = ROOT / dataset / "scripts" / "02_sample_query_gradient.sh"
                self.assertTrue(script.is_file())
                text = script.read_text()
                self.assertIn('bash "${ROOT}/scripts/00_sample_for_attribution.sh"', text)
                self.assertIn('ALGORITHMS_TEXT="${ALGORITHMS:-${ALGO:-${ALGORITHM:-das}}}"', text)
                self.assertIn('SAMPLE_SEEDS_TEXT="${SAMPLE_SEEDS:-${INITIAL_SEED:-${SAMPLE_SEED:-0}}}"', text)
                self.assertIn('SAMPLE_MODEL_MODE_TAG="prompted_solo"', text)
                self.assertIn('SAMPLE_MODEL_MODE_TAG="unprompted_solo"', text)
                self.assertIn('"${PYTHON_BIN}" 02_query_gradient.py', text)
                self.assertIn('INITIAL_SEED="${SAMPLE_SEED_VALUE}"', text)

        for dataset in DATASETS:
            for system in ("h100", "vista"):
                with self.subTest(dataset=dataset, system=system):
                    script = ROOT / dataset / "tacc" / system / "sample_query_gradient.sh"
                    self.assertTrue(script.is_file())
                    text = script.read_text()
                    self.assertIn('DATASET_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"', text)
                    self.assertIn('${DATASET_ROOT}/scripts/02_sample_query_gradient.sh', text)
                    self.assertNotIn("DATASET=", text)
                    self.assertNotIn("REFINE_ROOT", text)

    def test_lds_training_scripts_select_model_mode_seed_and_subset_size(self):
        for dataset in ("cifar2", "cifar10"):
            with self.subTest(dataset=dataset, mode="prompted"):
                text = (ROOT / dataset / "scripts" / "03_lds_training.sh").read_text()
                self.assertIn('SAMPLE_MODEL_MODE="${SAMPLE_MODEL_MODE:-prompted_solo}"', text)
                self.assertIn('LDS_MODEL_TRAIN_SEED="${LDS_MODEL_TRAIN_SEED:-${LDS_TRAIN_SEED:-${TRAIN_SEED:-42}}}"', text)
                self.assertIn('--sample-model-mode "${SAMPLE_MODEL_MODE}"', text)
                self.assertIn('--model-train-seed "${LDS_MODEL_TRAIN_SEED}"', text)
                self.assertIn('--m "${LDS_M:-${LDS_NUM_SUBSETS:-100}}"', text)
                self.assertIn("--dataset-percentage", text)
                self.assertIn("--k", text)
                self.assertIn("unset LDS_K LDS_SUBSET_SIZE", text)

            with self.subTest(dataset=dataset, mode="unprompted"):
                text = (ROOT / dataset / "scripts" / "03_lds_training_unprompted.sh").read_text()
                self.assertIn("--unprompted", text)
                self.assertIn('SAMPLE_MODEL_MODE="${SAMPLE_MODEL_MODE:-unprompted_solo}"', text)
                self.assertIn('--sample-model-mode "${SAMPLE_MODEL_MODE}"', text)
                self.assertIn("unset LDS_K LDS_SUBSET_SIZE", text)

    def test_old_algorithm_entrypoints_are_removed(self):
        for dataset in DATASETS:
            for algorithm in ALGORITHMS:
                with self.subTest(dataset=dataset, algorithm=algorithm):
                    algorithm_dir = ROOT / dataset / "data_attribution" / algorithm
                    self.assertFalse((algorithm_dir / "run_attribution.py").exists())
                    self.assertFalse((algorithm_dir / "run_training.py").exists())
                    self.assertFalse((algorithm_dir / "run_eval.py").exists())
                    self.assertFalse((algorithm_dir / "script.sh").exists())

    def test_prompted_attribution_script_uses_stage_two_and_three(self):
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                text = (ROOT / dataset / "scripts" / "01_data_attribution.sh").read_text()
                self.assertIn("02_query_gradient.py", text)
                self.assertIn("03_score.py", text)
                self.assertNotIn("run_attribution.py", text)

    def test_unprompted_attribution_script_uses_stage_two_and_three(self):
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                text = (ROOT / dataset / "scripts" / "01_data_attribution_unprompted.sh").read_text()
                self.assertIn("UNPROMPTED=1", text)
                self.assertIn("02_query_gradient.py", text)
                self.assertIn("03_score.py", text)
                self.assertNotIn("unprompted_jax_attribution.py", text)


if __name__ == "__main__":
    unittest.main()
