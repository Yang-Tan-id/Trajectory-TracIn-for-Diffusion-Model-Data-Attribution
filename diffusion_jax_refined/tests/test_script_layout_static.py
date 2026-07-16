from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("cifar2", "cifar10", "artbench")
ALGORITHMS = ("das", "dtrak", "end_tracin", "journey_trak", "traj_tracin")


class TestScriptLayoutStatic(unittest.TestCase):
    def test_legacy_tacc_scripts_are_removed(self):
        self.assertFalse((ROOT / "cifar2" / "tacc").exists())
        self.assertTrue((ROOT / "tacc" / "h100" / "script_0.sh").is_file())
        self.assertTrue((ROOT / "tacc" / "vista" / "script_0.sh").is_file())

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
        for system in ("h100", "vista"):
            with self.subTest(system=system):
                text = (ROOT / "tacc" / system / "script_0.sh").read_text()
                self.assertIn('DATASET="${DATASET:-cifar2}"', text)
                self.assertIn('TRAIN_MODE="${TRAIN_MODE:-prompted_multi}"', text)
                self.assertIn('TRAIN_MODES="${TRAIN_MODES:-${TRAIN_MODE}}"', text)
                self.assertIn('GPU_IDS="${GPU_IDS:-0,1,2,3}"', text)
                self.assertIn('JAX_NUM_DEVICES', text)
                self.assertIn('${REFINE_ROOT}/${DATASET}/scripts/script_0.sh', text)

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
                        self.assertIn("run_stage_config", stage_text)

        for system in ("h100", "vista"):
            with self.subTest(system=system):
                script = ROOT / "tacc" / system / "datapoint_gradients.sh"
                self.assertTrue(script.is_file())
                text = script.read_text()
                self.assertIn('DATASET="${DATASET:-cifar2}"', text)
                self.assertIn('ALGORITHMS="${ALGORITHMS:-${ALGO:-${ALGORITHM:-das}}}"', text)
                self.assertIn('GPU_IDS="${GPU_IDS:-0,1,2,3}"', text)
                self.assertIn('${REFINE_ROOT}/${DATASET}/scripts/01_datapoint_gradients.sh', text)

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


if __name__ == "__main__":
    unittest.main()
