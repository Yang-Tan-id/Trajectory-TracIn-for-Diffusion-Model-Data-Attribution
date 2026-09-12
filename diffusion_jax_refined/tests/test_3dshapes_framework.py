from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
THREED = ROOT / "3dshapes"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Test3DShapesFramework(unittest.TestCase):
    def test_balanced_selection_is_seeded_and_five_per_group(self):
        prep = load(THREED / "script" / "prepare_3dshapes.py", "prepare_3dshapes_test")
        labels = np.moveaxis(
            np.indices((10, 10, 10, 8, 4, 15), dtype=np.int16), 0, -1
        ).reshape(-1, 6).astype(np.float64)
        a = prep.select_balanced_indices(labels, seed=42, samples_per_group=5)
        b = prep.select_balanced_indices(labels, seed=42, samples_per_group=5)
        self.assertEqual(a.shape, (20000,))
        np.testing.assert_array_equal(a, b)
        chosen = labels[a][:, (0, 1, 2, 4)].astype(np.int64)
        _, counts = np.unique(chosen, axis=0, return_counts=True)
        self.assertEqual(len(counts), 4000)
        self.assertTrue(np.all(counts == 5))

    def test_conditions_are_34_way_with_four_active_tokens(self):
        prep = load(THREED / "script" / "prepare_3dshapes.py", "prepare_3dshapes_conditions_test")
        factor_ids = np.asarray([[2, 3, 4, 0, 1, 0], [9, 8, 7, 0, 3, 0]], dtype=np.int16)
        labels, names = prep.build_conditions(factor_ids)
        self.assertEqual(labels.shape, (2, 34))
        self.assertEqual(len(names), 34)
        np.testing.assert_array_equal(labels.sum(axis=1), np.asarray([4, 4]))

    def test_query_generation_is_deterministic_and_allows_repeated_categories(self):
        query_mod = load(THREED / "script" / "build_queries.py", "build_3dshapes_queries_test")
        first = query_mod.queries()
        self.assertEqual(first, query_mod.queries())
        self.assertEqual([x["query_seed"] for x in first], list(range(10)))
        self.assertTrue(all(len(x["labels"]) == 4 and len(set(x["labels"])) == 4 for x in first))
        categories = [[str(token).split("_hue_")[0] for token in x["labels"] if "_hue_" in str(token)] for x in first]
        self.assertTrue(any(len(group) != len(set(group)) for group in categories))

    def test_config_matches_requested_contract(self):
        cfg = load(THREED / "dataset_config.py", "three_d_shapes_config_test")
        self.assertEqual(cfg.NUM_CLASSES, 34)
        self.assertEqual(cfg.COMMON_CIFAR["max_train_points"], 5000)
        self.assertTrue(cfg.COMMON_CIFAR["random_subset"])
        self.assertIsNone(cfg.COMMON_CIFAR["score_index_ranges"])
        self.assertEqual(cfg.ATTRIBUTION_CONFIGS["traj_tracin"]["parameter_source"], "raw")
        self.assertEqual(cfg.ATTRIBUTION_CONFIGS["traj_tracin"]["ddim_steps"], 1000)
        self.assertEqual(cfg.ATTRIBUTION_CONFIGS["traj_tracin"]["num_traj_snapshots"], 10)
        self.assertEqual(cfg.ATTRIBUTION_CONFIGS["traj_tracin"]["train_mc_samples"], 10)
        das_timesteps = cfg.ATTRIBUTION_CONFIGS["das"]["timesteps"]
        self.assertEqual(len(das_timesteps), 100)
        self.assertEqual((das_timesteps[0], das_timesteps[-1]), (0, 999))
        self.assertEqual(len(set(das_timesteps)), 100)
        self.assertEqual(cfg.ATTRIBUTION_CONFIGS["das"]["num_mc_noise"], 1)
        self.assertEqual(min(cfg.DAS_DAMPING_SWEEP_VALUES), 0.1)
        self.assertEqual(max(cfg.DAS_DAMPING_SWEEP_VALUES), 10000)

    def test_lds_gpu_workers_split_64_models_without_overlap(self):
        launcher = load(
            THREED / "lds" / "run_training_multi_gpu.py",
            "three_d_shapes_lds_multi_gpu_test",
        )
        chunks = launcher.subset_chunks(64, 4)
        self.assertEqual([len(chunk) for chunk in chunks], [16, 16, 16, 16])
        self.assertEqual(chunks[0], list(range(0, 64, 4)))
        self.assertEqual(chunks[3], list(range(3, 64, 4)))
        self.assertEqual(sorted(value for chunk in chunks for value in chunk), list(range(64)))

    def test_lds_workers_reuse_prepared_subset_files(self):
        common_train = (ROOT / "common" / "lds_model_train.py").read_text()
        launcher = (THREED / "lds" / "run_training_multi_gpu.py").read_text()
        self.assertIn('"--reuse-prepared"', common_train)
        self.assertIn('"--finalize-only"', common_train)
        self.assertIn('base_cmd + ["--dry-run"]', launcher)
        self.assertIn('base_cmd + ["--reuse-prepared", "--finalize-only"]', launcher)


if __name__ == "__main__":
    unittest.main()
