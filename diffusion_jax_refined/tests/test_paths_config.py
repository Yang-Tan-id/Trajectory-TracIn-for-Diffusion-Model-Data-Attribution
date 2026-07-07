from __future__ import annotations

import os
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config_loader import load_config


CIFAR2_CONFIG = ROOT / "cifar2" / "dataset_config.py"


@contextmanager
def patched_env(**updates: str):
    old = {key: os.environ.get(key) for key in updates}
    try:
        os.environ.update({key: str(value) for key, value in updates.items()})
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class TestCifar2PathConfig(unittest.TestCase):
    def load_cifar2(self, *, query: str, initial_seed: int = 42, experiment: str = "experiment1_42"):
        with patched_env(QUERY=query, INITIAL_SEED=str(initial_seed), EXPERIMENT_TAG=experiment):
            return load_config(CIFAR2_CONFIG)

    def test_query_and_initial_seed_are_in_attribution_and_eval_roots(self):
        cfg = self.load_cifar2(query="horse", initial_seed=42)
        self.assertTrue(str(cfg.ATTRIBUTION_RUN_ROOT).endswith("attribution_score/query_horse/initial_seed_42"))
        self.assertTrue(str(cfg.EVAL_RUN_ROOT).endswith("eval/query_horse/initial_seed_42"))

    def test_multiple_queries_map_to_distinct_folders(self):
        folders = {}
        for query in ("horse", "automobile", "horse,automobile", "horse+auto——mobile"):
            cfg = self.load_cifar2(query=query, initial_seed=42)
            folders[query] = str(cfg.ATTRIBUTION_RUN_ROOT)
        self.assertEqual(len(set(folders.values())), len(folders), folders)
        self.assertIn("query_horse_automobile", folders["horse,automobile"])
        self.assertNotIn(",", folders["horse,automobile"])
        self.assertNotIn("——", folders["horse+auto——mobile"])

    def test_score_index_ranges_are_parsed_as_user_facing_one_based_ranges(self):
        with patched_env(QUERY="horse", INITIAL_SEED="42", SCORE_INDEX_RANGES="1-2000,2001-4000"):
            cfg = load_config(CIFAR2_CONFIG)
        self.assertEqual(cfg.SCORE_INDEX_RANGES, ((1, 2000), (2001, 4000)))

    def test_default_cifar2_query_sample_dir_contains_query_and_model_seed(self):
        cfg = self.load_cifar2(query="automobile", initial_seed=42)
        sample_dir = str(cfg.ATTRIBUTION_SAMPLE_DIR)
        self.assertIn("prompt_automobile", sample_dir)
        self.assertIn("ckpt_seed_42_epoch_0200", sample_dir)


if __name__ == "__main__":
    unittest.main()
