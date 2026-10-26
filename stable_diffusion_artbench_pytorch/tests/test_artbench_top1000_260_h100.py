from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
SRC = PACKAGE_ROOT / "src"
sys.path.insert(0, str(SRC))

from sd_artbench import artbench_top1000_260_plan as PLAN
from sd_artbench import topk_removal


RUNNER = PACKAGE_ROOT / "tacc/h100/run_artbench_top1000_260_h100.sh"
SLOT = PACKAGE_ROOT / "tacc/h100/run_artbench_top1000_260_slot.sh"
STAGER = PACKAGE_ROOT / "tacc/h100/stage_artbench_top1000_260_from_school.sh"
SUBMITTER = PACKAGE_ROOT / "tacc/h100/submit_artbench_top1000_260_h100.sh"
SETUP = PACKAGE_ROOT / "tacc/h100/setup_artbench_tacc_env.sh"

class ArtBenchTop1000260H100Test(unittest.TestCase):
    def test_exact_task_plan(self) -> None:
        tasks = PLAN.build_tasks()
        self.assertEqual(len(tasks), 260)
        self.assertEqual(len(set(tasks)), 260)
        for method in topk_removal.METHODS:
            ids = [query for task_method, query in tasks if task_method == method]
            self.assertEqual(ids, list(range(10, 60)))
        random_ids = [
            query
            for method, query in tasks
            if method == topk_removal.RANDOM_METHOD
        ]
        self.assertEqual(random_ids, list(range(60)))
        self.assertEqual(len(PLAN.SCORE_RELATIVE_PATHS), 3)
        self.assertTrue(PLAN.SCORE_RELATIVE_PATHS[0].endswith("endpoint_pollute_timestamp_sum_squared.npz"))
        self.assertTrue(PLAN.SCORE_RELATIVE_PATHS[1].endswith("endpoint_tracin.npz"))
        self.assertTrue(PLAN.SCORE_RELATIVE_PATHS[2].endswith("retrac.npz"))

    def test_random_selection_is_deterministic_query_specific(self) -> None:
        indices = np.arange(5000, dtype=np.int64)
        first = topk_removal.random_scores(
            query_index=0, base_indices=indices, random_seed=2026
        )[0]
        repeated = topk_removal.random_scores(
            query_index=0, base_indices=indices, random_seed=2026
        )[0]
        second_query = topk_removal.random_scores(
            query_index=1, base_indices=indices, random_seed=2026
        )[0]
        np.testing.assert_array_equal(first, repeated)
        self.assertFalse(np.array_equal(first, second_query))
        removed, _, _ = topk_removal.select_topk(
            first, indices, top_k=1000, ranking_sign=1
        )
        self.assertEqual(removed.shape, (1000,))
        self.assertEqual(len(np.unique(removed)), 1000)

    def test_ranking_contracts_and_validate_stage(self) -> None:
        source = (SRC / "sd_artbench/topk_removal.py").read_text()
        self.assertIn("endpoint-pollute timestamp-sum-squared", source)
        self.assertIn("query/train-L2 score descending", source)
        self.assertIn("endpoint-TracIn {variant} linear score ascending", source)
        self.assertIn("ReTrac query/train-L2 linear score ascending", source)
        self.assertIn('choices=("validate", "prepare", "train", "all")', source)
        self.assertIn("DEFAULT_RANDOM_SEED = 2026", source)

    def test_exact_score_files_variants_and_directions(self) -> None:
        transforms = np.asarray(["raw_gradient", "adamw_full"])
        variants = np.asarray(["raw", "query_l2", "train_l2", "query_train_l2"])
        indices = np.arange(5, dtype=np.int64)
        endpoint_scores = np.zeros((2, 4, 60, 5), dtype=np.float64)
        endpoint_scores[1, 3, 10] = np.arange(5) + 100.0
        endpoint_scores[1, 0, 10] = np.arange(5) + 200.0
        retrac_scores = np.zeros((2, 60, 5), dtype=np.float64)
        retrac_scores[1, 10] = np.arange(5) + 300.0

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            endpoint = root / "endpoint"
            endpoint.mkdir()
            for filename in (
                "endpoint_pollute_timestamp_sum_squared.npz",
                "endpoint_tracin.npz",
            ):
                np.savez(
                    endpoint / filename,
                    scores=endpoint_scores,
                    train_metadata_indices=indices,
                    transform_names=transforms,
                    variant_names=variants,
                )
            retrac_path = root / "retrac.npz"
            np.savez(
                retrac_path,
                scores=retrac_scores,
                train_metadata_indices=indices,
                transform_names=transforms,
            )

            expected = {
                "endpoint_pollute_query_train_l2": (100.0, 1),
                "endpoint_tracin_raw": (200.0, -1),
                "endpoint_tracin_query_train_l2": (100.0, -1),
                "retrac_query_train_l2": (300.0, -1),
            }
            for method, (first_score, sign) in expected.items():
                scores, found_indices, found_sign, _ = topk_removal.method_scores(
                    method=method,
                    query_index=10,
                    endpoint_root=endpoint,
                    retrac_path=retrac_path,
                )
                self.assertEqual(float(scores[0]), first_score)
                self.assertEqual(found_sign, sign)
                np.testing.assert_array_equal(found_indices, indices)

    def test_h100_runner_is_16_slot_resume_safe(self) -> None:
        runner = RUNNER.read_text()
        slot = SLOT.read_text()
        self.assertIn("#SBATCH -N 4", runner)
        self.assertIn("#SBATCH -n 16", runner)
        self.assertIn("#SBATCH --ntasks-per-node=4", runner)
        self.assertIn("#SBATCH -t 48:00:00", runner)
        self.assertIn("[preflight 1/4] unit tests", runner)
        self.assertIn("[preflight 2/4] exact dataset and score contracts", runner)
        self.assertIn("local_files_only=True", runner)
        self.assertIn("for slot in $(seq 0 15)", runner)
        self.assertIn("task_index+=slot_count", slot)
        self.assertIn("--stage all", slot)
        self.assertIn("--mode audit", runner)

    def test_transfer_direction_is_school_to_tacc_only(self) -> None:
        stager = STAGER.read_text()
        submitter = SUBMITTER.read_text()
        self.assertIn('"$tacc:$tacc_repo/"', stager)
        self.assertIn("--files-from=\"$manifest\"", stager)
        self.assertIn("5005 exact files", stager)
        self.assertIn("miniSD Hugging Face snapshot", stager)
        self.assertNotIn("SCHOOL_REMOTE", submitter)
        self.assertNotIn("rsync", submitter)
        self.assertIn("no school-server connection is attempted", submitter)

    def test_shell_syntax(self) -> None:
        for script in (RUNNER, SLOT, STAGER, SUBMITTER, SETUP):
            result = subprocess.run(
                ["bash", "-n", str(script)], capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
