from __future__ import annotations

import csv
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sd_artbench.artbench_lds_eval import (
    EXPECTED_MODELS,
    TRUE_F_FIELDS,
    expected_timesteps,
    index_path,
    membership_matrix,
    merge_true_f_rows,
    parse_ints,
    score_lds,
    tasks,
)


class ArtBenchLDSEvalTest(unittest.TestCase):
    def test_contracts_and_parsing(self) -> None:
        self.assertEqual(EXPECTED_MODELS, 192)
        self.assertEqual(len(tasks()), 192)
        self.assertEqual(parse_ints("0-2, 4 6"), [0, 1, 2, 4, 6])
        timesteps = expected_timesteps(1000, 50)
        self.assertEqual((len(timesteps), timesteps[0], timesteps[-1]), (50, 999, 19))

    def test_merge_rejects_incomplete_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shard.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=TRUE_F_FIELDS)
                writer.writeheader()
                writer.writerow(
                    {
                        "subset_seed": 0,
                        "subset_id": 0,
                        "query_index": 0,
                        "probe_count": 50,
                        "endpoint_difference": 1.0,
                        "trajectory_difference": 1.0,
                        "simple_loss_50x1": 1.0,
                        "noise_trajectory": 1.0,
                        "simple_loss_std": 0.0,
                        "noise_trajectory_std": 0.0,
                    }
                )
            with self.assertRaisesRegex(RuntimeError, "coverage"):
                merge_true_f_rows([path], [0])

    def test_membership_alignment_and_lds_orientation(self) -> None:
        # Patch the module constants to make a compact filesystem fixture.
        import sd_artbench.artbench_lds_eval as module

        old_banks = module.SUBSET_BANKS
        old_count = module.SUBSETS_PER_BANK
        old_size = module.EXPECTED_SUBSET_SIZE
        module.SUBSET_BANKS = (0,)
        module.SUBSETS_PER_BANK = 3
        module.EXPECTED_SUBSET_SIZE = 2
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                memberships = ([10, 11], [10, 12], [12, 13])
                for subset_id, values in enumerate(memberships):
                    path = index_path(root, 0, subset_id)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("wb") as handle:
                        pickle.dump(values, handle)
                matrix, ordered = membership_matrix(
                    root, np.asarray([10, 11, 12, 13])
                )
                np.testing.assert_array_equal(
                    matrix,
                    np.asarray([[1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 1, 1]]),
                )
                true_rows = [
                    {
                        "subset_seed": 0,
                        "subset_id": subset,
                        "query_index": 0,
                        "endpoint_difference": value,
                        "trajectory_difference": value,
                        "simple_loss_50x1": value,
                        "noise_trajectory": value,
                    }
                    for subset, value in enumerate([-3.0, -2.0, -1.0])
                ]
                # All five methods see the same synthetic score bank.  The raw
                # transform uses +1, so -(M @ score) exactly matches true-f.
                variant_scores = np.zeros((2, 4, 1, 4), dtype=np.float64)
                variant_scores[:, :, 0, :] = np.asarray([2.0, 1.0, 0.0, 1.0])
                retrac_scores = np.zeros((2, 1, 4), dtype=np.float64)
                retrac_scores[:, 0, :] = np.asarray([2.0, 1.0, 0.0, 1.0])
                artifacts = {
                    "pollute": {
                        "path": "p.npz",
                        "scores": variant_scores,
                        "indices": np.arange(4),
                        "transforms": ["raw_gradient", "adamw_full"],
                        "variants": ["raw", "query_l2", "train_l2", "query_train_l2"],
                    },
                    "tracin": {
                        "path": "t.npz",
                        "scores": variant_scores,
                        "indices": np.arange(4),
                        "transforms": ["raw_gradient", "adamw_full"],
                        "variants": ["raw", "query_l2", "train_l2", "query_train_l2"],
                    },
                    "retrac": {
                        "path": "r.npz",
                        "scores": retrac_scores,
                        "indices": np.arange(4),
                        "transforms": ["raw_gradient", "adamw_full"],
                        "variants": None,
                    },
                }
                rows = score_lds(
                    true_f_rows=true_rows,
                    membership=matrix,
                    ordered_tasks=ordered,
                    artifacts=artifacts,
                    query_ids=[0],
                )
                pint_raw = next(
                    row
                    for row in rows
                    if row["scope"] == "all_192"
                    and row["target"] == "simple_loss_50x1"
                    and row["method"] == "PINT / RAW"
                    and row["transform"] == "raw_gradient"
                )
                self.assertAlmostEqual(pint_raw["spearman"], 1.0)
        finally:
            module.SUBSET_BANKS = old_banks
            module.SUBSETS_PER_BANK = old_count
            module.EXPECTED_SUBSET_SIZE = old_size


if __name__ == "__main__":
    unittest.main()
