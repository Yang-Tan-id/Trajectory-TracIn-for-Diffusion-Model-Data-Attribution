from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from diffusion_jax_refined.common.stage_artifact_runner import (
    _das_score_term_indices,
    _load_das_denominator_cache,
)


class DasScoreTimestepCropTest(unittest.TestCase):
    def test_selects_all_mc_terms_for_requested_timestamps(self) -> None:
        term_ids = np.asarray(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 111, 0],
                [0, 111, 1],
                [0, 222, 0],
            ],
            dtype=np.int64,
        )
        with mock.patch.dict(os.environ, {"DAS_SCORE_TIMESTEP_ALLOWLIST": "0,111"}):
            selected = _das_score_term_indices(term_ids)
        np.testing.assert_array_equal(selected, np.asarray([0, 1, 2, 3]))

    def test_rejects_timestamp_missing_from_artifact(self) -> None:
        term_ids = np.asarray([[0, 0, 0], [0, 111, 0]], dtype=np.int64)
        with mock.patch.dict(os.environ, {"DAS_SCORE_TIMESTEP_ALLOWLIST": "0,999"}):
            with self.assertRaisesRegex(ValueError, "999"):
                _das_score_term_indices(term_ids)

    def test_full_denominator_cache_can_be_cropped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "denominator.npz"
            train_indices = np.asarray([4, 8], dtype=np.int64)
            denominator = np.arange(12, dtype=np.float32).reshape(6, 2)
            np.savez(
                path,
                denominator=denominator,
                score_indices=train_indices,
            )
            cropped = _load_das_denominator_cache(
                path,
                terms=2,
                train_indices=train_indices,
                term_indices=np.asarray([1, 4], dtype=np.int64),
            )
        np.testing.assert_array_equal(cropped, denominator[[1, 4]])


if __name__ == "__main__":
    unittest.main()
