from __future__ import annotations

import os
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from diffusion_jax_refined.common.stage_artifact_runner import (
    _combine_multiterm_dot_scores,
)


class TrajScoreContractionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.train = {
            "train_features": np.asarray([[[1.0, 0.0]], [[2.0, 0.0]]]),
            "ckpt_indices": np.asarray([0, 0]),
            "timesteps": np.asarray([0, 1]),
            "term_weights": np.asarray([0.25, 0.75]),
        }
        self.query = {
            "query_features": np.asarray([[3.0, 0.0], [4.0, 0.0]]),
            "ckpt_indices": np.asarray([0, 0]),
            "timesteps": np.asarray([0, 1]),
            "term_weights": np.asarray([0.25, 0.75]),
        }

    def score(self, contraction: str) -> np.ndarray:
        with mock.patch.dict(
            os.environ,
            {"TRACIN_SCORE_CONTRACTION": contraction, "TRACIN_SCORE_TQDM": "0"},
        ):
            return _combine_multiterm_dot_scores(
                self.train,
                self.query,
                train_path=Path("train.npz"),
                query_path=Path("query.npz"),
            )

    def test_linear_weights_each_term(self) -> None:
        np.testing.assert_allclose(self.score("linear"), [0.25 * 3.0 + 0.75 * 8.0])

    def test_squared_squares_each_term_before_weighting(self) -> None:
        expected = 0.25 * 3.0**2 + 0.75 * 8.0**2
        wrong_square_after_sum = (0.25 * 3.0 + 0.75 * 8.0) ** 2
        result = self.score("squared")
        np.testing.assert_allclose(result, [expected])
        self.assertNotAlmostEqual(float(result[0]), wrong_square_after_sum)

    def test_absolute_takes_each_absolute_value_before_weighting(self) -> None:
        query = dict(self.query)
        query["query_features"] = np.asarray([[3.0, 0.0], [-4.0, 0.0]])
        with mock.patch.dict(
            os.environ,
            {"TRACIN_SCORE_CONTRACTION": "absolute", "TRACIN_SCORE_TQDM": "0"},
        ):
            result = _combine_multiterm_dot_scores(
                self.train,
                query,
                train_path=Path("train.npz"),
                query_path=Path("query.npz"),
            )
        expected = 0.25 * abs(3.0) + 0.75 * abs(-8.0)
        wrong_absolute_after_sum = abs(0.25 * 3.0 + 0.75 * -8.0)
        np.testing.assert_allclose(result, [expected])
        self.assertNotAlmostEqual(float(result[0]), wrong_absolute_after_sum)


if __name__ == "__main__":
    unittest.main()
