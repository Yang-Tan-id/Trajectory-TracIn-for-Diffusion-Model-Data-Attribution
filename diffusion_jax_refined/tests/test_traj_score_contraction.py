from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from diffusion_jax_refined.common.stage_artifact_runner import (
    _apply_traj_checkpoint_weighting,
    _combine_multiterm_dot_scores,
    _run_fused_traj_score_batch,
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

    def test_timestamp_sum_squared_sums_checkpoints_before_square(self) -> None:
        train = {
            "train_features": np.asarray(
                [[[1.0]], [[2.0]], [[3.0]], [[4.0]]], dtype=np.float32
            ),
            "ckpt_indices": np.asarray([0, 0, 1, 1]),
            "timesteps": np.asarray([0, 1, 0, 1]),
            "term_weights": np.asarray([0.5, 0.5, 0.5, 0.5]),
        }
        query = {
            "query_features": np.ones((4, 1), dtype=np.float32),
            "ckpt_indices": np.asarray([0, 0, 1, 1]),
            "timesteps": np.asarray([0, 1, 0, 1]),
            "term_weights": np.asarray([0.5, 0.5, 0.5, 0.5]),
        }
        with mock.patch.dict(
            os.environ,
            {
                "TRACIN_SCORE_CONTRACTION": "timestamp_sum_squared",
                "TRACIN_SCORE_TQDM": "0",
            },
        ):
            result = _combine_multiterm_dot_scores(
                train,
                query,
                train_path=Path("train.npz"),
                query_path=Path("query.npz"),
            )
        expected = (0.5 * 1.0 + 0.5 * 3.0) ** 2 + (
            0.5 * 2.0 + 0.5 * 4.0
        ) ** 2
        termwise_square = 0.5 * (1.0**2 + 2.0**2 + 3.0**2 + 4.0**2)
        np.testing.assert_allclose(result, [expected])
        self.assertNotAlmostEqual(float(result[0]), termwise_square)

    def test_fused_timestamp_sum_squared_matches_nonfused_definition(self) -> None:
        train = np.asarray(
            [[[1.0]], [[2.0]], [[3.0]], [[4.0]]], dtype=np.float32
        )
        train_payload = {
            "train_features": train,
            "ckpt_indices": np.asarray([0, 0, 1, 1]),
            "timesteps": np.asarray([0, 1, 0, 1]),
            "term_weights": np.asarray([0.5, 0.5, 0.5, 0.5]),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            query_path = root / "query.npz"
            output_dir = root / "score"
            np.savez(
                query_path,
                query_features=np.ones((4, 1), dtype=np.float32),
                ckpt_indices=np.asarray([0, 0, 1, 1]),
                timesteps=np.asarray([0, 1, 0, 1]),
                term_weights=np.asarray([0.5, 0.5, 0.5, 0.5]),
            )
            with mock.patch.dict(
                os.environ,
                {
                    "TRACIN_SCORE_CONTRACTION": "timestamp_sum_squared",
                    "TRACIN_SCORE_TQDM": "0",
                    "TRACIN_ALIGN_TERMS_BY_CKPT_TIMESTEP": "1",
                },
            ):
                handled = _run_fused_traj_score_batch(
                    train_payload=train_payload,
                    train=train,
                    train_path=root / "train.npz",
                    indices=np.asarray([42]),
                    jobs=[
                        {
                            "label": "q0",
                            "query_path": str(query_path),
                            "output_dir": str(output_dir),
                        }
                    ],
                    normalize_query=False,
                    normalize_train=False,
                )
            self.assertTrue(handled)
            result = np.load(output_dir / "scores.npy")
        expected = (0.5 * 1.0 + 0.5 * 3.0) ** 2 + (
            0.5 * 2.0 + 0.5 * 4.0
        ) ** 2
        np.testing.assert_allclose(result, [expected])

    def test_previous_checkpoint_lr_shifts_checkpoint_totals(self) -> None:
        weights = np.asarray([0.1, 0.1, 0.2, 0.2, 0.3, 0.3])
        checkpoints = np.asarray([0, 0, 1, 1, 2, 2])
        with mock.patch.dict(
            os.environ,
            {"TRACIN_SCORE_CHECKPOINT_WEIGHTING": "previous_checkpoint_lr"},
        ):
            shifted = _apply_traj_checkpoint_weighting(weights, checkpoints)
        np.testing.assert_allclose(
            shifted,
            [0.0, 0.0, 0.1, 0.1, 0.2, 0.2],
        )

    def test_aligned_previous_lr_uses_query_lr_when_adamw_train_weights_are_uniform(self) -> None:
        train = np.asarray(
            [[[1.0]], [[2.0]], [[3.0]], [[4.0]]], dtype=np.float32
        )
        train_payload = {
            "train_features": train,
            "ckpt_indices": np.asarray([0, 0, 1, 1]),
            "timesteps": np.asarray([0, 1, 0, 1]),
            # AdamW-aware train features already contain their internal LR.
            "term_weights": np.asarray([0.5, 0.5, 0.5, 0.5]),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            query_path = root / "query.npz"
            output_dir = root / "score"
            np.savez(
                query_path,
                query_features=np.ones((4, 1), dtype=np.float32),
                ckpt_indices=np.asarray([0, 0, 1, 1]),
                timesteps=np.asarray([0, 1, 0, 1]),
                # Stored query totals are LR_0=0.2 and LR_1=0.4.
                term_weights=np.asarray([0.1, 0.1, 0.2, 0.2]),
            )
            with mock.patch.dict(
                os.environ,
                {
                    "TRACIN_SCORE_CONTRACTION": "timestamp_sum_squared",
                    "TRACIN_SCORE_CHECKPOINT_WEIGHTING": "previous_checkpoint_lr",
                    "TRACIN_SCORE_TQDM": "0",
                    "TRACIN_ALIGN_TERMS_BY_CKPT_TIMESTEP": "1",
                },
            ):
                handled = _run_fused_traj_score_batch(
                    train_payload=train_payload,
                    train=train,
                    train_path=root / "train.npz",
                    indices=np.asarray([42]),
                    jobs=[
                        {
                            "label": "q0",
                            "query_path": str(query_path),
                            "output_dir": str(output_dir),
                        }
                    ],
                    normalize_query=False,
                    normalize_train=False,
                )
            self.assertTrue(handled)
            result = np.load(output_dir / "scores.npy")
        # Checkpoint 0 is zero. Checkpoint 1 receives LR_0=0.2,
        # distributed equally across its two timestamps.
        expected = (0.1 * 3.0) ** 2 + (0.1 * 4.0) ** 2
        np.testing.assert_allclose(result, [expected])


if __name__ == "__main__":
    unittest.main()
