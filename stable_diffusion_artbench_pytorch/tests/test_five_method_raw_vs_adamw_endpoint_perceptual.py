from __future__ import annotations

import unittest

import numpy as np

from sd_artbench.analyze_five_method_raw_vs_adamw_endpoint_perceptual import (
    oriented_improvement,
    select_extreme_k,
)


class FiveMethodRawVsAdamWEndpointPerceptualTest(unittest.TestCase):
    def test_similarity_improvement_is_adamw_minus_raw(self) -> None:
        raw = np.asarray([0.5, 0.7])
        adamw = np.asarray([0.6, 0.8])
        np.testing.assert_allclose(
            oriented_improvement(raw, adamw, +1), [0.1, 0.1]
        )

    def test_distance_improvement_is_raw_minus_adamw(self) -> None:
        raw = np.asarray([0.5, 0.7])
        adamw = np.asarray([0.4, 0.6])
        np.testing.assert_allclose(
            oriented_improvement(raw, adamw, -1), [0.1, 0.1]
        )

    def test_extreme_selection_respects_common_ranking_orientation(self) -> None:
        scores = np.asarray([3.0, 1.0, 4.0, 2.0])
        indices = np.asarray([10, 11, 12, 13])
        self.assertEqual(
            select_extreme_k(
                scores, indices, k=2, ranking_sign=+1, extreme="top"
            ),
            {10, 12},
        )
        self.assertEqual(
            select_extreme_k(
                scores, indices, k=2, ranking_sign=+1, extreme="bottom"
            ),
            {11, 13},
        )

    def test_bottom_selection_reverses_ascending_stored_scores(self) -> None:
        scores = np.asarray([-3.0, -1.0, -4.0, -2.0])
        indices = np.asarray([10, 11, 12, 13])
        self.assertEqual(
            select_extreme_k(
                scores, indices, k=2, ranking_sign=-1, extreme="top"
            ),
            {10, 12},
        )
        self.assertEqual(
            select_extreme_k(
                scores, indices, k=2, ranking_sign=-1, extreme="bottom"
            ),
            {11, 13},
        )


if __name__ == "__main__":
    unittest.main()
