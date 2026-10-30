from __future__ import annotations

import unittest

import numpy as np

from sd_artbench.analyze_five_method_raw_vs_adamw_endpoint_perceptual import (
    oriented_improvement,
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


if __name__ == "__main__":
    unittest.main()
