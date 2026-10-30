from __future__ import annotations

import unittest

import numpy as np

from sd_artbench.analyze_five_method_raw_vs_adamw_endpoint_clip import label_swap


class FiveMethodRawVsAdamWEndpointClipTest(unittest.TestCase):
    def test_label_swap_detects_large_positive_shift(self) -> None:
        diff = np.asarray([1.0] * 20)
        p1 = label_swap(
            diff, permutations=10_000, seed=0, two_sided=False
        )
        p2 = label_swap(
            diff, permutations=10_000, seed=1, two_sided=True
        )
        self.assertLess(p1, 0.001)
        self.assertLess(p2, 0.001)

    def test_label_swap_null_is_not_extreme(self) -> None:
        diff = np.asarray([1.0, -1.0] * 10)
        p2 = label_swap(
            diff, permutations=10_000, seed=2, two_sided=True
        )
        self.assertEqual(p2, 1.0)


if __name__ == "__main__":
    unittest.main()
