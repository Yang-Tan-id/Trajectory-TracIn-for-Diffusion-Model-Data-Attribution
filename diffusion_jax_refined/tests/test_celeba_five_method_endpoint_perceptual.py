from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "celeba/script/analyze_five_method_raw_vs_adamw_endpoint_perceptual.py"
)
SPEC = importlib.util.spec_from_file_location("celeba_endpoint_perceptual", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CelebAFiveMethodEndpointPerceptualTest(unittest.TestCase):
    def test_similarity_improvement_is_adamw_minus_raw(self) -> None:
        raw = np.asarray([0.5, 0.7])
        adamw = np.asarray([0.6, 0.8])
        np.testing.assert_allclose(
            MODULE.oriented_improvement(raw, adamw, +1), [0.1, 0.1]
        )

    def test_distance_improvement_is_raw_minus_adamw(self) -> None:
        raw = np.asarray([0.5, 0.7])
        adamw = np.asarray([0.4, 0.6])
        np.testing.assert_allclose(
            MODULE.oriented_improvement(raw, adamw, -1), [0.1, 0.1]
        )

    def test_extreme_selection_respects_common_orientation(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scores = np.arange(10_000, dtype=np.float64)
            np.save(root / "scores.npy", scores)
            np.save(root / "score_indices.npy", np.arange(10_000))
            self.assertEqual(
                MODULE.select_extreme_k(
                    root, k=2, ranking_sign=+1, extreme="top"
                ),
                {9998, 9999},
            )
            self.assertEqual(
                MODULE.select_extreme_k(
                    root, k=2, ranking_sign=+1, extreme="bottom"
                ),
                {0, 1},
            )

    def test_bottom_reverses_adamw_ascending_storage(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scores = -np.arange(10_000, dtype=np.float64)
            np.save(root / "scores.npy", scores)
            np.save(root / "score_indices.npy", np.arange(10_000))
            self.assertEqual(
                MODULE.select_extreme_k(
                    root, k=2, ranking_sign=-1, extreme="top"
                ),
                {9998, 9999},
            )
            self.assertEqual(
                MODULE.select_extreme_k(
                    root, k=2, ranking_sign=-1, extreme="bottom"
                ),
                {0, 1},
            )


if __name__ == "__main__":
    unittest.main()
