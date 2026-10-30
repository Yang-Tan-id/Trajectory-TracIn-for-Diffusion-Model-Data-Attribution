from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "3dshapes/script/analyze_five_method_raw_vs_adamw_endpoint_perceptual.py"
)
SPEC = importlib.util.spec_from_file_location("shape3d_endpoint_perceptual", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Shape3DFiveMethodEndpointPerceptualTest(unittest.TestCase):
    def test_linear_ranking_directions(self) -> None:
        query_root = Path("/tmp/query")
        for method_key in (
            "tracin_raw",
            "tracin_query_train_l2",
            "retrac_query_train_l2",
        ):
            _raw_path, raw_sign = MODULE.score_bank_directory(
                query_root, "raw_gradient", method_key
            )
            _adamw_path, adamw_sign = MODULE.score_bank_directory(
                query_root, "adamw_full", method_key
            )
            self.assertEqual(raw_sign, +1)
            self.assertEqual(adamw_sign, -1)

    def test_pint_ranks_descending_for_both_transforms(self) -> None:
        query_root = Path("/tmp/query")
        for method_key in ("pollute_raw", "pollute_query_train_l2"):
            _raw_path, raw_sign = MODULE.score_bank_directory(
                query_root, "raw_gradient", method_key
            )
            _adamw_path, adamw_sign = MODULE.score_bank_directory(
                query_root, "adamw_full", method_key
            )
            self.assertEqual(raw_sign, +1)
            self.assertEqual(adamw_sign, +1)


if __name__ == "__main__":
    unittest.main()
