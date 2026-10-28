from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "diffusion_jax_refined"
    / "3dshapes"
    / "script"
    / "plot_endpoint_nine_row_top10.py"
)
SOURCE = SCRIPT.read_text()


def load_direction_function():
    tree = ast.parse(SOURCE)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "linear_ranking_sign"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace: dict[str, object] = {}
    exec(compile(module, str(SCRIPT), "exec"), namespace)
    return namespace["linear_ranking_sign"]


class Shape3DEndpointNineRowDirectionTest(unittest.TestCase):
    def test_linear_raw_and_adamw_use_opposite_orientations(self) -> None:
        ranking_sign = load_direction_function()
        self.assertEqual(ranking_sign("raw_gradient"), +1)
        self.assertEqual(ranking_sign("adamw_full"), -1)

    def test_both_tracin_and_retrac_use_the_transform_direction(self) -> None:
        self.assertEqual(SOURCE.count("ranking_sign=linear_sign"), 2)

    def test_pollute_timestamp_square_is_always_descending(self) -> None:
        self.assertIn("ranking_sign=+1", SOURCE)
        self.assertIn(
            '"stored timestamp-sum-squared score descending for both "',
            SOURCE,
        )


if __name__ == "__main__":
    unittest.main()
