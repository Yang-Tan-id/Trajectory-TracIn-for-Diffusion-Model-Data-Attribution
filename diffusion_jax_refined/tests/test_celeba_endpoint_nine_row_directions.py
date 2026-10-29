from __future__ import annotations

import ast
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / (
    "diffusion_jax_refined/celeba/script/"
    "plot_endpoint_full10k_top10_all_variants.py"
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


class CelebAEndpointNineRowDirectionsTest(unittest.TestCase):
    def test_linear_direction_depends_on_parameter_transform(self) -> None:
        ranking_sign = load_direction_function()
        self.assertEqual(ranking_sign("raw_gradient"), +1)
        self.assertEqual(ranking_sign("adamw_full"), -1)
        with self.assertRaises(ValueError):
            ranking_sign("unknown")

    def test_pollute_stays_descending_and_linear_banks_share_sign(self) -> None:
        self.assertIn("linear_sign = linear_ranking_sign(transform)", SOURCE)
        self.assertEqual(SOURCE.count("linear_sign,"), 2)
        self.assertIn('directory,\n                +1,', SOURCE)


if __name__ == "__main__":
    unittest.main()
