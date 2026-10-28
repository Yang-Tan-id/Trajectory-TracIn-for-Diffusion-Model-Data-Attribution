from __future__ import annotations

import ast
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / (
    "diffusion_jax_refined/celeba/script/"
    "analyze_pint_linear_vs_timestamp_square_overlap.py"
)
SOURCE = SCRIPT.read_text()


class CelebAPintLinearVsSquareOverlapTest(unittest.TestCase):
    def test_syntax_and_full10k_contract(self) -> None:
        ast.parse(SOURCE)
        self.assertIn('default=",".join(map(str, range(72)))', SOURCE)
        self.assertIn('parser.add_argument("--topk", type=int, default=1000)', SOURCE)
        self.assertIn("expected = np.arange(10_000", SOURCE)
        self.assertIn('TRANSFORMS = ("raw_gradient", "adamw_full")', SOURCE)

    def test_reductions_variants_and_directions(self) -> None:
        self.assertIn('for reduction in ("linear", "timestamp_sum_squared")', SOURCE)
        self.assertIn("square_oriented = square_scores", SOURCE)
        self.assertIn('("raw", "score")', SOURCE)
        self.assertIn(
            '("query_train_l2", "score_query_train_l2_normalized")', SOURCE
        )

    def test_transform_specific_linear_orientation(self) -> None:
        tree = ast.parse(SOURCE)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "orient_linear_scores"
        )
        function.returns = None
        for argument in function.args.args:
            argument.annotation = None
        namespace: dict[str, object] = {}
        ast.fix_missing_locations(function)
        exec(compile(ast.Module([function], []), str(SCRIPT), "exec"), namespace)
        orient = namespace["orient_linear_scores"]
        self.assertEqual(orient(3.0, "raw_gradient"), -3.0)
        self.assertEqual(orient(3.0, "adamw_full"), 3.0)
        with self.assertRaises(ValueError):
            orient(3.0, "unknown")

    def test_overlap_and_rank_outputs(self) -> None:
        self.assertIn('"overlap_fraction": overlap / args.topk', SOURCE)
        self.assertIn('"jaccard": overlap / (2 * args.topk - overlap)', SOURCE)
        self.assertIn('"ranking_spearman": spearman(', SOURCE)
        self.assertIn('"overlap_report.txt"', SOURCE)


if __name__ == "__main__":
    unittest.main()
