from __future__ import annotations

import ast
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / (
    "diffusion_jax_refined/celeba/script/"
    "analyze_pint_raw_vs_adamw_topk_endpoint_clip.py"
)
SOURCE = SCRIPT.read_text()


class CelebAPintTopkEndpointClipTest(unittest.TestCase):
    def test_syntax_and_default_contract(self) -> None:
        ast.parse(SOURCE)
        self.assertIn('default="timestamp_sum_squared"', SOURCE)
        self.assertIn('default=["query_train_l2"]', SOURCE)
        self.assertIn('choices=("pint", "tracin")', SOURCE)
        self.assertIn('parser.add_argument("--topk", type=int, default=1000)', SOURCE)
        self.assertIn('default=list(range(72))', SOURCE)

    def test_full10k_and_transform_ranking_contract(self) -> None:
        self.assertIn("expected = np.arange(10_000", SOURCE)
        self.assertIn('method == "pint" and reduction == "timestamp_sum_squared"', SOURCE)
        self.assertIn('return -scores if transform == "raw_gradient" else scores', SOURCE)
        self.assertIn('return scores if transform == "raw_gradient" else -scores', SOURCE)
        self.assertIn('TRANSFORMS = ("raw_gradient", "adamw_full")', SOURCE)

    def test_method_specific_ranking_directions(self) -> None:
        tree = ast.parse(SOURCE)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "ranking_scores"
        )
        function.returns = None
        for argument in function.args.args:
            argument.annotation = None
        namespace: dict[str, object] = {}
        ast.fix_missing_locations(function)
        exec(compile(ast.Module([function], []), str(SCRIPT), "exec"), namespace)
        rank = namespace["ranking_scores"]
        self.assertEqual(rank(3.0, "pint", "raw_gradient", "timestamp_sum_squared"), 3.0)
        self.assertEqual(rank(3.0, "pint", "adamw_full", "timestamp_sum_squared"), 3.0)
        self.assertEqual(rank(3.0, "pint", "raw_gradient", "linear"), -3.0)
        self.assertEqual(rank(3.0, "pint", "adamw_full", "linear"), 3.0)
        self.assertEqual(rank(3.0, "tracin", "raw_gradient", "linear"), 3.0)
        self.assertEqual(rank(3.0, "tracin", "adamw_full", "linear"), -3.0)

    def test_endpoint_clip_definition_and_outputs(self) -> None:
        self.assertIn("similarities = train_features[indices] @ endpoint", SOURCE)
        self.assertIn('"clip_cosine_mean"', SOURCE)
        self.assertIn('"endpoint_clip_report.txt"', SOURCE)
        self.assertIn('"per_query_endpoint_clip.csv"', SOURCE)
        self.assertIn("PAIRED LABEL-SWAP raw-adamw", SOURCE)

    def test_long_steps_report_progress(self) -> None:
        self.assertIn("[clip progress]", SOURCE)
        self.assertIn("images={completed}/{total}", SOURCE)
        self.assertIn("[label-swap progress]", SOURCE)
        self.assertIn("permutations={completed}/{permutations}", SOURCE)


if __name__ == "__main__":
    unittest.main()
