import ast
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "celeba"
    / "script"
    / "run_retrac_1x50.py"
)


class CelebaRetracSharedForwardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(SCRIPT.read_text())
        cls.function = next(
            node
            for node in cls.tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "make_exact_event_train_fn"
        )

    def test_exact_path_builds_one_vjp(self):
        vjp_calls = [
            node
            for node in ast.walk(self.function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "jax"
            and node.func.attr == "vjp"
        ]
        self.assertEqual(len(vjp_calls), 1)

    def test_exact_path_does_not_recompute_selected_loss_with_grad(self):
        nested = {
            node.name: node
            for node in ast.walk(self.function)
            if isinstance(node, ast.FunctionDef)
        }
        one_position = nested["one_position"]
        grad_calls = [
            node
            for node in ast.walk(one_position)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "jax"
            and node.func.attr == "grad"
        ]
        self.assertEqual(grad_calls, [])

    def test_each_position_uses_a_one_hot_pullback(self):
        source = ast.unparse(self.function)
        self.assertIn("jax.nn.one_hot", source)
        self.assertIn("pullback(cotangent)[0]", source)


if __name__ == "__main__":
    unittest.main()
