import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[2]
PYTHON_SCRIPT = ROOT / "diffusion_jax_refined/celeba/script/run_retrac_1x50.py"
LAUNCH_SCRIPT = ROOT / "diffusion_jax_refined/celeba/script/run_retrac_1x50_4gpu.sh"
TOPK_PLAN = ROOT / "diffusion_jax_refined/celeba/script/celeba_top1000_250_plan.py"


class CelebaRetracBatchOneContractTest(unittest.TestCase):
    def test_launcher_defaults_to_paper_batch_one(self):
        source = LAUNCH_SCRIPT.read_text()
        self.assertIn('replay_dropout="${REPLAY_DROPOUT:-0}"', source)
        self.assertIn('replay_args=(); replay_label="paper ReTrac batch-one"', source)
        self.assertIn('replay_args=(--replay-dropout)', source)

    def test_batch_one_function_uses_singleton_inputs(self):
        tree = ast.parse(PYTHON_SCRIPT.read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "make_batch_one_event_train_fn"
        )
        source = ast.unparse(function)
        self.assertIn("x0 = x0[None, ...]", source)
        self.assertIn("noise = noise[None, ...]", source)
        self.assertIn("t = timestep[None]", source)
        self.assertIn("jax.grad(loss_fn)(params)", source)
        self.assertIn("jax.vmap(one", source)

    def test_topk_uses_batch_one_retrac_scores(self):
        source = TOPK_PLAN.read_text()
        self.assertIn("train_replay4_batch1_", source)
        self.assertNotIn("train_replay4_exact_batch_dropout_", source)


if __name__ == "__main__":
    unittest.main()
