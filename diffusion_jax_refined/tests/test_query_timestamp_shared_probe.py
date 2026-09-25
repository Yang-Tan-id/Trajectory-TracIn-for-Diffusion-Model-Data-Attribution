import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ALGORITHM = ROOT / "legacy_jax" / "traj_tracin" / "algorithm.py"
RUNNER = ROOT / "3dshapes" / "script" / "run_traj_tracin_queries_and_scores.py"


class QueryTimestampSharedProbeTests(unittest.TestCase):
    def test_key_contract_includes_query_and_timestamp_but_not_checkpoint(self) -> None:
        source = ALGORITHM.read_text()
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "query_timestamp_shared_predicted_noise_probe_key"
        )
        self.assertEqual(
            ["seed", "query_seed", "timestep", "probe_index"],
            [argument.arg for argument in function.args.args],
        )
        function_source = ast.get_source_segment(source, function)
        self.assertIn("(0x51545350, query_seed, timestep)", function_source)
        self.assertNotIn("checkpoint_index", function_source)

    def test_runner_exposes_query_timestamp_shared_mode(self) -> None:
        self.assertIn(
            '"query_timestamp_shared_gaussian"',
            RUNNER.read_text(),
        )


if __name__ == "__main__":
    unittest.main()
