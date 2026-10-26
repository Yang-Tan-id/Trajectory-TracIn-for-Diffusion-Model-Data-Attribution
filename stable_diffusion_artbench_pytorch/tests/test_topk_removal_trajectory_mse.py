from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src/sd_artbench/evaluate_topk_removal_trajectory_mse.py"


class TopKRemovalTrajectoryMseTest(unittest.TestCase):
    def test_script_compiles_and_records_every_ddim_step(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn("callback=record_state", source)
        self.assertIn("callback_steps=1", source)
        self.assertIn("len(states) != steps", source)
        self.assertIn("steps_offset=stride - 1", source)

    def test_all_four_methods_are_included(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        for method in (
            "endpoint_pollute_query_train_l2",
            "endpoint_tracin_raw",
            "endpoint_tracin_query_train_l2",
            "retrac_query_train_l2",
        ):
            self.assertIn(f'"{method}"', source)

    def test_trajectory_mse_formula(self) -> None:
        namespace: dict[str, object] = {"np": np}
        module = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        function = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "trajectory_metrics"
        )
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(SCRIPT), "exec"), namespace)
        metric = namespace["trajectory_metrics"]
        reference = np.zeros((2, 1, 2, 2), dtype=np.float32)
        candidate = np.stack(
            [np.ones((1, 2, 2)), np.full((1, 2, 2), 3.0)], axis=0
        )
        per_step, mse, rmse = metric(reference, candidate)
        np.testing.assert_allclose(per_step, [1.0, 9.0])
        self.assertAlmostEqual(mse, 5.0)
        self.assertAlmostEqual(rmse, np.sqrt(5.0))


if __name__ == "__main__":
    unittest.main()
