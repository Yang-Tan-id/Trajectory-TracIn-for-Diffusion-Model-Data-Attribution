from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "3dshapes" / "script" / "run_retrac_endpoint_topk_removal.py"
SPEC = importlib.util.spec_from_file_location("run_retrac_endpoint_topk_removal", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RetracEndpointTopKRemovalTest(unittest.TestCase):
    def write_scores(self, root: Path) -> tuple[np.ndarray, np.ndarray]:
        indices = np.arange(MODULE.ATTRIBUTED_SIZE, dtype=np.int64)[::-1]
        scores = np.linspace(-5.0, 5.0, MODULE.ATTRIBUTED_SIZE, dtype=np.float64)
        np.save(root / "scores.npy", scores)
        np.save(root / "score_indices.npy", indices)
        return scores, indices

    def test_retrac_uses_negative_score_for_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scores, indices = self.write_scores(root)
            removed, raw, ranking = MODULE.select_topk(root, -1, 400)
            expected = np.argsort(scores, kind="stable")[:400]
            np.testing.assert_array_equal(removed, indices[expected])
            np.testing.assert_array_equal(raw, scores[expected])
            np.testing.assert_array_equal(ranking, -scores[expected])

    def test_endpoint_uses_stored_score_for_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scores, indices = self.write_scores(root)
            removed, raw, ranking = MODULE.select_topk(root, 1, 1000)
            expected = np.argsort(-scores, kind="stable")[:1000]
            np.testing.assert_array_equal(removed, indices[expected])
            np.testing.assert_array_equal(raw, scores[expected])
            np.testing.assert_array_equal(ranking, scores[expected])

    def test_score_namespaces_and_variants(self) -> None:
        record = {"prompt": "shape_cube,object_hue_0", "initial_seed": 100}
        retrac, retrac_sign, _ = MODULE.score_spec(
            result_root=Path("result"),
            train_seed=42,
            record=record,
            method="retrac_adamw_both_l2_neg",
        )
        endpoint, endpoint_sign, _ = MODULE.score_spec(
            result_root=Path("result"),
            train_seed=42,
            record=record,
            method="endpoint_pollute_adamw_timestamp_train_l2",
        )
        self.assertEqual(retrac_sign, -1)
        self.assertEqual(retrac.name, "score_query_train_l2_normalized")
        self.assertIn("paper_retrac_adamw_full_exact4_endpoint100x1", str(retrac))
        self.assertEqual(endpoint_sign, 1)
        self.assertEqual(endpoint.name, "score_train_l2_normalized")
        self.assertIn("timestamp_sum_squared_aligned100x1", str(endpoint))

    def test_dtrak_rtx_workers_use_ibrun(self) -> None:
        launcher = (
            ROOT
            / "3dshapes"
            / "tacc"
            / "rtx_small"
            / "run_dtrak_three_objectives_100x1_q0_99_rtx_small.sh"
        ).read_text()
        self.assertIn('ibrun -n 1 -o "$task_offset"', launcher)
        self.assertIn("run_train_objective simple_loss 0", launcher)
        self.assertIn("run_train_objective square 1", launcher)
        self.assertIn("run_query_shard 0 0", launcher)
        self.assertIn("run_query_shard 1 1", launcher)


if __name__ == "__main__":
    unittest.main()
