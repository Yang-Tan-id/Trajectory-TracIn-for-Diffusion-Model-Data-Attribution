import json
import pickle
from pathlib import Path

import numpy as np

from sd_artbench.das_topk_removal import load_query_scores, select_topk


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "run_artbench_das_lambda1000_top1000_q0_q59_4gpu.sh"


def test_load_query_scores_uses_query_ids_not_row_number(tmp_path):
    path = tmp_path / "scores.npz"
    np.savez(
        path,
        scores=np.asarray([[1.0, 2.0], [9.0, 8.0]]),
        query_ids=np.asarray([7, 3]),
        train_metadata_indices=np.asarray([10, 20]),
        manifest_json=np.asarray(json.dumps({"damping": 1000.0})),
    )
    scores, indices, manifest = load_query_scores(path, 3)
    np.testing.assert_array_equal(scores, [9.0, 8.0])
    np.testing.assert_array_equal(indices, [10, 20])
    assert manifest["damping"] == 1000.0


def test_das_topk_is_descending_stable():
    order = select_topk(
        np.asarray([3.0, 9.0, 9.0, 1.0]), np.asarray([10, 20, 30, 40]), 2
    )
    np.testing.assert_array_equal(order, [1, 2])


def test_launcher_contract():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "seq -s, 0 59" in text
    assert "--methods das_lambda_1000" in text
    assert "--top-k 1000" in text
    assert "evaluate_topk_removal_trajectory_mse" in text
    assert "evaluate_topk_removal_perceptual" in text
    assert "combine_das_removal_metrics" in text
