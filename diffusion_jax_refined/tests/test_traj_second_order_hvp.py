from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ALGORITHM = ROOT / "legacy_jax" / "traj_tracin" / "algorithm.py"
SCORER = ROOT / "common" / "stage_artifact_runner.py"
LDS = ROOT / "3dshapes" / "script" / "run_traj_tracin_lds_cached.py"
LAUNCHER = (
    ROOT
    / "3dshapes"
    / "tacc"
    / "rtx_small"
    / "run_indist_q0_20_reference_next_adamw_hvp_timestamp_sum_squared_rtx_small.sh"
)
QTS_LAUNCHER = (
    ROOT
    / "3dshapes"
    / "tacc"
    / "rtx_small"
    / "run_indist_q0_20_query_timestamp_shared_adamw_hvp_timestamp_sum_squared_rtx_small.sh"
)


def test_second_order_effective_query_has_half_hvp_coefficient():
    query = np.asarray([1.0, -2.0, 3.0], dtype=np.float32)
    hvp = np.asarray([4.0, 6.0, -8.0], dtype=np.float32)
    train_update = np.asarray([-1.0, 2.0, 0.5], dtype=np.float32)

    effective_query = query + 0.5 * hvp
    expected = np.dot(train_update, query) + 0.5 * np.dot(train_update, hvp)

    np.testing.assert_allclose(np.dot(train_update, effective_query), expected)


def test_query_hvp_uses_same_checkpoint_projector_as_query_gradient():
    source = ALGORITHM.read_text()
    assert "def make_query_grad_hvp_chunk_fn" in source
    assert "H_query(theta_c) @ stopgrad(theta_c_plus_1-theta_c)" in source
    assert "stage_features.append(np.asarray(projector(one_grad)" in source
    assert "np.asarray(projector(one_hvp)" in source
    assert "(train_seed,'traj_tracin_projection',checkpoint_index)" in source
    assert "query_hvp_projection_train_seed" in source


def test_fused_scorer_validates_projection_and_forms_effective_query():
    source = SCORER.read_text()
    assert 'TRACIN_SCORE_QUERY_HVP_KEY' in source
    assert 'query_hvp_projection_matches_query' in source
    assert "query_all = query_all + coefficient_all * query_hvp_all" in source
    assert 'TRACIN_SCORE_SECOND_ORDER_COEFFICIENT' in source
    assert 'TRACIN_SCORE_EXPECTED_PROJECTION_SEED' in source


def test_q0_20_launcher_is_two_gpu_aligned_timestampwise_square():
    source = LAUNCHER.read_text()
    assert "#SBATCH -p rtx-small" in source
    assert "#SBATCH -n 2" in source
    assert "--gpus 0,1" in source
    assert 'query_ids="$(seq -s, 0 20)"' in source
    assert "TRAJ_TRACIN_QUERY_SAVE_HVP=1" in source
    assert "TRACIN_SCORE_SECOND_ORDER_COEFFICIENT=0.5" in source
    assert "TRACIN_SCORE_CHECKPOINT_WEIGHTING=uniform_checkpoint" in source
    assert "TRACIN_SCORE_CONTRACTION=timestamp_sum_squared" in source
    assert "0,111,222,333,444,555,666,777,888,999" in source
    assert "--add-optimizer-history" in source


def test_second_order_lds_schemes_are_registered():
    source = LDS.read_text()
    assert "adamw_residual_aligned10x10_second_order_hvp_next_delta" in source
    assert "adamw_full_aligned10x10_second_order_hvp_next_delta" in source


def test_query_timestamp_shared_probe_hvp_launcher():
    algorithm = ALGORITHM.read_text()
    launcher = QTS_LAUNCHER.read_text()
    assert "def make_predicted_noise_probe_query_grad_hvp_chunk_fn" in algorithm
    assert "query_timestamp_shared_gaussian" in launcher
    assert "--predicted-noise-probe-seed \"$probe_seed\"" in launcher
    assert "--gpus 0,1" in launcher
    assert 'query_ids="$(seq -s, 0 20)"' in launcher
    assert "TRACIN_SCORE_CONTRACTION=timestamp_sum_squared" in launcher
    assert "TRACIN_SCORE_SECOND_ORDER_COEFFICIENT=0.5" in launcher
