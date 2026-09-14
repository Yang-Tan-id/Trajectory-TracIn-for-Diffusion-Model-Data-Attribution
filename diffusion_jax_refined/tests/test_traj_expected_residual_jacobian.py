from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ALGORITHM = ROOT / "legacy_jax" / "traj_tracin" / "algorithm.py"
LAUNCHER = (
    ROOT
    / "3dshapes"
    / "tacc"
    / "rtx_small"
    / "run_traj_tracin_expected_residual_jacobian_train_rtx_small.sh"
)
PIPELINE = (
    ROOT
    / "3dshapes"
    / "tacc"
    / "rtx_small"
    / "run_expected_residual_jacobian_pipeline_rtx_small.sh"
)
SCORER = ROOT / "3dshapes" / "script" / "run_expected_residual_jacobian_scores.py"


def test_expected_residual_jacobian_is_not_expected_loss_gradient():
    residuals = np.asarray([1.0, 3.0])
    jacobians = np.asarray([[2.0, 0.0], [0.0, 4.0]])

    expected_loss_gradient = np.mean(residuals[:, None] * jacobians, axis=0)
    decomposed = residuals.mean() * jacobians.mean(axis=0)

    assert not np.allclose(expected_loss_gradient, decomposed)
    np.testing.assert_allclose(decomposed, np.asarray([2.0, 4.0]))


def test_algorithm_saves_both_normalized_contractions_and_norm():
    source = ALGORITHM.read_text()
    assert "TRAJ_TRACIN_TRAIN_DECOMPOSE_RESIDUAL_JACOBIAN" in source
    assert "train_jacobian_norms=np.stack(train_jacobian_norm_terms" in source
    assert "train_features_v_l2_normalized=np.stack" in source
    assert "hutchinson_projected_expected_jacobian_transpose_expected_residual_over_frobenius_norm" in source
    assert "TRAJ_TRACIN_JACOBIAN_NORM_PROBES" in source
    assert 'archive.open("train_jacobian_norms.npy"' in source
    assert "mean_residual = jax.lax.stop_gradient" in source
    assert "contraction_contributions" in source
    assert "normalizer * jnp.mean" in source
    assert "def residual_contraction" not in source
    assert "projected_residual * unit_probe_gradient" in source


def test_rtx_launcher_uses_isolated_resumable_artifact():
    source = LAUNCHER.read_text()
    assert "#SBATCH -p rtx-small" in source
    assert "TRAJ_TRACIN_TRAIN_DECOMPOSE_RESIDUAL_JACOBIAN=1" in source
    assert "TRAJ_QUERY_OBJECTIVE=trajectory_predicted_noise_probe" in source
    assert "TRAJ_TRAIN_MC_SAMPLES=10" in source
    assert "TRAJ_NUM_SNAPSHOTS=10" in source
    assert "TRAJ_TRACIN_JACOBIAN_NORM_PROBES" in source
    assert 'TRAJ_TRACIN_JACOBIAN_NORM_PROBES:-1' in source
    assert "traj_tracin_expected_residual_jacobian" in source
    assert "TRAJ_TRACIN_CKPT_SHARD_COUNT=2" in source
    assert '"train_jacobian_norms"' in source
    assert '"train_features_v_l2_normalized"' in source


def test_pipeline_materializes_eight_scores():
    launcher = PIPELINE.read_text()
    scorer = SCORER.read_text()
    assert "trajectory_next_checkpoint_noise_mse" in launcher
    assert "trajectory_predicted_noise_probe" in launcher
    assert "expected_residual_jacobian_fnorm_original_f" in launcher
    assert "expected_residual_jacobian_v_l2_original_f" in launcher
    assert "expected_residual_jacobian_fnorm_predicted_noise" in launcher
    assert "expected_residual_jacobian_v_l2_predicted_noise" in launcher
    assert "original_terms != 490" in scorer
    assert "predicted_terms != 500" in scorer
    assert 'for query_variant in ("raw", "query_l2")' in scorer
    assert 'component = "score" if query_variant == "raw" else "score_query_normalized"' in scorer
    assert "2 train normalizations x 2 query targets x 2 query normalizations" in scorer
    assert "320 query/score/target evaluations" in launcher

    lds = (ROOT / "3dshapes" / "script" / "run_traj_tracin_lds_cached.py").read_text()
    assert "EXPECTED_RESIDUAL_JACOBIAN_VARIANTS" in lds
    assert '("raw", "score")' in lds
    assert '("query_l2", "score_query_normalized")' in lds
