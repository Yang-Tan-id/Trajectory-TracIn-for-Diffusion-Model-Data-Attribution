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
H100_100X1 = (
    ROOT
    / "3dshapes"
    / "tacc"
    / "h100"
    / "run_traj_tracin_probe_aligned_v_l2_100x1_h100.sh"
)


def test_expected_residual_jacobian_is_not_expected_loss_gradient():
    residuals = np.asarray([1.0, 3.0])
    jacobians = np.asarray([[2.0, 0.0], [0.0, 4.0]])

    expected_loss_gradient = np.mean(residuals[:, None] * jacobians, axis=0)
    decomposed = residuals.mean() * jacobians.mean(axis=0)

    assert not np.allclose(expected_loss_gradient, decomposed)
    np.testing.assert_allclose(decomposed, np.asarray([2.0, 4.0]))


def test_algorithm_saves_single_probe_v_l2_feature():
    source = ALGORITHM.read_text()
    assert "TRAJ_TRACIN_TRAIN_DECOMPOSE_RESIDUAL_JACOBIAN" in source
    assert "TRAJ_TRACIN_JACOBIAN_NORM_PROBES" in source
    assert "mean_residual = jax.lax.stop_gradient" in source
    assert "def residual_contraction" not in source
    assert "projected_residual * unit_probe_gradient" in source
    assert '"mean_probe_projected_residual_times_unit_probe_gradient"' in source
    assert "cfg.seed + 91_337" not in source
    assert "base_probe_key = predicted_noise_probe_key(\n                            cfg.seed," in source


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
    assert '"train_features"' in source


def test_pipeline_materializes_four_scores():
    launcher = PIPELINE.read_text()
    scorer = SCORER.read_text()
    assert "trajectory_next_checkpoint_noise_mse" in launcher
    assert "trajectory_predicted_noise_probe" in launcher
    assert "expected_residual_jacobian_probe_aligned_v_l2_original_f" in launcher
    assert "expected_residual_jacobian_probe_aligned_v_l2_predicted_noise" in launcher
    assert "expected_original_terms = (args.num_checkpoints - 1) * args.num_snapshots" in scorer
    assert "expected_predicted_terms = args.num_checkpoints * args.num_snapshots" in scorer
    assert 'for query_variant in ("raw", "query_l2")' in scorer
    assert 'component = "score" if query_variant == "raw" else "score_query_normalized"' in scorer
    assert "1 train normalization x 2 query targets x 2 query normalizations" in scorer
    assert "160 query/score/target evaluations" in launcher

    lds = (ROOT / "3dshapes" / "script" / "run_traj_tracin_lds_cached.py").read_text()
    assert "EXPECTED_RESIDUAL_JACOBIAN_VARIANTS" in lds
    assert '("raw", "score")' in lds
    assert '("query_l2", "score_query_normalized")' in lds


def test_h100_100x1_uses_probe_aligned_v_l2_artifact():
    source = H100_100X1.read_text()
    assert "#SBATCH -N 4" in source
    assert "#SBATCH -n 16" in source
    assert "TRAJ_NUM_SNAPSHOTS=100" in source
    assert "TRAJ_TRAIN_MC_SAMPLES=1" in source
    assert "TRAJ_TRACIN_TRAIN_DECOMPOSE_RESIDUAL_JACOBIAN=1" in source
    assert "TRAJ_TRACIN_JACOBIAN_NORM_PROBES=1" in source
    assert "probe_aligned_100x1" in source
    assert "--num-checkpoints 50" in source
    assert "--num-snapshots 100" in source
    assert "trajectory_next_checkpoint_noise_mse" in source
    assert "trajectory_predicted_noise_probe" in source
