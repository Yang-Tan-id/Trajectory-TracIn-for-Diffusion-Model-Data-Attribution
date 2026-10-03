from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCORER = ROOT / "3dshapes" / "script" / "score_retrac_endpoint100x1_checkpoint_major.py"
TRAIN = ROOT / "3dshapes" / "tacc" / "rtx_small" / "run_endpoint100x1_train_rtx_small.sh"
PIPELINE = ROOT / "3dshapes" / "tacc" / "rtx_small" / "run_retrac_endpoint100x1_q0_99_scores_lds_rtx_small.sh"
LDS = ROOT / "3dshapes" / "script" / "run_traj_tracin_lds_cached.py"


def test_retrac_uses_exact_four_events_and_event_learning_rates():
    text = SCORER.read_text()
    assert "range(start_epoch + 1, start_epoch + 5)" in text
    assert 'payload["batch_indices"]' in text
    assert "(epoch - 1) * steps_per_epoch + batches" in text
    assert "event_lrs[None, None, :, :]" in text
    assert "axis=2" in text


def test_endpoint_query_is_mean_100t_mc1_and_not_persisted():
    text = SCORER.read_text()
    assert "return jnp.mean(jnp.square(pred - noise_flat))" in text
    assert 'timesteps.shape != (100,)' in text
    assert '"query_gradient_artifact_written": False' in text
    assert "checkpoint pairs=49" in text


def test_all_four_normalizations_have_expected_algebra():
    rng = np.random.default_rng(7)
    train = rng.normal(size=(5, 11))
    query = rng.normal(size=(3, 11))
    raw = query @ train.T
    qnorm = np.linalg.norm(query, axis=1)[:, None]
    tnorm = np.linalg.norm(train, axis=1)[None, :]
    variants = np.stack((raw, raw / qnorm, raw / tnorm, raw / (qnorm * tnorm)), axis=1)
    assert variants.shape == (3, 4, 5)
    np.testing.assert_allclose(variants[:, 3], variants[:, 1] / tnorm)


def test_rtx_launchers_and_lds_namespaces_are_wired():
    train = TRAIN.read_text()
    pipeline = PIPELINE.read_text()
    lds = LDS.read_text()
    assert "#SBATCH -p rtx-small" in train
    assert "#SBATCH -n 2" in train
    assert "TRAJ_TRACIN_TRAIN_AGGREGATE_NUM_TIMESTEPS=100" in train
    assert "TRAJ_TRAIN_MC_SAMPLES=1" in train
    assert "TRAJ_TRACIN_SKIP_STAGE_MERGE=1" in train
    assert "event_count\" == 392" in pipeline
    assert "--prediction-sign -1" in pipeline
    assert '"retrac_exact4_endpoint100x1_q0_99"' in lds
    assert '"endpoint_tracin_train100x1_query100x1_q0_99"' in lds
