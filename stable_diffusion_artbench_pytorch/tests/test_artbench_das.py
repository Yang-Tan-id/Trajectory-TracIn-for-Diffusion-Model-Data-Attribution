from pathlib import Path

import numpy as np
import torch

from sd_artbench.artbench_das import dual_term_scores


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src" / "sd_artbench" / "artbench_das.py"
LAUNCHER = ROOT / "run_artbench_das_50x1_4gpu.sh"


def test_artbench_das_contract():
    text = MODULE.read_text()
    assert 'select_timesteps(50)' in text
    assert 'eigenvectors.shape != (50, 5000, 5000)' in text
    assert 'default=32768' in text
    assert '"solver": "dual_eigendecomposition"' in text
    assert '"gradient_parameters": "lora_only"' in text
    assert '"residual": "latent_noise_prediction_projection"' in text
    assert '"sherman_morrison_denominator": True' in text
    assert '"score_contraction": "squared"' in text
    assert 'prediction = -(membership @ scores[slot])' in text


def test_launcher_uses_four_disjoint_ddim_shards_and_q0_q59():
    text = LAUNCHER.read_text()
    values = []
    for line in text.splitlines():
        stripped = line.strip().strip('"')
        if stripped and all(piece.isdigit() for piece in stripped.split(',')):
            values.extend(int(piece) for piece in stripped.split(','))
    assert sorted(values) == sorted(range(999, 18, -20))
    assert len(values) == len(set(values)) == 50
    assert 'seq -s, 0 59' in text
    assert 'artbench_das_50x1_full5000_dual32768_sm' in text
    assert 'DAS_PROJ_DIM:-32768' in text
    assert 'DAS_TRAIN_BATCH_SIZE:-2' in text
    assert 'DAS_RUN_LDS:-0' in text
    assert 'set DAS_RUN_LDS=1 after true-f is available' in text


def test_dual_solver_matches_primal_das_with_sherman_morrison():
    rng = np.random.default_rng(7)
    train = rng.normal(size=(7, 11)).astype(np.float32)
    query = rng.normal(size=(3, 11)).astype(np.float32)
    residual = rng.normal(size=7).astype(np.float32)
    damping = 0.7

    gram = train.T @ train + damping * np.eye(train.shape[1], dtype=np.float32)
    inverse = np.linalg.inv(gram)
    primal_prediction = train @ inverse @ query.T
    primal_denominator = 1.0 - np.sum((train @ inverse) * train, axis=1)
    expected = primal_prediction * residual[:, None] / primal_denominator[:, None]

    dual = train @ train.T
    eigenvalues, eigenvectors = np.linalg.eigh(dual)
    actual = dual_term_scores(
        eigenvalues.astype(np.float32), eigenvectors.astype(np.float32),
        (train @ query.T).astype(np.float32), residual, damping, torch.device("cpu"),
    ).numpy()
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
