from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src" / "sd_artbench" / "artbench_das.py"
LAUNCHER = ROOT / "run_artbench_das_50x1_4gpu.sh"


def test_artbench_das_contract():
    text = MODULE.read_text()
    assert 'select_timesteps(50)' in text
    assert 'train.shape != (50, 5000, 4096)' in text
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
    assert 'artbench_das_50x1_full5000_sm' in text
    assert 'DAS_TRAIN_BATCH_SIZE:-2' in text
    assert 'DAS_RUN_LDS:-0' in text
    assert 'set DAS_RUN_LDS=1 after true-f is available' in text
