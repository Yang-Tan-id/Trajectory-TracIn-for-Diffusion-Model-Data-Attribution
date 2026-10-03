import ast
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ALGORITHM = ROOT / "legacy_jax" / "traj_tracin" / "algorithm.py"
QUERY = ROOT / "3dshapes" / "script" / "direction20_query_common.py"
SCORER = ROOT / "3dshapes" / "script" / "score_direction20_timestamp_square_checkpoint_major.py"
LDS = ROOT / "3dshapes" / "script" / "run_traj_tracin_lds_cached.py"
RTX = ROOT / "3dshapes" / "tacc" / "rtx_small"
H100 = ROOT / "3dshapes" / "tacc" / "h100"


def test_direction_key_is_checkpoint_specific_and_shared_by_contract():
    source = ALGORITHM.read_text()
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "checkpoint_direction_shared_noise_key"
    )
    assert [arg.arg for arg in function.args.args] == [
        "train_seed",
        "checkpoint_index",
        "direction_index",
    ]
    function_source = ast.get_source_segment(source, function)
    assert "checkpoint_index" in function_source
    assert "direction_index" in function_source


def test_train_feature_is_gradient_of_fixed_direction_timestamp_mean():
    source = ALGORITHM.read_text()
    assert "def train_losses_at_t_sequence_fixed_noise_vectorized" in source
    assert "jnp.broadcast_to(one_noise, x0_rep.shape)" in source
    assert "per_timestamp.reshape((batch_size, timestamp_count)).mean(axis=1)" in source
    assert '"TRAJ_TRACIN_TRAIN_ALIGNED_DIRECTION_COUNT"' in source
    assert "checkpoint_direction_shared_noise_key" in source
    assert "state.tx.update(" in source
    assert '"optimizer_history_features"' in source


def test_query_and_train_use_the_same_direction_and_projection_rules():
    query = QUERY.read_text() + SCORER.read_text()
    assert "checkpoint_direction_shared_noise_key(" in query
    assert 'seed_parts=(args.train_seed, "traj_tracin_projection", ckpt_i)' in query
    assert "for ckpt_i in range(49):" in query
    assert "for query_start in range(0, len(queries), args.query_batch_size):" in query
    assert "target - eps" in query
    assert "unit_delta" in query


def test_timestampwise_square_reduction_is_checkpoint_sum_then_square():
    # Two checkpoints, two directions, three timestamps, two train points.
    checkpoint_dots = np.asarray(
        [
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
            ],
            [
                [[-1.0, 1.0], [2.0, -2.0], [0.5, 1.5]],
                [[1.0, -1.0], [3.0, 2.0], [-4.0, 0.0]],
            ],
        ]
    )
    implemented = np.square(checkpoint_dots.sum(axis=0)).sum(axis=(0, 1))
    explicit = np.zeros(2)
    for direction in range(2):
        for timestamp in range(3):
            explicit += checkpoint_dots[:, direction, timestamp, :].sum(axis=0) ** 2
    np.testing.assert_allclose(implemented, explicit)
    scorer = SCORER.read_text()
    assert "] += np.asarray(" in scorer
    assert "np.square(accumulators[query_i].astype(np.float64)).sum(axis=1)" in scorer


def test_four_variants_and_rtx_pipeline_are_wired():
    scorer = SCORER.read_text()
    assert '("score", "raw")' in scorer
    assert '("score_query_normalized", "query_l2_normalized")' in scorer
    assert '("score_train_l2_normalized", "train_l2_normalized")' in scorer
    assert '("score_query_train_l2_normalized", "query_train_l2_normalized")' in scorer
    assert "adamw_full_residual_plus_history" in scorer
    assert '"checkpoint_weighting": "inside_adamw_update_only"' in scorer
    assert '"query_gradient_artifact_written": False' in scorer

    score_launcher = (
        RTX / "run_direction20_mean100t_q0_99_scores_lds_rtx_small.sh"
    ).read_text()
    assert "#SBATCH -p rtx-small" in score_launcher
    assert "#SBATCH -n 2" in score_launcher

    train = (H100 / "run_direction20_mean100t_train_adamw_h100.sh").read_text()
    assert "#SBATCH -p h100" in train
    assert "#SBATCH -N 4" in train
    assert "#SBATCH -n 16" in train
    assert "TRAJ_TRACIN_TRAIN_ALIGNED_DIRECTION_COUNT=20" in train
    assert "TRAJ_TRACIN_TRAIN_OPTIMIZER_TRANSFORM=adamw_residual_update" in train
    assert "TRAJ_TRACIN_SKIP_STAGE_MERGE=1" in train
    assert "np.linspace(0, 999, 100" in train

    lds = LDS.read_text()
    assert (
        "recreate_adamw_full_direction20_mean100t_delta_l2normalized_"
        "timestamp_sum_squared_q0_99"
    ) in lds
