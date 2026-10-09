from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALGORITHM = ROOT / "legacy_jax" / "traj_tracin" / "algorithm.py"
QUERY = ROOT / "3dshapes" / "script" / "generate_endpoint_simple_loss_query100x1_bank.py"
MERGE = ROOT / "3dshapes" / "script" / "merge_fused_endpoint_tracin100x1_scores.py"
LAUNCHER = (
    ROOT
    / "3dshapes"
    / "tacc"
    / "rtx_small"
    / "run_endpoint_tracin_adamw_full_train100x1_query100x1_fused_rtx_small.sh"
)
LDS = ROOT / "3dshapes" / "script" / "run_traj_tracin_lds_cached.py"


def test_query_is_exact_endpoint_simple_loss_mean100x1():
    text = QUERY.read_text()
    assert "make_endpoint_query_fn" in text
    assert "(100,) + query[\"endpoint\"].shape" in text
    assert 'timesteps=np.asarray([-1]' in text
    assert 'query_timestamp_aggregation=np.asarray("mean_loss_then_gradient")' in text


def test_aggregate_gradient_is_transformed_before_projection_and_streamed():
    text = ALGORITHM.read_text()
    aggregate = text.index("def train_phi_one_aggregate_chunk")
    projection = text.index("return projector(grads)", aggregate)
    update = text.index("state.tx.update(", aggregate, projection)
    assert update < projection
    assert "fused checkpoint-level AdamW-full" in text
    assert 'fused_lookup.get((int(ckpt_i), -1))' in text
    assert "full_update = term_features + adamw_history_feature[None, :]" in text


def test_launcher_has_four_variants_and_never_persists_train_features():
    launcher = LAUNCHER.read_text()
    merge = MERGE.read_text()
    assert "#SBATCH -p rtx-small" in launcher
    assert "ibrun -n 1" in launcher
    assert "TRAJ_TRACIN_TRAIN_AGGREGATE_NUM_TIMESTEPS=100" in launcher
    assert "TRAJ_TRACIN_TRAIN_TIMESTAMP_CHUNK_SIZE=100" in launcher
    assert "TRAJ_TRACIN_TRAIN_OPTIMIZER_TRANSFORM=adamw_residual_update" in launcher
    assert "TRAJ_TRACIN_FUSED_STREAM_QUERY_ARTIFACTS" in launcher
    assert "persistent_train_gradient_artifact\": False" in merge
    for directory in (
        "score",
        "score_query_normalized",
        "score_train_l2_normalized",
        "score_query_train_l2_normalized",
    ):
        assert f'(\"{directory}\",' in merge


def test_lds_namespace_and_previous_endpoint_sign_are_wired():
    namespace = "endpoint_tracin_adamw_full_train100x1_query100x1_q0_99"
    assert namespace in LDS.read_text()
    launcher = LAUNCHER.read_text()
    assert f'score_namespace="{namespace}"' in launcher
    assert "--prediction-sign -1" in launcher
