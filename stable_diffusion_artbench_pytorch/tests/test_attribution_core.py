import json
from pathlib import Path

import numpy as np
import torch

from sd_artbench.attribution.autograd import batched_loss_gradients
from sd_artbench.attribution.launch_shards import save_checkpoint_sum
from sd_artbench.attribution.merge_shards import merge_shards
from sd_artbench.attribution.optimizer import (
    adamw_full_direction,
    adamw_full_direction_batch,
)
from sd_artbench.attribution.projection import CountSketchProjector
from sd_artbench.attribution.scoring import (
    endpoint_pollute_reductions,
    score_variants,
)
from sd_artbench.trajectory import (
    ddim_stride_timesteps,
    load_query_trajectory,
    save_query_trajectory,
)


def test_ddim50_uses_exact_twenty_index_stride():
    timesteps = ddim_stride_timesteps(50)
    assert timesteps.tolist() == list(range(999, -1, -20))


def test_query_trajectory_round_trip(tmp_path: Path):
    path = tmp_path / "query.trajectory.npz"
    latents = np.arange(3 * 4 * 2 * 2, dtype=np.float32).reshape(3, 4, 2, 2)
    predictions = -latents
    save_query_trajectory(
        path,
        latents=latents,
        predicted_noise=predictions,
        timesteps=np.asarray([999, 979, 959]),
    )
    loaded = load_query_trajectory(path, expected_steps=3)
    assert loaded.latents.dtype == np.float16
    assert np.array_equal(loaded.latents, latents.astype(np.float16))
    assert np.array_equal(loaded.predicted_noise, predictions.astype(np.float16))
    assert loaded.timesteps.tolist() == [999, 979, 959]


def test_countsketch_is_deterministic_and_normalizes_in_full_space():
    tensors = [torch.tensor([3.0, 4.0]), torch.tensor([0.0])]
    projector = CountSketchProjector(proj_dim=17, seed=42, chunk_size=2)
    first, norm = projector.project(tensors, full_l2_normalize=True)
    second, second_norm = projector.project(tensors, full_l2_normalize=True)
    assert norm == 5.0
    assert second_norm == 5.0
    assert torch.equal(first, second)


def test_batched_countsketch_matches_individual_projection():
    tensors = [torch.tensor([[3.0, 4.0], [5.0, 12.0]])]
    projector = CountSketchProjector(proj_dim=17, seed=42, chunk_size=1)
    batched, norms = projector.project_batch(tensors, full_l2_normalize=True)
    individual = torch.stack(
        [
            projector.project([tensors[0][row]], full_l2_normalize=True)[0]
            for row in range(2)
        ]
    )
    assert torch.allclose(norms, torch.tensor([5.0, 13.0]))
    assert torch.equal(batched, individual)


def test_batched_vjp_matches_individual_gradients():
    parameter = torch.nn.Parameter(torch.tensor([1.5, -0.5]))
    inputs = torch.tensor([[1.0, 2.0], [-3.0, 4.0], [2.0, -1.0]])
    losses = (inputs * parameter).square().sum(dim=1)
    chunks = list(
        batched_loss_gradients(losses, [parameter], [0, 1, 2], chunk_size=2)
    )
    actual = torch.cat([chunk[1][0] for chunk in chunks], dim=0)
    expected = 2.0 * inputs.square() * parameter.detach()
    assert torch.allclose(actual, expected)


def test_adamw_full_direction_matches_one_real_optimizer_step():
    parameter = torch.nn.Parameter(torch.tensor([1.5, -0.5]))
    optimizer = torch.optim.AdamW(
        [parameter], lr=0.03, betas=(0.8, 0.9), eps=1e-6, weight_decay=0.02
    )
    warmup_gradient = torch.tensor([0.2, -0.4])
    parameter.grad = warmup_gradient.clone()
    optimizer.step()
    optimizer.zero_grad()

    candidate_gradient = torch.tensor([-0.7, 0.1])
    before = parameter.detach().clone()
    directions, learning_rate = adamw_full_direction(
        [parameter], [candidate_gradient], optimizer
    )
    parameter.grad = candidate_gradient.clone()
    optimizer.step()
    expected = (parameter.detach() - before) / learning_rate
    assert torch.allclose(directions[0], expected, atol=2e-6, rtol=2e-5)


def test_adamw_full_direction_includes_training_gradient_clip():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.01, weight_decay=0.03)
    candidate_gradient = torch.tensor([30.0, 40.0])
    before = parameter.detach().clone()
    directions, learning_rate = adamw_full_direction(
        [parameter], [candidate_gradient], optimizer, max_grad_norm=1.0
    )
    parameter.grad = candidate_gradient.clone()
    torch.nn.utils.clip_grad_norm_([parameter], 1.0)
    optimizer.step()
    expected = (parameter.detach() - before) / learning_rate
    assert torch.allclose(directions[0], expected, atol=2e-6, rtol=2e-5)


def test_batched_adamw_matches_individual_directions():
    parameter = torch.nn.Parameter(torch.tensor([1.5, -0.5]))
    optimizer = torch.optim.AdamW([parameter], lr=0.03, weight_decay=0.02)
    parameter.grad = torch.tensor([0.2, -0.4])
    optimizer.step()
    optimizer.zero_grad()
    candidates = torch.tensor([[-0.7, 0.1], [3.0, 4.0]])
    batched, learning_rate = adamw_full_direction_batch(
        [parameter], [candidates], optimizer, max_grad_norm=1.0
    )
    individual = torch.stack(
        [
            adamw_full_direction(
                [parameter], [candidate], optimizer, max_grad_norm=1.0
            )[0][0]
            for candidate in candidates
        ]
    )
    assert learning_rate == 0.03
    assert torch.allclose(batched[0], individual)


def test_four_score_variants_have_expected_cosines():
    query = torch.tensor([[3.0, 0.0]])
    train = torch.tensor([[2.0, 0.0], [0.0, 4.0]])
    scores = score_variants(query, train)
    assert scores.shape == (4, 1, 2)
    assert torch.equal(scores[0], torch.tensor([[6.0, 0.0]]))
    assert torch.equal(scores[1], torch.tensor([[2.0, 0.0]]))
    assert torch.equal(scores[2], torch.tensor([[3.0, 0.0]]))
    assert torch.allclose(scores[3], torch.tensor([[1.0, 0.0]]))


def test_endpoint_pollute_reductions_do_not_confuse_square_orders():
    components = torch.tensor([[[[[1.0]]], [[[2.0]]]], [[[[3.0]]], [[[4.0]]]]])
    reduced = endpoint_pollute_reductions(components)
    assert reduced["linear"].item() == 10.0
    assert reduced["termwise_squared"].item() == 30.0
    assert reduced["timestamp_sum_squared"].item() == 52.0
    assert reduced["total_sum_squared"].item() == 100.0


def _write_shard(path: Path, start: int, values: list[float]) -> None:
    width = len(values)
    manifest = {
        "method": "retrac",
        "parameter_source": "raw_lora",
        "checkpoint_indices": [0],
        "timesteps": [0],
        "proj_dim": 8,
        "train_positions": [start, start + width],
    }
    np.savez_compressed(
        path,
        scores=np.asarray(values, dtype=np.float64).reshape(1, 1, width),
        train_metadata_indices=np.arange(start + 100, start + 100 + width),
        transform_names=np.asarray(["raw_gradient"]),
        manifest_json=np.asarray(json.dumps(manifest)),
    )


def test_merge_shards_sorts_by_training_interval(tmp_path: Path):
    later = tmp_path / "later.npz"
    earlier = tmp_path / "earlier.npz"
    _write_shard(later, 2, [3.0, 4.0])
    _write_shard(earlier, 0, [1.0, 2.0])
    scores, indices, manifest, _, variants = merge_shards([later, earlier])
    assert scores.tolist() == [[[1.0, 2.0, 3.0, 4.0]]]
    assert indices.tolist() == [100, 101, 102, 103]
    assert manifest["train_positions"] == [0, 4]
    assert variants is None


def test_checkpoint_shards_are_summed(tmp_path: Path):
    first = tmp_path / "checkpoint-first.npz"
    second = tmp_path / "checkpoint-second.npz"
    output = tmp_path / "merged.npz"
    _write_shard(first, 0, [1.0, 2.0])
    _write_shard(second, 0, [3.0, 4.0])
    with np.load(second, allow_pickle=False) as payload:
        fields = {key: np.asarray(payload[key]) for key in payload.files}
    manifest = json.loads(str(fields["manifest_json"].item()))
    manifest["checkpoint_indices"] = [1]
    fields["manifest_json"] = np.asarray(json.dumps(manifest))
    np.savez_compressed(second, **fields)
    save_checkpoint_sum([first, second], output)
    with np.load(output, allow_pickle=False) as payload:
        assert payload["scores"].tolist() == [[[4.0, 6.0]]]
        merged = json.loads(str(payload["manifest_json"].item()))
    assert merged["checkpoint_indices"] == [0, 1]
