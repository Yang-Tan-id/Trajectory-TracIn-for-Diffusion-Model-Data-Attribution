import json
from pathlib import Path

import numpy as np
import torch

from sd_artbench.attribution.merge_shards import merge_shards
from sd_artbench.attribution.optimizer import adamw_full_direction
from sd_artbench.attribution.projection import CountSketchProjector
from sd_artbench.attribution.scoring import (
    endpoint_pollute_reductions,
    score_variants,
)


def test_countsketch_is_deterministic_and_normalizes_in_full_space():
    tensors = [torch.tensor([3.0, 4.0]), torch.tensor([0.0])]
    projector = CountSketchProjector(proj_dim=17, seed=42, chunk_size=2)
    first, norm = projector.project(tensors, full_l2_normalize=True)
    second, second_norm = projector.project(tensors, full_l2_normalize=True)
    assert norm == 5.0
    assert second_norm == 5.0
    assert torch.equal(first, second)


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
