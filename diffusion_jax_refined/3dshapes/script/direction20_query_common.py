"""Shared query-side kernels for direction-aligned streaming scores."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp

from dataset_config import _prompt_tag


def direction_alignment_positions(count: int, ddim_steps: int = 1000) -> np.ndarray:
    if count <= 0 or count > ddim_steps:
        raise ValueError(f"timestamp count must be in [1,{ddim_steps}], got {count}")
    return np.linspace(0, ddim_steps - 1, count, dtype=np.int32)


def load_queries(
    query_file: Path,
    query_ids: list[int],
    sample_root: Path,
    checkpoint: Path,
    adapter: Any,
    dataset: Any,
    cfg: Any,
) -> list[dict[str, Any]]:
    records = json.loads(query_file.read_text())["queries"]
    queries = []
    for query_id in query_ids:
        record = records[query_id]
        prompt = str(record["prompt"])
        seed = int(record["initial_seed"])
        endpoint_path = (
            sample_root
            / "cifar"
            / f"prompt_{_prompt_tag(prompt)}"
            / f"model_prompted_solo__ckpt_{checkpoint.stem}"
            / f"seed_{seed:06d}"
            / "final_state.npy"
        )
        if not endpoint_path.is_file():
            raise FileNotFoundError(endpoint_path)
        endpoint = np.load(endpoint_path)
        if endpoint.ndim != 4 or endpoint.shape[0] < 1:
            raise ValueError(f"invalid endpoint {endpoint_path}: {endpoint.shape}")
        queries.append(
            {
                "id": query_id,
                "prompt": prompt,
                "seed": seed,
                "endpoint": np.asarray(endpoint[:1], dtype=np.float32),
                "cond": np.asarray(adapter.make_query_cond(dataset, prompt, cfg)),
            }
        )
    return queries


def make_projected_query_batch_fn(adapter, model, projector):
    def scalar_with_norm(params, target_params, xt, t_scalar, cond):
        t = jnp.full((xt.shape[0],), t_scalar, dtype=jnp.int32)
        eps = adapter.eps_apply(model, params, xt, t, cond)
        target = jax.lax.stop_gradient(
            adapter.eps_apply(model, target_params, xt, t, cond)
        )
        delta = jax.lax.stop_gradient(target - eps)
        norm = jnp.sqrt(jnp.sum(jnp.square(delta), dtype=jnp.float32))
        unit_delta = delta / jnp.maximum(norm, jnp.asarray(1e-12, jnp.float32))
        return jnp.mean(eps * unit_delta), norm

    value_grad = jax.value_and_grad(scalar_with_norm, has_aux=True)

    def one(params, target_params, xt, t_scalar, cond):
        (_value, norm), grad = value_grad(
            params, target_params, xt, t_scalar, cond
        )
        return projector(grad).astype(jnp.float32), norm.astype(jnp.float32)

    by_term = jax.vmap(one, in_axes=(None, None, 0, 0, None))
    by_query = jax.vmap(by_term, in_axes=(None, None, 0, None, 0))
    return jax.jit(by_query)
