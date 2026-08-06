from __future__ import annotations

"""Measure query-gradient alignment with actual checkpoint-to-checkpoint updates."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from common.config_loader import load_config, require_attr
from common.paths import add_legacy_jax_to_path, chdir_legacy_jax_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute <grad F(theta_c), theta_c - theta_{c+1}> over trajectory terms.")
    parser.add_argument("config", help="Dataset dataset_config.py")
    parser.add_argument("--mode", choices=["prompted", "unprompted"], default="prompted")
    parser.add_argument("--query", default=None, help="Prompt/query. Use 'horse,automobile' for combined prompted.")
    parser.add_argument("--initial-seed", type=int, default=None)
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--checkpoint-limit", type=int, default=None)
    parser.add_argument("--snapshot-chunk-size", type=int, default=None)
    parser.add_argument("--out", required=True, help="Output .npz path.")
    args = parser.parse_args()

    if args.query is not None:
        os.environ["QUERY"] = args.query
    if args.initial_seed is not None:
        os.environ["INITIAL_SEED"] = str(args.initial_seed)
        os.environ["SAMPLE_SEED"] = str(args.initial_seed)
    if args.train_seed is not None:
        os.environ["TRAIN_SEED"] = str(args.train_seed)
    if args.mode == "unprompted":
        os.environ["UNPROMPTED"] = "1"
        os.environ["ATTRIBUTION_SCORE_MODEL_MODE"] = "unprompted_solo"
        os.environ["ATTRIBUTION_SAMPLE_MODEL_MODE"] = "unprompted_solo"
        os.environ["SAMPLE_MODEL_MODE"] = "unprompted_solo"
        os.environ["QUERY"] = "unconditional"
    else:
        os.environ.setdefault("ATTRIBUTION_SCORE_MODEL_MODE", "prompted_solo")
        os.environ.setdefault("ATTRIBUTION_SAMPLE_MODEL_MODE", "prompted_solo")
        os.environ.setdefault("SAMPLE_MODEL_MODE", "prompted_solo")

    cfg_module = load_config(args.config)
    dataset_name = require_attr(cfg_module, "DATASET_NAME")
    experiment_tag = require_attr(cfg_module, "EXPERIMENT_TAG")
    if args.mode == "unprompted":
        config_values = dict(require_attr(cfg_module, "unprompted_attribution_config")("traj_tracin"))
    else:
        config_values = dict(require_attr(cfg_module, "attribution_config")("traj_tracin"))

    if args.query is not None and args.mode == "prompted":
        config_values["query"] = args.query
    if args.initial_seed is not None:
        config_values["attribution_sample_seed"] = int(args.initial_seed)
    if args.checkpoint_limit is not None:
        config_values["checkpoint_limit"] = int(args.checkpoint_limit)
    if args.snapshot_chunk_size is not None:
        config_values["snapshot_chunk_size"] = int(args.snapshot_chunk_size)
    config_values.setdefault("out_dir", "./query_update_alignment")

    add_legacy_jax_to_path()
    chdir_legacy_jax_root()

    import jax
    import jax.numpy as jnp
    from traj_tracin.algorithm import (
        TrajAttributionConfig,
        apply_checkpoint_config,
        array_to_device,
        first_leaf_device_str,
        format_seconds,
        get_adapter,
        list_checkpoints_sorted,
        load_attribution_trajectory,
        make_diffusion_schedule,
        make_query_grad_chunk_fn,
        schedule_to_device,
        tree_l2_norm,
        tree_to_device,
        tree_vdot,
    )

    cfg = TrajAttributionConfig(**config_values)
    ckpts = list_checkpoints_sorted(cfg.checkpoint_dir)
    if cfg.checkpoint_limit is not None and int(cfg.checkpoint_limit) > 0:
        idx = np.linspace(0, len(ckpts) - 1, min(int(cfg.checkpoint_limit), len(ckpts)), dtype=np.int32)
        ckpts = [ckpts[i] for i in idx]
    if len(ckpts) < 2:
        raise ValueError(f"Need at least two checkpoints, got {len(ckpts)}")

    reference_ckpt = cfg.reference_ckpt or ckpts[-1]
    apply_checkpoint_config(cfg, reference_ckpt)

    print("=" * 90)
    print("Query-gradient alignment with actual checkpoint updates")
    print(f"dataset        : {dataset_name}")
    print(f"experiment     : {experiment_tag}")
    print(f"mode           : {args.mode}")
    print(f"query          : {cfg.query}")
    print(f"initial_seed   : {cfg.attribution_sample_seed}")
    print(f"query_objective: {cfg.query_objective}")
    print(f"checkpoints    : {len(ckpts)}")
    print(f"reference_ckpt : {reference_ckpt}")
    print(f"out            : {args.out}")
    print("=" * 90)

    adapter = get_adapter(cfg)
    device = adapter.choose_device(cfg.prefer_device)
    print(f"[setup] using device: {device}")
    ds = adapter.iter_dataset(cfg)
    model = adapter.build_model(cfg)
    state_template = adapter.build_state_template(cfg, model, device)
    _schedule = schedule_to_device(make_diffusion_schedule(cfg.timesteps, cfg.beta_start, cfg.beta_end), device)
    query_cond = array_to_device(adapter.make_query_cond(ds, cfg.query, cfg), device)

    xt_refs_raw, t_seq, pos_seq, sample_meta = load_attribution_trajectory(cfg)
    xt_refs = [array_to_device(x, device) for x in xt_refs_raw]
    chunk_size = max(1, int(cfg.snapshot_chunk_size))

    print(f"[reference] loading {reference_ckpt}")
    reference_state, _reference_payload = adapter.restore_state(reference_ckpt, state_template)
    reference_params = tree_to_device(reference_state.ema_params, device)
    print(f"[device-check] reference_params={first_leaf_device_str(reference_params)}")

    query_grad_chunk_fn = make_query_grad_chunk_fn(adapter, model, cfg.query_objective)

    def tree_sub(a, b):
        return jax.tree_util.tree_map(lambda x, y: x - y, a, b)

    def slice_tree(tree, i: int):
        return jax.tree_util.tree_map(lambda x: x[i], tree)

    alignment_rows: list[np.ndarray] = []
    cosine_rows: list[np.ndarray] = []
    query_norm_rows: list[np.ndarray] = []
    update_norms: list[float] = []
    started = time.time()

    print("[setup] loading checkpoint 1")
    current_state, _payload = adapter.restore_state(ckpts[0], state_template)
    current_params = tree_to_device(current_state.ema_params, device)

    for ckpt_i in range(len(ckpts) - 1):
        ckpt_start = time.time()
        print(f"[transition] {ckpt_i}/{len(ckpts) - 2} loading next {os.path.basename(ckpts[ckpt_i + 1])}")
        next_state, _payload = adapter.restore_state(ckpts[ckpt_i + 1], state_template)
        next_params = tree_to_device(next_state.ema_params, device)
        descent_update = tree_sub(current_params, next_params)
        update_norm = float(jax.device_get(tree_l2_norm(descent_update)))
        update_norms.append(update_norm)

        alignments = []
        cosines = []
        query_norms = []
        for chunk_start in range(0, len(t_seq), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(t_seq))
            xt_chunk = array_to_device(jnp.stack([xt_refs[i] for i in range(chunk_start, chunk_end)], axis=0), device)
            t_chunk = array_to_device(jnp.asarray([int(t_seq[i]) for i in range(chunk_start, chunk_end)], dtype=jnp.int32), device)
            query_grads = query_grad_chunk_fn(current_params, reference_params, xt_chunk, t_chunk, query_cond)
            query_grads = tree_to_device(query_grads, device)
            for local_i in range(chunk_end - chunk_start):
                q_grad = slice_tree(query_grads, local_i)
                dot = tree_vdot(q_grad, descent_update)
                q_norm = tree_l2_norm(q_grad)
                dot_f = float(jax.device_get(dot))
                q_norm_f = float(jax.device_get(q_norm))
                alignments.append(dot_f)
                query_norms.append(q_norm_f)
                cosines.append(dot_f / max(q_norm_f * update_norm, 1e-30))

        alignment_row = np.asarray(alignments, dtype=np.float64)
        cosine_row = np.asarray(cosines, dtype=np.float64)
        query_norm_row = np.asarray(query_norms, dtype=np.float64)
        alignment_rows.append(alignment_row)
        cosine_rows.append(cosine_row)
        query_norm_rows.append(query_norm_row)
        print(
            f"[transition] {ckpt_i}->{ckpt_i + 1} | "
            f"alignment_mean={float(np.mean(alignment_row)):.6g} | "
            f"positive_frac={float(np.mean(alignment_row > 0.0)):.3f} | "
            f"cos_mean={float(np.mean(cosine_row)):.6g} | "
            f"update_norm={update_norm:.6g} | "
            f"elapsed={format_seconds(time.time() - ckpt_start)}"
        )
        current_params = next_params

    alignment = np.stack(alignment_rows, axis=0).astype(np.float64)
    cosine = np.stack(cosine_rows, axis=0).astype(np.float64)
    query_norm = np.stack(query_norm_rows, axis=0).astype(np.float64)

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        query_update_alignment_by_transition=alignment.astype(np.float32),
        query_update_cosine_by_transition=cosine.astype(np.float32),
        query_grad_norm_by_transition=query_norm.astype(np.float32),
        update_norm_by_transition=np.asarray(update_norms, dtype=np.float32),
        transition_from_ckpt_paths=np.asarray(ckpts[:-1]),
        transition_to_ckpt_paths=np.asarray(ckpts[1:]),
        timesteps=np.asarray(t_seq, dtype=np.int32),
        snapshot_positions=np.asarray(pos_seq, dtype=np.int32),
        query=np.asarray(str(cfg.query)),
        mode=np.asarray(str(args.mode)),
        initial_seed=np.asarray(int(cfg.attribution_sample_seed if cfg.attribution_sample_seed is not None else -1), dtype=np.int32),
        attribution_sample_dir=np.asarray(str(cfg.attribution_sample_dir)),
        sample_meta_json=np.asarray(json.dumps(sample_meta, sort_keys=True)),
    )
    print(f"[saved] {out_path.resolve()}")
    print(
        "[summary] "
        f"alignment_mean={float(np.mean(alignment)):.6g} | "
        f"alignment_positive_fraction={float(np.mean(alignment > 0.0)):.3f} | "
        f"cosine_mean={float(np.mean(cosine)):.6g} | "
        f"cosine_positive_fraction={float(np.mean(cosine > 0.0)):.3f} | "
        f"elapsed={format_seconds(time.time() - started)}"
    )


if __name__ == "__main__":
    main()
