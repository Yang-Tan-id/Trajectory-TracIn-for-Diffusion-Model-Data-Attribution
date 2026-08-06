from __future__ import annotations

"""Build checkpoint-by-snapshot predicted-noise-change weights for Traj-TracIn.

For a saved query trajectory (x_s, t_s), this computes

    delta[c, s] = mean((eps_{theta_{c+1}}(x_s, t_s) - eps_{theta_c}(x_s, t_s)) ** 2)
    ref_mse[c, s] = mean((eps_{theta_c}(x_s, t_s) - eps_{theta_ref}(x_s, t_s)) ** 2)
    cosine[c, s] = cos(eps_{theta_{c+1}} - eps_{theta_c}, eps_{theta_ref} - eps_{theta_c})
    progress[c, s] = ref_mse[c, s] - ref_mse[c + 1, s]

and stores both the raw deltas and simple normalized weight tables. The output
is intentionally small: 49x100/50x100 floats for the usual CIFAR2 setup.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from common.config_loader import load_config, require_attr
from common.paths import add_legacy_jax_to_path, chdir_legacy_jax_root, path_tag


def normalize_per_timestamp(delta: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.nanmean(delta, axis=0, keepdims=True)
    weights = delta / np.maximum(denom, eps)
    return np.where(np.isfinite(weights), weights, 0.0)


def normalize_global(delta: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    weights = delta / max(float(np.nanmean(delta)), eps)
    return np.where(np.isfinite(weights), weights, 0.0)


def pad_last_checkpoint_transition(delta_49: np.ndarray, num_ckpts: int) -> np.ndarray:
    if delta_49.shape[0] == num_ckpts:
        return delta_49
    if delta_49.shape[0] != num_ckpts - 1:
        raise ValueError(f"Expected {num_ckpts - 1} or {num_ckpts} transition rows, got {delta_49.shape[0]}")
    return np.concatenate([delta_49, delta_49[-1:, :]], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate predicted-noise-change weights for full-dim term scores.")
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
    config_values.setdefault("out_dir", "./predicted_noise_change_weights")

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
        schedule_to_device,
        tree_to_device,
    )

    cfg = TrajAttributionConfig(**config_values)
    ckpts = list_checkpoints_sorted(cfg.checkpoint_dir)
    if cfg.checkpoint_limit is not None and int(cfg.checkpoint_limit) > 0:
        idx = np.linspace(0, len(ckpts) - 1, min(int(cfg.checkpoint_limit), len(ckpts)), dtype=np.int32)
        ckpts = [ckpts[i] for i in idx]
    if len(ckpts) < 2:
        raise ValueError(f"Need at least two checkpoints to compute checkpoint-to-checkpoint deltas, got {len(ckpts)}")

    reference_ckpt = cfg.reference_ckpt or ckpts[-1]
    apply_checkpoint_config(cfg, reference_ckpt)

    print("=" * 90)
    print("Predicted-noise-change weight generation")
    print(f"dataset             : {dataset_name}")
    print(f"experiment          : {experiment_tag}")
    print(f"mode                : {args.mode}")
    print(f"query               : {cfg.query}")
    print(f"initial_seed        : {cfg.attribution_sample_seed}")
    print(f"checkpoint_dir      : {cfg.checkpoint_dir}")
    print(f"reference_ckpt      : {reference_ckpt}")
    print(f"checkpoints         : {len(ckpts)}")
    print(f"sample_dir          : {cfg.attribution_sample_dir}")
    print(f"out                 : {args.out}")
    print("=" * 90)

    adapter = get_adapter(cfg)
    device = adapter.choose_device(cfg.prefer_device)
    print(f"[setup] using device: {device}")

    ds = adapter.iter_dataset(cfg)
    model = adapter.build_model(cfg)
    state_template = adapter.build_state_template(cfg, model, device)
    schedule = schedule_to_device(make_diffusion_schedule(cfg.timesteps, cfg.beta_start, cfg.beta_end), device)
    query_cond = array_to_device(adapter.make_query_cond(ds, cfg.query, cfg), device)
    print(f"[setup] query_cond shape={tuple(query_cond.shape)}")

    xt_refs_raw, t_seq, pos_seq, sample_meta = load_attribution_trajectory(cfg)
    xt_refs = [array_to_device(x, device) for x in xt_refs_raw]
    chunk_size = max(1, int(cfg.snapshot_chunk_size))
    print(f"[setup] loaded trajectory snapshots={len(t_seq)} | chunk_size={chunk_size}")

    def eps_chunk(params, xt_chunk, t_chunk, cond):
        t_vec = t_chunk.astype(jnp.int32)
        return jax.vmap(lambda x_one, t_one: adapter.eps_apply(model, params, x_one, t_one[None], cond))(
            xt_chunk,
            t_vec,
        )

    eps_chunk_jit = jax.jit(eps_chunk)

    print(f"[reference] loading {reference_ckpt}")
    reference_state, _reference_payload = adapter.restore_state(reference_ckpt, state_template)
    reference_params = tree_to_device(reference_state.ema_params, device)
    print(f"[device-check] reference_params={first_leaf_device_str(reference_params)}")
    reference_eps_chunks = []
    for chunk_start in range(0, len(t_seq), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(t_seq))
        xt_chunk = array_to_device(jnp.stack([xt_refs[i] for i in range(chunk_start, chunk_end)], axis=0), device)
        t_chunk = array_to_device(jnp.asarray([int(t_seq[i]) for i in range(chunk_start, chunk_end)], dtype=jnp.int32), device)
        eps = eps_chunk_jit(reference_params, xt_chunk, t_chunk, query_cond)
        eps.block_until_ready()
        reference_eps_chunks.append(np.asarray(jax.device_get(eps), dtype=np.float32))
    print("[reference] eps chunks ready")

    previous_eps_chunks: list[np.ndarray] | None = None
    previous_ref_mse_row: np.ndarray | None = None
    delta_rows: list[np.ndarray] = []
    cosine_rows: list[np.ndarray] = []
    progress_rows: list[np.ndarray] = []
    ref_mse_rows: list[np.ndarray] = []
    started = time.time()

    for ckpt_i, ckpt_path in enumerate(ckpts):
        ckpt_start = time.time()
        print(f"[ckpt] {ckpt_i + 1}/{len(ckpts)} loading {os.path.basename(ckpt_path)}")
        state, _payload = adapter.restore_state(ckpt_path, state_template)
        params = tree_to_device(state.ema_params, device)
        print(f"[device-check] params={first_leaf_device_str(params)}")

        current_eps_chunks = []
        for chunk_start in range(0, len(t_seq), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(t_seq))
            xt_chunk = array_to_device(jnp.stack([xt_refs[i] for i in range(chunk_start, chunk_end)], axis=0), device)
            t_chunk = array_to_device(jnp.asarray([int(t_seq[i]) for i in range(chunk_start, chunk_end)], dtype=jnp.int32), device)
            eps = eps_chunk_jit(params, xt_chunk, t_chunk, query_cond)
            eps.block_until_ready()
            current_eps_chunks.append(np.asarray(jax.device_get(eps), dtype=np.float32))

        ref_mses = []
        for ref, curr in zip(reference_eps_chunks, current_eps_chunks):
            mse = np.mean((curr.astype(np.float64) - ref.astype(np.float64)) ** 2, axis=tuple(range(1, curr.ndim)))
            ref_mses.append(mse.astype(np.float64))
        ref_mse_row = np.concatenate(ref_mses, axis=0)
        ref_mse_rows.append(ref_mse_row)

        if previous_eps_chunks is not None:
            deltas = []
            cosines = []
            for chunk_id, (prev, curr) in enumerate(zip(previous_eps_chunks, current_eps_chunks)):
                move = curr.astype(np.float64) - prev.astype(np.float64)
                to_ref = reference_eps_chunks[chunk_id].astype(np.float64) - prev.astype(np.float64)
                reduce_axes = tuple(range(1, curr.ndim))
                mse = np.mean(move ** 2, axis=reduce_axes)
                dot = np.sum(move * to_ref, axis=reduce_axes)
                move_norm = np.sqrt(np.sum(move ** 2, axis=reduce_axes))
                ref_norm = np.sqrt(np.sum(to_ref ** 2, axis=reduce_axes))
                cosine = dot / np.maximum(move_norm * ref_norm, 1e-30)
                deltas.append(mse.astype(np.float64))
                cosines.append(cosine.astype(np.float64))
            delta_row = np.concatenate(deltas, axis=0)
            cosine_row = np.concatenate(cosines, axis=0)
            assert previous_ref_mse_row is not None
            progress_row = previous_ref_mse_row - ref_mse_row
            delta_rows.append(delta_row)
            cosine_rows.append(cosine_row)
            progress_rows.append(progress_row)
            print(
                f"[delta] transition {ckpt_i}/{len(ckpts) - 1} | "
                f"mean={float(np.mean(delta_row)):.6g} | "
                f"min={float(np.min(delta_row)):.6g} | max={float(np.max(delta_row)):.6g}"
            )
            print(
                f"[direction] transition {ckpt_i}/{len(ckpts) - 1} | "
                f"cos_mean={float(np.mean(cosine_row)):.6g} | "
                f"cos_pos_frac={float(np.mean(cosine_row > 0.0)):.3f} | "
                f"progress_pos_frac={float(np.mean(progress_row > 0.0)):.3f}"
            )
        print(
            f"[ref-mse] ckpt {ckpt_i + 1}/{len(ckpts)} | "
            f"mean={float(np.mean(ref_mse_row)):.6g} | "
            f"min={float(np.min(ref_mse_row)):.6g} | max={float(np.max(ref_mse_row)):.6g}"
        )

        previous_eps_chunks = current_eps_chunks
        previous_ref_mse_row = ref_mse_row
        elapsed = time.time() - started
        print(
            f"[ckpt] {ckpt_i + 1}/{len(ckpts)} done | "
            f"ckpt_elapsed={format_seconds(time.time() - ckpt_start)} | "
            f"total_elapsed={format_seconds(elapsed)}"
        )

    delta_transition = np.stack(delta_rows, axis=0).astype(np.float64)
    cosine_transition = np.stack(cosine_rows, axis=0).astype(np.float64)
    progress_transition = np.stack(progress_rows, axis=0).astype(np.float64)
    ref_mse_by_ckpt_snapshot = np.stack(ref_mse_rows, axis=0).astype(np.float64)
    delta_by_ckpt_snapshot = pad_last_checkpoint_transition(delta_transition, len(ckpts))
    cosine_by_ckpt_snapshot = pad_last_checkpoint_transition(cosine_transition, len(ckpts))
    progress_by_ckpt_snapshot = pad_last_checkpoint_transition(progress_transition, len(ckpts))
    weight_per_timestamp = normalize_per_timestamp(delta_by_ckpt_snapshot)
    weight_global = normalize_global(delta_by_ckpt_snapshot)

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        eps_delta_mse_by_transition=delta_transition.astype(np.float32),
        eps_to_ref_cosine_by_transition=cosine_transition.astype(np.float32),
        eps_to_ref_progress_by_transition=progress_transition.astype(np.float32),
        eps_ref_mse_by_ckpt_snapshot=ref_mse_by_ckpt_snapshot.astype(np.float32),
        delta_by_ckpt_snapshot=delta_by_ckpt_snapshot.astype(np.float32),
        eps_to_ref_cosine_by_ckpt_snapshot=cosine_by_ckpt_snapshot.astype(np.float32),
        eps_to_ref_progress_by_ckpt_snapshot=progress_by_ckpt_snapshot.astype(np.float32),
        change_weight_by_ckpt_snapshot=weight_per_timestamp.astype(np.float32),
        change_weight_global_linear=weight_global.astype(np.float32),
        ckpt_paths=np.asarray(ckpts),
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
        f"delta_mean={float(np.mean(delta_by_ckpt_snapshot)):.6g} | "
        f"ref_mse_start={float(np.mean(ref_mse_by_ckpt_snapshot[0])):.6g} | "
        f"ref_mse_end={float(np.mean(ref_mse_by_ckpt_snapshot[-1])):.6g} | "
        f"cos_mean={float(np.mean(cosine_transition)):.6g} | "
        f"progress_pos_frac={float(np.mean(progress_transition > 0.0)):.3f} | "
        f"per_ts_weight_mean={float(np.mean(weight_per_timestamp)):.6g} | "
        f"elapsed={format_seconds(time.time() - started)}"
    )


if __name__ == "__main__":
    main()
