from __future__ import annotations

"""
DM___data_attribution_sampler.py

Bulk trajectory sampler for data attribution workflows.

What it saves
-------------
- model-space reverse trajectory snapshots (`trajectory_xt.npy`)
- corresponding diffusion timesteps (`trajectory_t.npy`)
- corresponding snapshot positions in the full reverse chain (`trajectory_pos.npy`)
- final model-space endpoint (`final_state.npy`)
- decoded final sample(s) for inspection (`decoded_final.npy` + PNGs)

This is designed so attribution jobs can reuse precomputed trajectories instead
of re-sampling each query.

Examples
--------
python DM___data_attribution_sampler.py --adapter cifar --code-file DM__training_CIFAR10_pixel.py --checkpoint models/cifar10_checkpoints_horse_automobile/seed_0_epoch_0200.ckpt --model-tag horse_automobile --prompt horse --seeds 0,1,2,3 --batch-size 1 --num-trajectory-steps 100 --outdir ./attribution_samples --prefer-device gpu

python DM___data_attribution_sampler.py --adapter artbench_latent --code-file DM__training_ARTBENCH_latent.py --checkpoint models/artbench_latent_dm_checkpoints256/seed_0_epoch_0100.ckpt --model-tag artbench256 --artbench-ae-checkpoint models/artbench_latent_autoencoder/ae_state.ckpt --prompt "baroque,surrealism" --seeds 0,1,2,3 --batch-size 1 --num-trajectory-steps 100 --outdir ./attribution_samples --prefer-device auto
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

import DM___sampler as sampler


def format_seconds(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}h {m:02d}m {s:05.2f}s"


def parse_seeds_arg(seeds_text: str) -> List[int]:
    tokens = [tok.strip() for tok in str(seeds_text).split(",") if tok.strip()]
    if not tokens:
        raise ValueError("--seeds must contain at least one integer.")
    return [int(tok) for tok in tokens]


def resolve_output_root(
    outdir: str,
    adapter_name: str,
    checkpoint: str,
    prompt: str,
    model_tag: Optional[str] = None,
) -> str:
    ckpt_stem = os.path.splitext(os.path.basename(checkpoint))[0]
    safe_prompt = sampler.sanitize_for_path(prompt)
    safe_model_tag = sampler.sanitize_for_path(model_tag) if model_tag else None
    ckpt_dir = f"model_{safe_model_tag}__ckpt_{ckpt_stem}" if safe_model_tag else f"ckpt_{ckpt_stem}"
    return os.path.join(
        outdir,
        adapter_name,
        f"prompt_{safe_prompt}",
        ckpt_dir,
    )


def sample_model_space_trajectory(
    adapter: sampler.ModelAdapter,
    seed: int,
    prompt: str,
    batch_size: int,
    save_timesteps: Sequence[int],
) -> Tuple[np.ndarray, Dict[int, np.ndarray], np.ndarray]:
    rng = jax.random.PRNGKey(seed)
    cond = adapter.make_condition(prompt=prompt, batch_size=batch_size)
    shape = adapter.sample_shape(batch_size)

    betas = adapter.schedule.betas
    alphas = adapter.schedule.alphas
    alphas_cumprod = adapter.schedule.alphas_cumprod
    cond_y = cond if getattr(adapter.cfg, "class_cond", True) else None
    t_seq = jnp.arange(adapter.cfg.timesteps - 1, -1, -1, dtype=jnp.int32)

    @jax.jit
    def _sample_scan_loop(init_rng: jax.Array):
        init_x = jax.random.normal(init_rng, shape)

        def body_fn(carry, i):
            x_t, loop_rng = carry
            t = jnp.full((shape[0],), i, dtype=jnp.int32)
            pred = adapter.model.apply(
                {"params": adapter.state.ema_params},
                x_t,
                t,
                cond_y,
                train=False,
            )

            x0_pred = pred if adapter.cfg.predict_x0 else adapter.predict_x0_from_eps(x_t, t, pred)
            eps = pred if not adapter.cfg.predict_x0 else (
                x_t - jnp.sqrt(alphas_cumprod[i]) * x0_pred
            ) / jnp.sqrt(1.0 - alphas_cumprod[i])

            alpha_t = alphas[i]
            abar_t = alphas_cumprod[i]
            beta_t = betas[i]
            coef1 = 1.0 / jnp.sqrt(alpha_t)
            coef2 = beta_t / jnp.sqrt(1.0 - abar_t)
            mean = coef1 * (x_t - coef2 * eps)

            loop_rng, step_rng = jax.random.split(loop_rng)
            noise = jax.random.normal(step_rng, shape)
            next_x = jax.lax.cond(
                i > 0,
                lambda _: mean + jnp.sqrt(beta_t) * noise,
                lambda _: mean,
                operand=None,
            )
            return (next_x, loop_rng), x_t

        (final_x, _), xt_seq = jax.lax.scan(body_fn, (init_x, init_rng), t_seq)
        return final_x, xt_seq

    final_x, xt_seq = _sample_scan_loop(rng)
    final_x_np = np.asarray(final_x)
    xt_seq_np = np.asarray(xt_seq)

    saved = {}
    for timestep in save_timesteps:
        if timestep < 0 or timestep >= adapter.cfg.timesteps:
            continue
        seq_idx = adapter.cfg.timesteps - 1 - int(timestep)
        saved[int(timestep)] = np.asarray(xt_seq_np[seq_idx])

    if 0 not in saved:
        saved[0] = np.asarray(final_x_np)

    decoded_final = np.asarray(adapter.decode_samples(final_x_np))
    return final_x_np, saved, decoded_final


def save_seed_outputs(
    seed_dir: str,
    adapter: sampler.ModelAdapter,
    seed: int,
    prompt: str,
    model_tag: Optional[str],
    ordered_timesteps: Sequence[int],
    save_timesteps: Sequence[int],
    final_state: np.ndarray,
    saved_states: Dict[int, np.ndarray],
    decoded_final: np.ndarray,
    upscale: int,
    max_png_side: int,
):
    os.makedirs(seed_dir, exist_ok=True)

    traj_stack = np.stack([saved_states[t] for t in ordered_timesteps], axis=0)
    t_arr = np.asarray(list(ordered_timesteps), dtype=np.int32)
    pos_arr = np.asarray(
        [adapter.cfg.timesteps - 1 - int(timestep) for timestep in ordered_timesteps],
        dtype=np.int32,
    )

    np.save(os.path.join(seed_dir, "trajectory_xt.npy"), traj_stack)
    np.save(os.path.join(seed_dir, "trajectory_t.npy"), t_arr)
    np.save(os.path.join(seed_dir, "trajectory_pos.npy"), pos_arr)
    np.save(os.path.join(seed_dir, "final_state.npy"), final_state)
    np.save(os.path.join(seed_dir, "decoded_final.npy"), decoded_final)

    for sample_idx, img in enumerate(decoded_final):
        sampler.save_image_nhwc(
            img,
            os.path.join(seed_dir, f"decoded_final_{sample_idx:03d}.png"),
            upscale=max(1, int(upscale)),
            max_side=int(max_png_side),
        )

    info = {
        "seed": int(seed),
        "prompt": prompt,
        "model_tag": model_tag,
        "batch_size": int(decoded_final.shape[0]),
        "save_timesteps": list(int(x) for x in ordered_timesteps),
        "save_positions": pos_arr.tolist(),
        "trajectory_xt_shape": list(traj_stack.shape),
        "final_state_shape": list(final_state.shape),
        "decoded_final_shape": list(decoded_final.shape),
    }
    with open(os.path.join(seed_dir, "seed_info.json"), "w") as f:
        json.dump(info, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Bulk sampler for data attribution trajectories.")
    parser.add_argument("--adapter", type=str, required=True, choices=sorted(sampler.ADAPTERS.keys()))
    parser.add_argument("--code-file", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--model-tag",
        type=str,
        default=None,
        help=(
            "Optional short model label written into the output folder and metadata, "
            "for example horse_automobile or full_cifar10."
        ),
    )
    parser.add_argument("--prompt", type=str, required=True, help="Fixed prompt used for all seeds.")
    parser.add_argument("--seeds", type=str, required=True, help="Comma-separated integer seeds, e.g. 0,1,2,3")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prefer-device", type=str, default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--outdir", type=str, default="./attribution_samples")
    parser.add_argument("--num-trajectory-steps", type=int, default=100)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument(
        "--max-png-side",
        type=int,
        default=2048,
        help="Shrink saved PNGs so their longest side is at most this many pixels. Use 0 to disable.",
    )
    parser.add_argument("--cifar-data-root", type=str, default=None)
    parser.add_argument("--artbench-ae-checkpoint", type=str, default=None)

    args = parser.parse_args()

    seeds = parse_seeds_arg(args.seeds)
    adapter = sampler.make_adapter(
        name=args.adapter,
        code_file=args.code_file,
        checkpoint=args.checkpoint,
        prefer_device=args.prefer_device,
        cifar_data_root=args.cifar_data_root,
        artbench_ae_checkpoint=args.artbench_ae_checkpoint,
    )
    adapter.setup()

    save_timesteps = sampler.selected_timesteps_evenly(adapter.cfg.timesteps, args.num_trajectory_steps)
    ordered_timesteps = sorted(save_timesteps, reverse=True)

    run_root = resolve_output_root(
        args.outdir,
        args.adapter,
        args.checkpoint,
        args.prompt,
        model_tag=args.model_tag,
    )
    os.makedirs(run_root, exist_ok=True)

    manifest = {
        "adapter": args.adapter,
        "code_file": os.path.abspath(args.code_file),
        "checkpoint": os.path.abspath(args.checkpoint),
        "model_tag": args.model_tag,
        "prompt": args.prompt,
        "seeds": seeds,
        "batch_size": int(args.batch_size),
        "timesteps_total": int(adapter.cfg.timesteps),
        "num_trajectory_steps_requested": int(args.num_trajectory_steps),
        "saved_timesteps": [int(x) for x in ordered_timesteps],
        "metadata": adapter.metadata(),
    }
    with open(os.path.join(run_root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[setup] adapter={args.adapter}")
    print(f"[setup] model_tag={args.model_tag}")
    print(f"[setup] prompt={args.prompt}")
    print(f"[setup] num_seeds={len(seeds)} | batch_size={args.batch_size}")
    print(f"[setup] saving {len(ordered_timesteps)} evenly spaced timesteps")
    print(f"[setup] output_root={run_root}")

    all_start = time.time()
    for seed_idx, seed in enumerate(seeds, start=1):
        seed_start = time.time()
        print(f"[seed {seed_idx}/{len(seeds)}] sampling seed={seed}")

        final_state, saved_states, decoded_final = sample_model_space_trajectory(
            adapter=adapter,
            seed=seed,
            prompt=args.prompt,
            batch_size=args.batch_size,
            save_timesteps=ordered_timesteps,
        )

        seed_dir = os.path.join(run_root, f"seed_{seed:06d}")
        save_seed_outputs(
            seed_dir=seed_dir,
            adapter=adapter,
            seed=seed,
            prompt=args.prompt,
            model_tag=args.model_tag,
            ordered_timesteps=ordered_timesteps,
            save_timesteps=ordered_timesteps,
            final_state=final_state,
            saved_states=saved_states,
            decoded_final=decoded_final,
            upscale=args.upscale,
            max_png_side=args.max_png_side,
        )

        elapsed = time.time() - seed_start
        print(
            f"[seed {seed_idx}/{len(seeds)}] done | "
            f"elapsed={format_seconds(elapsed)} | "
            f"saved_to={seed_dir}"
        )

    total_elapsed = time.time() - all_start
    print(f"[done] total_time={format_seconds(total_elapsed)}")
    print(f"[done] output_root={run_root}")


if __name__ == "__main__":
    main()
