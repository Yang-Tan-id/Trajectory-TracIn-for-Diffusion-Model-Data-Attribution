from __future__ import annotations

"""
DM___sampler.py — unified JAX diffusion sampler with optional trajectory saving.

About 5 minutes to run a 1000-step trajectory (typical).

Example
-------
python3 "diffusion model_jax/DM___sampler.py"   --adapter cifar   --code-file "diffusion model_jax/DM__training_CIFAR10_pixel.py"   --checkpoint "diffusion model_jax/models/cifar10_checkpoints/seed_0_epoch_0200.ckpt"   --seed 8   --prompt "truck"   --batch-size 1   --num-trajectory-steps 20   --outdir "./samples/cifar"   --prefer-device gpu  --print-metadata

python DM___sampler.py \\
  --adapter x3 \\
  --code-file "DM__training_X3_pixel.py" \\
  --checkpoint "models/x3_checkpoints/epoch_0005.ckpt" \\
  --seed 0 \\
  --prompt "background_color_red,shape_color_blue,shape_ring" \\
  --batch-size 1 \\
  --num-trajectory-steps 9 \\
  --outdir ./samples/x3
"""

import argparse
import csv
import importlib.util
import json
import math
import os
import pickle
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw


# ============================================================
# Small utilities
# ============================================================

def load_python_module(module_path: str, module_name: str):
    module_path = os.path.abspath(module_path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_checkpoint_payload(ckpt_path: str) -> Dict[str, Any]:
    with open(ckpt_path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload is not a dict.")
    return payload


def choose_device(prefer: str = "auto"):
    if prefer == "gpu":
        gpus = jax.devices("gpu")
        if not gpus:
            raise RuntimeError("Requested GPU, but JAX sees no GPU devices.")
        return gpus[0]
    if prefer == "cpu":
        return jax.devices("cpu")[0]
    backend = jax.default_backend()
    return jax.devices(backend)[0]


def sanitize_for_path(x: str, max_len: int = 80) -> str:
    x = x.strip()
    if not x:
        return "empty_prompt"
    x = x.replace(",", "__")
    x = re.sub(r"[^A-Za-z0-9._-]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    if not x:
        x = "prompt"
    return x[:max_len]


def save_image_nhwc(img: np.ndarray, out_path: str, upscale: int = 1):
    arr = np.clip(img, 0.0, 1.0)
    arr = (arr * 255.0).round().astype(np.uint8)
    pil = Image.fromarray(arr)
    if upscale > 1:
        pil = pil.resize((pil.width * upscale, pil.height * upscale), resample=Image.NEAREST)
    pil.save(out_path)


def selected_timesteps_evenly(total_timesteps: int, num_steps: int) -> List[int]:
    if num_steps <= 1:
        return [0]
    xs = np.linspace(total_timesteps - 1, 0, num_steps)
    xs = [int(round(x)) for x in xs]
    # deduplicate while preserving order
    out = []
    seen = set()
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    if out[-1] != 0:
        out.append(0)
    return out


def make_image_grid(images: List[np.ndarray], labels: List[str], out_path: str, upscale: int = 1):
    if not images:
        raise ValueError("No images to place in grid.")

    pil_imgs = []
    for img in images:
        arr = np.clip(img, 0.0, 1.0)
        arr = (arr * 255.0).round().astype(np.uint8)
        pil = Image.fromarray(arr)
        if upscale > 1:
            pil = pil.resize((pil.width * upscale, pil.height * upscale), resample=Image.NEAREST)
        pil_imgs.append(pil)

    tile_w, tile_h = pil_imgs[0].size
    label_h = 18
    n = len(pil_imgs)
    cols = min(n, 4)
    rows = int(math.ceil(n / cols))

    canvas = Image.new("RGB", (cols * tile_w, rows * (tile_h + label_h)), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for idx, (pil, label) in enumerate(zip(pil_imgs, labels)):
        r = idx // cols
        c = idx % cols
        x0 = c * tile_w
        y0 = r * (tile_h + label_h)
        canvas.paste(pil, (x0, y0))
        draw.text((x0 + 2, y0 + tile_h + 1), label, fill=(0, 0, 0))

    canvas.save(out_path)


# ============================================================
# Adapter base class
# ============================================================

class ModelAdapter(ABC):
    def __init__(self, code_file: str, checkpoint: str, prefer_device: str = "auto"):
        self.code_file = code_file
        self.checkpoint = checkpoint
        self.payload = load_checkpoint_payload(checkpoint)
        self.cfg_dict = dict(self.payload.get("config", {}))
        if not self.cfg_dict:
            raise ValueError("Checkpoint does not contain a saved config.")
        self.device = choose_device(prefer_device)
        self.module = None
        self.cfg = None
        self.model = None
        self.state = None
        self.schedule = None

    def setup(self):
        self.module = self._load_module()
        self.cfg = self._build_config()
        self.schedule = self.module.make_diffusion_schedule(
            self.cfg.timesteps, self.cfg.beta_start, self.cfg.beta_end
        )
        self.model, self.state = self._restore_model_and_state()

    @abstractmethod
    def _load_module(self):
        raise NotImplementedError

    @abstractmethod
    def _build_config(self):
        raise NotImplementedError

    @abstractmethod
    def _restore_model_and_state(self):
        raise NotImplementedError

    @abstractmethod
    def make_condition(self, prompt: str, batch_size: int):
        raise NotImplementedError

    @abstractmethod
    def sample_shape(self, batch_size: int) -> Tuple[int, ...]:
        raise NotImplementedError

    @abstractmethod
    def decode_samples(self, samples_nhwc: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        raise NotImplementedError

    def predict_x0_from_eps(self, xt: jnp.ndarray, t: jnp.ndarray, eps: jnp.ndarray):
        return (
            xt - self.module.extract(self.schedule.sqrt_one_minus_alphas_cumprod, t, xt.shape) * eps
        ) / self.module.extract(self.schedule.sqrt_alphas_cumprod, t, xt.shape)

    def sample_with_trajectory(
        self,
        seed: int,
        prompt: str,
        batch_size: int,
        save_timesteps: List[int],
    ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        rng = jax.random.PRNGKey(seed)
        y = self.make_condition(prompt=prompt, batch_size=batch_size)
        shape = self.sample_shape(batch_size)

        betas = self.schedule.betas
        alphas = self.schedule.alphas
        alphas_cumprod = self.schedule.alphas_cumprod
        cond_y = y if getattr(self.cfg, "class_cond", True) else None
        t_seq = jnp.arange(self.cfg.timesteps - 1, -1, -1, dtype=jnp.int32)

        @jax.jit
        def _sample_scan_loop(init_rng: jax.Array):
            init_x = jax.random.normal(init_rng, shape)

            def body_fn(carry, i):
                x, loop_rng = carry
                t = jnp.full((shape[0],), i, dtype=jnp.int32)
                pred = self.model.apply(
                    {"params": self.state.ema_params},
                    x,
                    t,
                    cond_y,
                    train=False,
                )

                x0_pred = pred if self.cfg.predict_x0 else self.predict_x0_from_eps(x, t, pred)
                eps = pred if not self.cfg.predict_x0 else (
                    x - jnp.sqrt(alphas_cumprod[i]) * x0_pred
                ) / jnp.sqrt(1.0 - alphas_cumprod[i])

                alpha_t = alphas[i]
                abar_t = alphas_cumprod[i]
                beta_t = betas[i]
                coef1 = 1.0 / jnp.sqrt(alpha_t)
                coef2 = beta_t / jnp.sqrt(1.0 - abar_t)
                mean = coef1 * (x - coef2 * eps)

                loop_rng, step_rng = jax.random.split(loop_rng)
                noise = jax.random.normal(step_rng, shape)
                next_x = jax.lax.cond(
                    i > 0,
                    lambda _: mean + jnp.sqrt(beta_t) * noise,
                    lambda _: mean,
                    operand=None,
                )
                return (next_x, loop_rng), x0_pred

            (final_x, _), x0_preds = jax.lax.scan(body_fn, (init_x, init_rng), t_seq)
            return final_x, x0_preds

        final_x, x0_preds_seq = _sample_scan_loop(rng)
        final_images = self.decode_samples(np.array(final_x))
        saved: Dict[int, np.ndarray] = {}
        x0_preds_seq_np = np.array(x0_preds_seq)

        for i in save_timesteps:
            if i < 0 or i >= self.cfg.timesteps:
                continue
            seq_idx = self.cfg.timesteps - 1 - i
            saved[i] = np.array(self.decode_samples(x0_preds_seq_np[seq_idx]))

        if 0 not in saved:
            saved[0] = np.array(final_images)
        return final_images, saved


# ============================================================
# CIFAR adapter
# ============================================================

class CIFARAdapter(ModelAdapter):
    def __init__(
        self,
        code_file: str,
        checkpoint: str,
        prefer_device: str = "auto",
        cifar_data_root: Optional[str] = None,
    ):
        super().__init__(code_file, checkpoint, prefer_device)
        self._cifar_data_root_override = cifar_data_root

    def _load_module(self):
        return load_python_module(self.code_file, "user_cifar_train_module")

    def _build_config(self):
        return self.module.TrainConfig(**self.cfg_dict)

    def setup(self):
        self.module = self._load_module()
        self.cfg = self._build_config()
        self.cfg.data_root = self._resolve_cifar_data_root()
        self.schedule = self.module.make_diffusion_schedule(
            self.cfg.timesteps, self.cfg.beta_start, self.cfg.beta_end
        )
        self.model, self.state = self._restore_model_and_state()

    def _resolve_cifar_data_root(self) -> str:
        """batches.meta must exist; checkpoint paths are often relative to cwd or the training script dir."""
        meta_name = "batches.meta"

        def has_meta(root: str) -> bool:
            return os.path.isfile(os.path.join(root, meta_name))

        if self._cifar_data_root_override is not None:
            root = os.path.abspath(os.path.expanduser(self._cifar_data_root_override))
            if not has_meta(root):
                raise FileNotFoundError(
                    f"CIFAR {meta_name} not found under --cifar-data-root.\n"
                    f"  Expected: {os.path.join(root, meta_name)}"
                )
            return root

        cfg_root = os.path.abspath(os.path.expanduser(self.cfg.data_root))
        if has_meta(cfg_root):
            return cfg_root

        raw = self.cfg.data_root
        if not os.path.isabs(os.path.expanduser(raw)):
            rel = raw[2:] if raw.startswith("./") else raw
            code_dir = os.path.dirname(os.path.abspath(self.code_file))
            candidate = os.path.normpath(os.path.join(code_dir, rel))
            if has_meta(candidate):
                print(
                    f"[cifar] data_root not found at {cfg_root}; using {candidate} (relative to --code-file directory)"
                )
                return candidate

        raise FileNotFoundError(
            "Could not find CIFAR batches.meta for class names / prompts.\n"
            f"  Tried checkpoint config path: {os.path.join(cfg_root, meta_name)}\n"
            "  If you run from a directory that does not contain ./databases/..., either:\n"
            "    cd to the same directory you used for training, or\n"
            '    pass --cifar-data-root /path/to/cifar-10-batches-py\n'
            "  (the directory that directly contains batches.meta)."
        )

    def _load_label_names(self) -> List[str]:
        meta_path = os.path.join(self.cfg.data_root, "batches.meta")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f, encoding="bytes")
        labels = meta[b"label_names"]
        out = []
        for x in labels:
            out.append(x.decode("utf-8") if isinstance(x, bytes) else str(x))
        return out

    def _restore_model_and_state(self):
        model = self.module.build_model(self.cfg)
        rng = jax.random.PRNGKey(self.cfg.seed)
        state_template = self.module.create_train_state(self.cfg, model, rng, self.device)
        state, start_epoch = self.module._restore_checkpoint(self.checkpoint, state_template)
        print(f"[cifar] restored checkpoint through epoch {start_epoch}")
        return model, state

    def _cond_mode(self) -> str:
        return getattr(self.cfg, "cond_mode", "class_id")

    def _parse_prompt_tokens(self, prompt: str) -> List[str]:
        tokens = [x.strip() for x in prompt.split(",") if x.strip()]
        if not tokens:
            prompt = prompt.strip()
            if prompt:
                tokens = [prompt]
        return tokens

    def _parse_prompt_to_class_id(self, prompt: str) -> int:
        labels = self._load_label_names()
        name_to_id = {name: i for i, name in enumerate(labels)}

        prompt = prompt.strip()
        if prompt.isdigit():
            cid = int(prompt)
            if cid < 0 or cid >= len(labels):
                raise ValueError(f"Class id {cid} out of range [0, {len(labels)-1}]")
            return cid

        if prompt not in name_to_id:
            raise ValueError(f"Unknown CIFAR class prompt '{prompt}'. Valid names: {labels}")
        return name_to_id[prompt]

    def _parse_prompt_to_multi_hot(self, prompt: str) -> np.ndarray:
        labels = self._load_label_names()
        name_to_id = {name: i for i, name in enumerate(labels)}
        tokens = self._parse_prompt_tokens(prompt)

        bad = []
        vec = np.zeros((len(labels),), dtype=np.float32)

        for tok in tokens:
            if tok.isdigit():
                cid = int(tok)
                if 0 <= cid < len(labels):
                    vec[cid] = 1.0
                else:
                    bad.append(tok)
            elif tok in name_to_id:
                vec[name_to_id[tok]] = 1.0
            else:
                bad.append(tok)

        if bad:
            raise ValueError(
                f"Unknown CIFAR prompt token(s): {bad}. "
                f"Valid names: {labels}; valid ids: 0..{len(labels)-1}"
            )

        if vec.sum() == 0:
            raise ValueError("CIFAR multi_hot prompt produced an empty conditioning vector.")
        return vec

    def make_condition(self, prompt: str, batch_size: int):
        if not self.cfg.class_cond:
            return None

        cond_mode = self._cond_mode()

        if cond_mode == "class_id":
            tokens = self._parse_prompt_tokens(prompt)
            if len(tokens) != 1:
                raise ValueError(
                    "This CIFAR checkpoint uses cond_mode='class_id', so --prompt must contain exactly one class "
                    "name or one class id."
                )
            class_id = self._parse_prompt_to_class_id(tokens[0])
            with jax.default_device(self.device):
                return jax.device_put(jnp.full((batch_size,), class_id, dtype=jnp.int32))

        if cond_mode == "multi_hot":
            vec = self._parse_prompt_to_multi_hot(prompt)
            mat = np.repeat(vec[None, :], batch_size, axis=0)
            with jax.default_device(self.device):
                return jax.device_put(jnp.array(mat, dtype=jnp.float32))

        raise ValueError(f"Unknown CIFAR cond_mode: {cond_mode}")

    def sample_shape(self, batch_size: int) -> Tuple[int, ...]:
        return (batch_size, self.cfg.image_size, self.cfg.image_size, self.cfg.in_channels)

    def decode_samples(self, samples_nhwc: np.ndarray) -> np.ndarray:
        return np.clip((samples_nhwc + 1.0) / 2.0, 0.0, 1.0)

    def metadata(self) -> Dict[str, Any]:
        labels = self._load_label_names()
        return {
            "adapter": "cifar",
            "data_root": self.cfg.data_root,
            "image_size": int(self.cfg.image_size),
            "in_channels": int(self.cfg.in_channels),
            "class_cond": bool(self.cfg.class_cond),
            "cond_mode": self._cond_mode(),
            "label_names": labels,
        }


# ============================================================
# X3 adapter
# ============================================================

class X3Adapter(ModelAdapter):
    def _load_module(self):
        return load_python_module(self.code_file, "user_x3_train_module")

    def _build_config(self):
        return self.module.TrainConfig(**self.cfg_dict)

    def _build_vocab(self) -> Dict[str, int]:
        csv_path = self.cfg.csv_path
        grid_size = int(self.cfg.grid_size)
        num_tiles = grid_size * grid_size
        label_start = (1 + num_tiles) if self.cfg.label_start is None else int(self.cfg.label_start)

        rows: List[List[str]] = []
        with open(csv_path, newline="") as fh:
            reader = csv.reader(fh)
            for row in reader:
                if not row:
                    continue
                if str(row[0]).lower() == "id":
                    continue
                rows.append(row)

        if self.cfg.row_indices is not None and len(self.cfg.row_indices) > 0:
            selected = [rows[int(i)] for i in self.cfg.row_indices]
        elif self.cfg.subset_ranges is not None and len(self.cfg.subset_ranges) > 0:
            expanded = []
            for start, count in self.cfg.subset_ranges:
                start = int(start)
                count = int(count)
                expanded.extend(list(range(start, min(start + count, len(rows)))))
            selected = [rows[i] for i in expanded]
        else:
            selected = rows

        labels = set()
        for row in selected:
            for lab in row[label_start:]:
                if lab:
                    labels.add(lab)

        labels = sorted(labels)
        return {lab: i for i, lab in enumerate(labels)}

    def _restore_model_and_state(self):
        vocab = self._build_vocab()
        cond_dim = len(vocab)
        model = self.module.build_model(self.cfg, cond_dim=cond_dim)
        rng = jax.random.PRNGKey(self.cfg.seed)
        state_template = self.module.create_train_state(self.cfg, model, rng, self.device, cond_dim=cond_dim)
        state, start_epoch = self.module._restore_checkpoint(self.checkpoint, state_template)
        print(f"[x3] restored checkpoint through epoch {start_epoch}, cond_dim={cond_dim}")
        return model, state

    def make_condition(self, prompt: str, batch_size: int):
        if not self.cfg.class_cond:
            return None

        vocab = self._build_vocab()
        tokens = [x.strip() for x in prompt.split(",") if x.strip()]
        if not tokens:
            raise ValueError("For x3, prompt must contain at least one comma-separated label.")

        bad = [tok for tok in tokens if tok not in vocab]
        if bad:
            valid_preview = list(vocab.keys())[:30]
            raise ValueError(
                f"Unknown x3 labels: {bad}. "
                f"Here are some valid labels: {valid_preview}"
            )

        vec = np.zeros((len(vocab),), dtype=np.float32)
        for tok in tokens:
            vec[vocab[tok]] = 1.0

        mat = np.repeat(vec[None, :], batch_size, axis=0)
        with jax.default_device(self.device):
            return jax.device_put(jnp.array(mat, dtype=jnp.float32))

    def sample_shape(self, batch_size: int) -> Tuple[int, ...]:
        return (batch_size, self.cfg.image_size, self.cfg.image_size, self.cfg.in_channels)

    def decode_samples(self, samples_nhwc: np.ndarray) -> np.ndarray:
        return np.clip(samples_nhwc, 0.0, 1.0)

    def metadata(self) -> Dict[str, Any]:
        vocab = self._build_vocab()
        return {
            "adapter": "x3",
            "image_size": int(self.cfg.image_size),
            "in_channels": int(self.cfg.in_channels),
            "class_cond": bool(self.cfg.class_cond),
            "vocab_size": len(vocab),
            "vocab_preview": list(vocab.keys())[:50],
        }


ADAPTERS = {
    "cifar": CIFARAdapter,
    "x3": X3Adapter,
}


def make_adapter(
    name: str,
    code_file: str,
    checkpoint: str,
    prefer_device: str = "auto",
    cifar_data_root: Optional[str] = None,
) -> ModelAdapter:
    if name not in ADAPTERS:
        raise ValueError(f"Unknown adapter '{name}'. Available: {sorted(ADAPTERS.keys())}")
    if cifar_data_root is not None and name != "cifar":
        raise ValueError("--cifar-data-root is only valid with --adapter cifar")
    if name == "cifar":
        return CIFARAdapter(
            code_file=code_file,
            checkpoint=checkpoint,
            prefer_device=prefer_device,
            cifar_data_root=cifar_data_root,
        )
    return ADAPTERS[name](code_file=code_file, checkpoint=checkpoint, prefer_device=prefer_device)


def main():
    parser = argparse.ArgumentParser(description="Unified JAX diffusion sampler with trajectory saving.")
    parser.add_argument("--adapter", type=str, required=True, choices=sorted(ADAPTERS.keys()))
    parser.add_argument("--code-file", type=str, required=True, help="Path to the training .py file for that adapter.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a saved checkpoint (.ckpt).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", type=str, required=True, help="Conditioning prompt. Format depends on adapter.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prefer-device", type=str, default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--outdir", type=str, default="./generated_samples")
    parser.add_argument("--prefix", type=str, default="sample")
    parser.add_argument("--upscale", type=int, default=64, help="Only affects saved PNG size.")
    parser.add_argument("--num-trajectory-steps", type=int, default=9,
                        help="Number of evenly spaced reverse timesteps to save, including near start and final.")
    parser.add_argument("--print-metadata", action="store_true")
    parser.add_argument(
        "--cifar-data-root",
        type=str,
        default=None,
        help="Path to cifar-10-batches-py (folder containing batches.meta). Use when the saved data_root is wrong for your cwd.",
    )

    args = parser.parse_args()

    adapter = make_adapter(
        name=args.adapter,
        code_file=args.code_file,
        checkpoint=args.checkpoint,
        prefer_device=args.prefer_device,
        cifar_data_root=args.cifar_data_root,
    )
    adapter.setup()

    if args.print_metadata:
        meta = adapter.metadata()
        print("=== adapter metadata ===")
        for k, v in meta.items():
            print(f"{k}: {v}")
        print("========================")

    safe_prompt = sanitize_for_path(args.prompt)
    result_dir = os.path.join(args.outdir, f"result_{safe_prompt}_seed{args.seed}")
    os.makedirs(result_dir, exist_ok=True)

    save_timesteps = selected_timesteps_evenly(adapter.cfg.timesteps, args.num_trajectory_steps)

    generation_start_time = time.time()
    final_images, saved = adapter.sample_with_trajectory(
        seed=args.seed,
        prompt=args.prompt,
        batch_size=args.batch_size,
        save_timesteps=save_timesteps,
    )
    generation_end_time = time.time()
    generation_seconds = float(generation_end_time - generation_start_time)
    seconds_per_sample = float(generation_seconds / max(1, args.batch_size))

    # Save final batch
    np.save(os.path.join(result_dir, "final_samples.npy"), final_images)

    info = {
        "adapter": args.adapter,
        "code_file": args.code_file,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "prompt": args.prompt,
        "batch_size": args.batch_size,
        "num_trajectory_steps_requested": args.num_trajectory_steps,
        "saved_timesteps": save_timesteps,
        "result_dir": result_dir,
        "generation_seconds_total": generation_seconds,
        "generation_seconds_per_sample": seconds_per_sample,
        "metadata": adapter.metadata(),
    }
    with open(os.path.join(result_dir, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    # Save trajectory arrays and images
    sample_dirs = []
    ordered_ts = sorted(saved.keys(), reverse=True)
    for b in range(args.batch_size):
        sample_dir = os.path.join(result_dir, f"sample_{b:03d}")
        os.makedirs(sample_dir, exist_ok=True)
        sample_dirs.append(sample_dir)

        traj_stack = []
        labels = []

        for t in ordered_ts:
            img = saved[t][b]
            traj_stack.append(img)
            labels.append(f"t={t}")

            png_path = os.path.join(sample_dir, f"t_{t:04d}.png")
            save_image_nhwc(img, png_path, upscale=max(1, int(args.upscale)))

        traj_arr = np.stack(traj_stack, axis=0)
        np.save(os.path.join(sample_dir, "trajectory.npy"), traj_arr)

        grid_path = os.path.join(sample_dir, "trajectory_grid.png")
        make_image_grid(traj_stack, labels, grid_path, upscale=max(1, int(args.upscale)))

        final_png = os.path.join(sample_dir, "final.png")
        save_image_nhwc(final_images[b], final_png, upscale=max(1, int(args.upscale)))

    # Also save one batch-wide final image copy for convenience
    for i, img in enumerate(final_images):
        save_image_nhwc(img, os.path.join(result_dir, f"{args.prefix}_final_{i:03d}.png"),
                        upscale=max(1, int(args.upscale)))

    print(f"Saved results to: {result_dir}")
    print(f"Saved timesteps: {ordered_ts}")
    print(f"Generation time (total): {generation_seconds:.4f} seconds")
    print(f"Generation time (per sample): {seconds_per_sample:.4f} seconds")
    print("Per sample, you now have:")
    print("  - one PNG per saved timestep")
    print("  - trajectory.npy")
    print("  - trajectory_grid.png")
    print("  - final.png")


if __name__ == "__main__":
    main()
