"""
50000 data points, 200 epochs, 1000 timesteps, 128 batch size, 2e-4 lr, 1e-4 weight decay, 0.999 ema decay, 100 log_every
3.5hrs on A100 40GB, 10.5hr on V100 32GB, 15.5hrs on RTX 3090 24GB (with mixed precision)
"""


import os
import math
import time
import sys
import pickle
import csv
import colorsys
import functools
from collections import deque
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import jax_utils
from flax.training import train_state
from flax.serialization import to_bytes, from_bytes
import optax


# ============================================================
# 3x3 dataset loader
# ============================================================

FIXED_SATURATION = 0.9
FIXED_VALUE = 0.9
TILE_SIZE = 3


class ColorGridDatasetJAX:
    def __init__(
        self,
        csv_path: str,
        grid_size: int = 3,
        fixed_s: float = 0.9,
        fixed_v: float = 0.9,
        label_start: Optional[int] = None,
        row_indices: Optional[Sequence[int]] = None,
        subset_ranges: Optional[Sequence[Tuple[int, int]]] = None,
    ):
        self.csv_path = csv_path
        self.grid_size = int(grid_size)
        self.num_tiles = self.grid_size * self.grid_size
        self.fixed_s = fixed_s
        self.fixed_v = fixed_v
        self.label_start = (1 + self.num_tiles) if label_start is None else int(label_start)

        if row_indices is not None and subset_ranges is not None and len(subset_ranges) > 0:
            raise ValueError("Use only one of row_indices or subset_ranges, not both.")

        if row_indices is None or len(row_indices) == 0:
            self.row_indices = None
        else:
            self.row_indices = [int(i) for i in row_indices]

        if subset_ranges is None or len(subset_ranges) == 0:
            self.subset_ranges = None
        else:
            self.subset_ranges = [(int(start), int(count)) for start, count in subset_ranges]

        self.rows: List[List[str]] = []
        self.vocab: Dict[str, int] = {}
        self.id_to_label: Dict[int, str] = {}
        self._load()

    def _expand_subset_ranges(self, n: int) -> List[int]:
        assert self.subset_ranges is not None
        out: List[int] = []
        for start, count in self.subset_ranges:
            if count <= 0:
                raise ValueError(f"subset_ranges contains non-positive count: {(start, count)}")
            if start < 0:
                raise ValueError(f"subset_ranges contains negative start: {(start, count)}")
            if start >= n:
                raise IndexError(f"subset_ranges start {start} is out of range for dataset size {n}")

            end = min(start + count, n)
            out.extend(range(start, end))
        return out

    def _load(self):
        all_rows: List[List[str]] = []
        with open(self.csv_path, newline="") as fh:
            reader = csv.reader(fh)
            for row in reader:
                if not row or row[0].lower() == "id":
                    continue
                all_rows.append(row)

        n = len(all_rows)

        if self.row_indices is not None:
            bad = [i for i in self.row_indices if i < 0 or i >= n]
            if bad:
                raise IndexError(
                    f"row_indices contain out-of-range values: {bad[:10]} (dataset has {n} rows)"
                )
            self.rows = [all_rows[i] for i in self.row_indices]
        elif self.subset_ranges is not None:
            expanded = self._expand_subset_ranges(n)
            self.rows = [all_rows[i] for i in expanded]
        else:
            self.rows = all_rows

        labels = set()
        for row in self.rows:
            for label in row[self.label_start:]:
                if label:
                    labels.add(label)

        labels = sorted(labels)
        self.vocab = {lab: i for i, lab in enumerate(labels)}
        self.id_to_label = {i: lab for lab, i in self.vocab.items()}

    def __len__(self):
        return len(self.rows)

    def _hues_from_row(self, row: List[str]) -> List[float]:
        hues = []
        for i in range(1, 1 + self.num_tiles):
            val = row[i]
            try:
                hue = float(val)
            except Exception:
                hue = float(val.strip('"'))
            hues.append(hue)
        return hues

    def _image_from_hues(self, hues: List[float]) -> np.ndarray:
        h = w = self.grid_size
        img = np.zeros((h, w, 3), dtype=np.float32)

        k = 0
        for i in range(h):
            for j in range(w):
                hue = hues[k]
                r, g, b = colorsys.hsv_to_rgb(hue, self.fixed_s, self.fixed_v)
                img[i, j, :] = (r, g, b)
                k += 1

        return img  # H,W,C

    def _cond_vector_from_row(self, row: List[str]) -> np.ndarray:
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        for label in row[self.label_start:]:
            if label and label in self.vocab:
                vec[self.vocab[label]] = 1.0
        return vec

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        row = self.rows[idx]
        hues = self._hues_from_row(row)
        img = self._image_from_hues(hues)
        cond = self._cond_vector_from_row(row)
        return img.astype(np.float32, copy=False), cond.astype(np.float32, copy=False)

    def batch_iterator(
        self,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        indices = np.arange(len(self))
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            end = start + batch_size
            if end > len(indices) and drop_last:
                break

            batch_idx = indices[start:end]
            xs: List[np.ndarray] = []
            ys: List[np.ndarray] = []
            for i in batch_idx:
                x, y = self[i]
                xs.append(x)
                ys.append(y)

            yield np.stack(xs, axis=0), np.stack(ys, axis=0)


# ============================================================
# Device helpers
# ============================================================

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


def choose_devices(prefer: str = "auto") -> List[jax.Device]:
    if prefer == "gpu":
        gpus = jax.devices("gpu")
        if not gpus:
            raise RuntimeError("Requested GPU, but JAX sees no GPU devices.")
        return gpus
    if prefer == "cpu":
        return jax.devices("cpu")
    backend = jax.default_backend()
    return jax.devices(backend)


def maybe_to_dtype(x: jnp.ndarray, use_bfloat16: bool) -> jnp.ndarray:
    return x.astype(jnp.bfloat16) if use_bfloat16 else x.astype(jnp.float32)


def resolve_compute_dtype(use_bfloat16: bool):
    return jnp.bfloat16 if use_bfloat16 else jnp.float32


def resolve_param_dtype():
    # Keep params in fp32 for stability; use compute dtype for activations.
    return jnp.float32


# ============================================================
# Diffusion schedule
# ============================================================

@dataclass
class DiffusionSchedule:
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alphas_cumprod: jnp.ndarray
    sqrt_alphas_cumprod: jnp.ndarray
    sqrt_one_minus_alphas_cumprod: jnp.ndarray


def make_diffusion_schedule(T: int, beta_start: float, beta_end: float) -> DiffusionSchedule:
    betas = jnp.linspace(beta_start, beta_end, T, dtype=jnp.float32)
    alphas = 1.0 - betas
    alphas_cumprod = jnp.cumprod(alphas)
    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        sqrt_alphas_cumprod=jnp.sqrt(alphas_cumprod),
        sqrt_one_minus_alphas_cumprod=jnp.sqrt(1.0 - alphas_cumprod),
    )


def extract(a: jnp.ndarray, t: jnp.ndarray, x_shape: Tuple[int, ...]) -> jnp.ndarray:
    out = a[t]
    return out.reshape((x_shape[0],) + (1,) * (len(x_shape) - 1))


def q_sample(schedule: DiffusionSchedule, x0: jnp.ndarray, t: jnp.ndarray, noise: jnp.ndarray) -> jnp.ndarray:
    return (
        extract(schedule.sqrt_alphas_cumprod, t, x0.shape) * x0
        + extract(schedule.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
    )


# ============================================================
# Embeddings / model
# ============================================================

def sinusoidal_time_embedding(timesteps: jnp.ndarray, dim: int) -> jnp.ndarray:
    half = dim // 2
    freqs = jnp.exp(-math.log(10000) * jnp.arange(half) / max(half - 1, 1))
    args = timesteps[:, None].astype(jnp.float32) * freqs[None, :]
    emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
    if dim % 2 == 1:
        emb = jnp.pad(emb, ((0, 0), (0, 1)))
    return emb


class TimeMLP(nn.Module):
    emb_dim: int
    out_dim: int
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        x = sinusoidal_time_embedding(t, self.emb_dim)
        x = nn.Dense(self.out_dim * 4, dtype=self.dtype, param_dtype=self.param_dtype)(x)
        x = nn.swish(x)
        x = nn.Dense(self.out_dim, dtype=self.dtype, param_dtype=self.param_dtype)(x)
        return x


class CondMLP(nn.Module):
    cond_dim: int
    out_dim: int
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, y: jnp.ndarray) -> jnp.ndarray:
        x = y.astype(self.dtype)
        x = nn.Dense(self.out_dim * 2, dtype=self.dtype, param_dtype=self.param_dtype)(x)
        x = nn.swish(x)
        x = nn.Dense(self.out_dim, dtype=self.dtype, param_dtype=self.param_dtype)(x)
        return x


class ResBlock(nn.Module):
    channels: int
    emb_dim: int
    dropout: float = 0.0
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, emb: jnp.ndarray, train: bool) -> jnp.ndarray:
        h = nn.GroupNorm(
            num_groups=min(8, self.channels),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(x)
        h = nn.swish(h)
        h = nn.Conv(
            self.channels,
            kernel_size=(3, 3),
            padding="SAME",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(h)

        emb_out = nn.Dense(self.channels, dtype=self.dtype, param_dtype=self.param_dtype)(nn.swish(emb))
        h = h + emb_out[:, None, None, :]

        h = nn.GroupNorm(
            num_groups=min(8, self.channels),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(h)
        h = nn.swish(h)
        h = nn.Dropout(rate=self.dropout)(h, deterministic=not train)
        h = nn.Conv(
            self.channels,
            kernel_size=(3, 3),
            padding="SAME",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(h)

        if x.shape[-1] != self.channels:
            x = nn.Conv(
                self.channels,
                kernel_size=(1, 1),
                padding="SAME",
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )(x)
        return x + h


class CNNDenoiser(nn.Module):
    in_channels: int = 3
    base_channels: int = 64
    time_emb_dim: int = 128
    cond_dim: int = 0
    class_cond: bool = True
    dropout: float = 0.0
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, y: Optional[jnp.ndarray], train: bool) -> jnp.ndarray:
        ch = self.base_channels
        emb = TimeMLP(self.time_emb_dim, ch, dtype=self.dtype, param_dtype=self.param_dtype)(t)

        if self.class_cond and y is not None:
            emb = emb + CondMLP(
                self.cond_dim,
                ch,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )(y)

        h = nn.Conv(ch, kernel_size=(3, 3), padding="SAME", dtype=self.dtype, param_dtype=self.param_dtype)(x)
        h = ResBlock(ch, ch, self.dropout, dtype=self.dtype, param_dtype=self.param_dtype)(h, emb, train)
        h = ResBlock(ch, ch, self.dropout, dtype=self.dtype, param_dtype=self.param_dtype)(h, emb, train)
        h = nn.GroupNorm(num_groups=min(8, ch), dtype=self.dtype, param_dtype=self.param_dtype)(h)
        h = nn.swish(h)
        out = nn.Conv(
            self.in_channels,
            kernel_size=(3, 3),
            padding="SAME",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(h)
        return out


# ============================================================
# Train state / checkpoint
# ============================================================

class TrainState(train_state.TrainState):
    ema_params: Any
    rng: jax.Array


def _save_checkpoint(checkpoint_dir: str, epoch: int, state: TrainState, cfg, keep_last_k: int = 5):
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"seed_{cfg.seed}_epoch_{epoch:04d}.ckpt")
    payload = {
        "epoch": int(epoch),
        "config": asdict(cfg),
        "state_bytes": to_bytes(state),
    }
    with open(ckpt_path, "wb") as f:
        pickle.dump(payload, f)

    paths = []
    for name in os.listdir(checkpoint_dir):
        if name.endswith(".ckpt") and "_epoch_" in name:
            try:
                ep = int(name.split("_epoch_")[-1].split(".ckpt")[0])
                paths.append((ep, os.path.join(checkpoint_dir, name)))
            except Exception:
                pass
    paths.sort(key=lambda x: x[0])

    if keep_last_k is not None and keep_last_k > 0 and len(paths) > keep_last_k:
        for _, old_path in paths[:-keep_last_k]:
            try:
                os.remove(old_path)
            except FileNotFoundError:
                pass


def _restore_checkpoint(ckpt_path: str, state_template: TrainState) -> Tuple[TrainState, int]:
    with open(ckpt_path, "rb") as f:
        payload = pickle.load(f)
    restored_state = from_bytes(state_template, payload["state_bytes"])
    start_epoch = int(payload.get("epoch", 0))
    return restored_state, start_epoch


# ============================================================
# Config
# ============================================================

@dataclass
class TrainConfig:
    # data
    csv_path: str = "generated_database/sample.csv"
    grid_size: int = 3
    fixed_s: float = 0.9
    fixed_v: float = 0.9
    label_start: Optional[int] = None
    row_indices: Optional[Tuple[int, ...]] = None
    subset_ranges: Optional[Tuple[Tuple[int, int], ...]] = None

    # model
    image_size: int = 3
    in_channels: int = 3
    base_channels: int = 160
    time_emb_dim: int = 128
    class_cond: bool = True
    dropout: float = 0.1

    # diffusion
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    predict_x0: bool = False

    # training
    seed: int = 0
    epochs: int = 200
    batch_size: int = 128
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    ema_decay: float = 0.999
    log_every: int = 100

    # device / precision
    prefer_device: str = "gpu"
    use_bfloat16: bool = False
    use_data_parallel: bool = True
    num_devices: Optional[int] = None

    # checkpoint / resume
    checkpoint_dir: str = "./models/colorgrid_checkpoints"
    save_every_epochs: int = 1
    keep_last_k: int = 5
    resume_from: Optional[str] = None

    # misc
    use_tqdm: bool = True

    # logging
    use_wandb: bool = True
    wandb_project: str = "DA-cnn-x3-pixel-training"
    wandb_entity: Optional[str] = "clearoboticslab"
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"  # "online", "offline", or "disabled"
    wandb_log_step_metrics: bool = False


# ============================================================
# Build model/state
# ============================================================

def build_model(cfg: TrainConfig, cond_dim: int) -> nn.Module:
    compute_dtype = resolve_compute_dtype(cfg.use_bfloat16)
    param_dtype = resolve_param_dtype()
    return CNNDenoiser(
        in_channels=cfg.in_channels,
        base_channels=cfg.base_channels,
        time_emb_dim=cfg.time_emb_dim,
        cond_dim=cond_dim,
        class_cond=cfg.class_cond,
        dropout=cfg.dropout,
        dtype=compute_dtype,
        param_dtype=param_dtype,
    )


def create_train_state(cfg: TrainConfig, model: nn.Module, rng: jax.Array, device, cond_dim: int) -> TrainState:
    compute_dtype = resolve_compute_dtype(cfg.use_bfloat16)
    dummy_x = jnp.zeros((1, cfg.image_size, cfg.image_size, cfg.in_channels), dtype=compute_dtype)
    dummy_t = jnp.zeros((1,), dtype=jnp.int32)
    dummy_y = jnp.zeros((1, cond_dim), dtype=jnp.float32) if cfg.class_cond else None

    with jax.default_device(device):
        variables = model.init(rng, dummy_x, dummy_t, dummy_y, train=True)
        params = variables["params"]

    tx = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        optax.adamw(learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay),
    )
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        ema_params=params,
        rng=rng,
    )


# ============================================================
# Loss / steps
# ============================================================

def mse_loss(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((pred - target) ** 2)


def make_train_step(schedule: DiffusionSchedule, cfg: TrainConfig):
    @jax.jit
    def train_step(state: TrainState, x0: jnp.ndarray, y: jnp.ndarray):
        rng, noise_rng, t_rng, dropout_rng = jax.random.split(state.rng, 4)
        B = x0.shape[0]
        t = jax.random.randint(t_rng, (B,), 0, cfg.timesteps)
        noise = jax.random.normal(noise_rng, x0.shape, dtype=x0.dtype)
        xt = q_sample(schedule, x0, t, noise)
        target = x0 if cfg.predict_x0 else noise

        def loss_fn(params):
            pred = state.apply_fn(
                {"params": params},
                xt,
                t,
                y if cfg.class_cond else None,
                train=True,
                rngs={"dropout": dropout_rng},
            )
            return mse_loss(pred, target)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        new_state = state.apply_gradients(grads=grads)
        ema_params = optax.incremental_update(
            new_state.params,
            state.ema_params,
            step_size=1.0 - cfg.ema_decay,
        )
        new_state = new_state.replace(ema_params=ema_params, rng=rng)
        metrics = {"loss": loss}
        return new_state, metrics

    return train_step


def make_eval_step(schedule: DiffusionSchedule, cfg: TrainConfig):
    @jax.jit
    def eval_step(state: TrainState, x0: jnp.ndarray, y: jnp.ndarray):
        rng, t_rng, noise_rng = jax.random.split(state.rng, 3)
        B = x0.shape[0]
        t = jax.random.randint(t_rng, (B,), 0, cfg.timesteps)
        noise = jax.random.normal(noise_rng, x0.shape, dtype=x0.dtype)
        xt = q_sample(schedule, x0, t, noise)
        target = x0 if cfg.predict_x0 else noise

        pred = state.apply_fn(
            {"params": state.ema_params},
            xt,
            t,
            y if cfg.class_cond else None,
            train=False,
        )
        loss = mse_loss(pred, target)
        new_state = state.replace(rng=rng)
        return new_state, {"loss": loss}

    return eval_step


def make_train_step_pmap(schedule: DiffusionSchedule, cfg: TrainConfig):
    @functools.partial(jax.pmap, axis_name="data")
    def train_step(state: TrainState, x0: jnp.ndarray, y: jnp.ndarray):
        rng, noise_rng, t_rng, dropout_rng = jax.random.split(state.rng, 4)
        B = x0.shape[0]
        t = jax.random.randint(t_rng, (B,), 0, cfg.timesteps)
        noise = jax.random.normal(noise_rng, x0.shape, dtype=x0.dtype)
        xt = q_sample(schedule, x0, t, noise)
        target = x0 if cfg.predict_x0 else noise

        def loss_fn(params):
            pred = state.apply_fn(
                {"params": params},
                xt,
                t,
                y if cfg.class_cond else None,
                train=True,
                rngs={"dropout": dropout_rng},
            )
            return mse_loss(pred, target)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        grads = jax.lax.pmean(grads, axis_name="data")
        loss = jax.lax.pmean(loss, axis_name="data")
        new_state = state.apply_gradients(grads=grads)
        ema_params = optax.incremental_update(
            new_state.params,
            state.ema_params,
            step_size=1.0 - cfg.ema_decay,
        )
        new_state = new_state.replace(ema_params=ema_params, rng=rng)
        metrics = {"loss": loss}
        return new_state, metrics

    return train_step


def make_eval_step_pmap(schedule: DiffusionSchedule, cfg: TrainConfig):
    @functools.partial(jax.pmap, axis_name="data")
    def eval_step(state: TrainState, x0: jnp.ndarray, y: jnp.ndarray):
        rng, t_rng, noise_rng = jax.random.split(state.rng, 3)
        B = x0.shape[0]
        t = jax.random.randint(t_rng, (B,), 0, cfg.timesteps)
        noise = jax.random.normal(noise_rng, x0.shape, dtype=x0.dtype)
        xt = q_sample(schedule, x0, t, noise)
        target = x0 if cfg.predict_x0 else noise

        pred = state.apply_fn(
            {"params": state.ema_params},
            xt,
            t,
            y if cfg.class_cond else None,
            train=False,
        )
        loss = mse_loss(pred, target)
        loss = jax.lax.pmean(loss, axis_name="data")
        new_state = state.replace(rng=rng)
        return new_state, {"loss": loss}

    return eval_step


# ============================================================
# Sampling
# ============================================================

def predict_x0_from_eps(schedule: DiffusionSchedule, xt: jnp.ndarray, t: jnp.ndarray, eps: jnp.ndarray):
    return (
        xt - extract(schedule.sqrt_one_minus_alphas_cumprod, t, xt.shape) * eps
    ) / extract(schedule.sqrt_alphas_cumprod, t, xt.shape)


def p_sample_loop(
    state: TrainState,
    model: nn.Module,
    schedule: DiffusionSchedule,
    cfg: TrainConfig,
    rng: jax.Array,
    shape: Tuple[int, ...],
    y: Optional[jnp.ndarray] = None,
):
    betas = schedule.betas
    alphas = schedule.alphas
    alphas_cumprod = schedule.alphas_cumprod
    cond_y = y if cfg.class_cond else None
    t_seq = jnp.arange(cfg.timesteps - 1, -1, -1, dtype=jnp.int32)

    @jax.jit
    def _sample_scan_loop(init_rng: jax.Array):
        init_x = jax.random.normal(init_rng, shape)

        def body_fn(carry, i):
            x, loop_rng = carry
            t = jnp.full((shape[0],), i, dtype=jnp.int32)
            pred = model.apply({"params": state.ema_params}, x, t, cond_y, train=False)
            eps = pred if not cfg.predict_x0 else (x - jnp.sqrt(alphas_cumprod[i]) * pred) / jnp.sqrt(1.0 - alphas_cumprod[i])
            x0_pred = pred if cfg.predict_x0 else predict_x0_from_eps(schedule, x, t, pred)
            x0_pred = jnp.clip(x0_pred, 0.0, 1.0)

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
            return (next_x, loop_rng), None

        (final_x, _), _ = jax.lax.scan(body_fn, (init_x, init_rng), t_seq)
        return final_x

    return _sample_scan_loop(rng)


# ============================================================
# Data prep
# ============================================================

def numpy_batch_to_jax(x_np: np.ndarray, y_np: np.ndarray, device, use_bfloat16: bool):
    with jax.default_device(device):
        x = jax.device_put(x_np)
        x = maybe_to_dtype(x, use_bfloat16)
        y = jax.device_put(y_np.astype(np.float32))
    return x, y


def numpy_batch_to_jax_pmap(
    x_np: np.ndarray,
    y_np: np.ndarray,
    devices: Sequence[jax.Device],
    use_bfloat16: bool,
):
    n_devices = len(devices)
    if x_np.shape[0] % n_devices != 0:
        raise ValueError(
            f"Batch size {x_np.shape[0]} must be divisible by number of devices {n_devices} for pmap."
        )

    per_device = x_np.shape[0] // n_devices
    x_np = x_np.reshape((n_devices, per_device) + x_np.shape[1:])
    y_np = y_np.reshape((n_devices, per_device) + y_np.shape[1:])

    x_np = x_np.astype(np.float32, copy=False)
    y_np = y_np.astype(np.float32, copy=False)

    x_dtype = jnp.bfloat16 if use_bfloat16 else jnp.float32
    x = jax.device_put_sharded([jnp.asarray(x_np[i], dtype=x_dtype) for i in range(n_devices)], devices)
    y = jax.device_put_sharded([y_np[i] for i in range(n_devices)], devices)
    return x, y


def prefetch_device_batches(
    np_batch_iter: Iterator[Tuple[np.ndarray, np.ndarray]],
    device,
    use_bfloat16: bool,
    prefetch_size: int = 2,
):
    queue = deque()

    def _push_one():
        x_np, y_np = next(np_batch_iter)
        queue.append(numpy_batch_to_jax(x_np, y_np, device, use_bfloat16))

    for _ in range(max(1, prefetch_size)):
        try:
            _push_one()
        except StopIteration:
            break

    while queue:
        batch = queue.popleft()
        try:
            _push_one()
        except StopIteration:
            pass
        yield batch


def prefetch_device_batches_pmap(
    np_batch_iter: Iterator[Tuple[np.ndarray, np.ndarray]],
    devices: Sequence[jax.Device],
    use_bfloat16: bool,
    prefetch_size: int = 2,
):
    queue = deque()

    def _push_one():
        x_np, y_np = next(np_batch_iter)
        queue.append(numpy_batch_to_jax_pmap(x_np, y_np, devices, use_bfloat16))

    for _ in range(max(1, prefetch_size)):
        try:
            _push_one()
        except StopIteration:
            break

    while queue:
        batch = queue.popleft()
        try:
            _push_one()
        except StopIteration:
            pass
        yield batch


# ============================================================
# Train loop
# ============================================================

def train(cfg: TrainConfig):
    all_devices = choose_devices(cfg.prefer_device)
    if cfg.num_devices is not None:
        if cfg.num_devices <= 0:
            raise ValueError(f"num_devices must be positive or None, got {cfg.num_devices}")
        if cfg.num_devices > len(all_devices):
            raise ValueError(
                f"Requested num_devices={cfg.num_devices}, but only {len(all_devices)} devices are visible."
            )
        devices = all_devices[: cfg.num_devices]
    else:
        devices = all_devices

    use_pmap = cfg.use_data_parallel and len(devices) > 1
    device = devices[0]
    print(f"Using backend={jax.default_backend()}, primary_device={device}")
    print(f"Visible devices ({len(devices)}): {[str(d) for d in devices]}")
    print(f"Data parallel enabled: {use_pmap}")
    print(
        "Precision policy:",
        f"compute_dtype={resolve_compute_dtype(cfg.use_bfloat16)},",
        f"param_dtype={resolve_param_dtype()}",
    )
    print("Config:", asdict(cfg))
    wandb_run = None
    if cfg.use_wandb:
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "cfg.use_wandb=True but wandb is not installed. "
                "Install with: pip install wandb"
            ) from e
        wandb_run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_run_name,
            mode=cfg.wandb_mode,
            config=asdict(cfg),
        )
        # Make epoch the default x-axis for train/eval metrics.
        wandb_run.define_metric("train/epoch")
        wandb_run.define_metric("train/loss_epoch", step_metric="train/epoch")
        wandb_run.define_metric("train/epoch_time_s", step_metric="train/epoch")
        wandb_run.define_metric("eval/loss", step_metric="train/epoch")

    ds = ColorGridDatasetJAX(
        csv_path=cfg.csv_path,
        grid_size=cfg.grid_size,
        fixed_s=cfg.fixed_s,
        fixed_v=cfg.fixed_v,
        label_start=cfg.label_start,
        row_indices=cfg.row_indices,
        subset_ranges=cfg.subset_ranges,
    )

    cond_dim = len(ds.vocab)
    print(f"Loaded {len(ds)} samples, vocab size={cond_dim}")

    if len(ds) == 0:
        raise ValueError("Dataset is empty.")
    if cfg.class_cond and cond_dim == 0:
        example_cols = len(ds.rows[0]) if len(ds.rows) > 0 else 0
        raise ValueError(
            "class_cond=True but no label vocabulary was found (cond_dim=0). "
            f"Check CSV label columns and label_start={ds.label_start}. "
            f"First row has {example_cols} columns. "
            "If your CSV has no labels, set class_cond=False."
        )

    total_train_start = time.time()

    rng = jax.random.PRNGKey(cfg.seed)
    schedule = make_diffusion_schedule(cfg.timesteps, cfg.beta_start, cfg.beta_end)
    model = build_model(cfg, cond_dim=cond_dim)
    state = create_train_state(cfg, model, rng, device, cond_dim=cond_dim)
    if use_pmap:
        state = jax_utils.replicate(state, devices=devices)
        state = state.replace(rng=jax.random.split(rng, len(devices)))
        train_step = make_train_step_pmap(schedule, cfg)
        eval_step = make_eval_step_pmap(schedule, cfg)
    else:
        train_step = make_train_step(schedule, cfg)
        eval_step = make_eval_step(schedule, cfg)

    start_epoch = 0
    if cfg.resume_from is not None:
        restore_template = jax_utils.unreplicate(state) if use_pmap else state
        restored_state, start_epoch = _restore_checkpoint(cfg.resume_from, restore_template)
        if use_pmap:
            state = jax_utils.replicate(restored_state, devices=devices)
            state = state.replace(rng=jax.random.split(restored_state.rng, len(devices)))
        else:
            state = restored_state
        print(f"Resumed from checkpoint: {cfg.resume_from} (completed through epoch {start_epoch})")

    train_drop_last = True
    eval_drop_last = False

    steps_per_epoch = len(ds) // cfg.batch_size if train_drop_last else math.ceil(len(ds) / cfg.batch_size)
    if steps_per_epoch == 0:
        raise ValueError(
            f"Dataset too small for batch_size={cfg.batch_size} when drop_last={train_drop_last}. "
            f"Either reduce batch_size or set train_drop_last=False."
        )
    if use_pmap and (cfg.batch_size % len(devices) != 0):
        raise ValueError(
            f"batch_size={cfg.batch_size} must be divisible by num_devices={len(devices)} for pmap training."
        )
    total_steps = steps_per_epoch * cfg.epochs

    global_step = start_epoch * steps_per_epoch

    try:
        for epoch in range(start_epoch + 1, cfg.epochs + 1):
            start_time = time.time()
            epoch_loss_sum = jnp.zeros((len(devices),), dtype=jnp.float32) if use_pmap else jnp.array(0.0, dtype=jnp.float32)
            epoch_steps = 0

            train_np_iter = ds.batch_iterator(
                batch_size=cfg.batch_size,
                shuffle=True,
                seed=cfg.seed + epoch,
                drop_last=train_drop_last,
            )
            if use_pmap:
                train_iter = prefetch_device_batches_pmap(train_np_iter, devices, cfg.use_bfloat16)
            else:
                train_iter = prefetch_device_batches(train_np_iter, device, cfg.use_bfloat16)

            if cfg.use_tqdm:
                try:
                    from tqdm.auto import tqdm
                except ImportError as e:
                    raise ImportError(
                        "cfg.use_tqdm=True but tqdm is not installed. "
                        "Install with: pip install tqdm"
                    ) from e
                train_iter = tqdm(
                    train_iter,
                    total=steps_per_epoch,
                    desc=f"Epoch {epoch}/{cfg.epochs}",
                    leave=True,
                    dynamic_ncols=True,
                    file=sys.stdout,
                )

            for x, y in train_iter:
                state, metrics = train_step(state, x, y)
                epoch_loss_sum = epoch_loss_sum + metrics["loss"]
                epoch_steps += 1
                global_step += 1

                if global_step % cfg.log_every == 0:
                    loss_val = float(metrics["loss"][0]) if use_pmap else float(metrics["loss"])
                    print(
                        f"epoch={epoch}/{cfg.epochs} "
                        f"step={global_step}/{total_steps} "
                        f"loss={loss_val:.6f}"
                    )
                    if cfg.use_tqdm:
                        train_iter.set_postfix(loss=f"{loss_val:.4f}")
                    if wandb_run is not None and cfg.wandb_log_step_metrics:
                        wandb_run.log(
                            {
                                "train/loss_step": loss_val,
                                "train/epoch": epoch,
                                "train/global_step": global_step,
                            }
                        )

            epoch_loss_arr = epoch_loss_sum / max(1, epoch_steps)
            epoch_loss = float(epoch_loss_arr[0]) if use_pmap else float(epoch_loss_arr)
            elapsed = time.time() - start_time
            print(f"[epoch {epoch}/{cfg.epochs}] train_loss={epoch_loss:.6f} time={elapsed:.1f}s")
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss_epoch": epoch_loss,
                        "train/epoch_time_s": elapsed,
                        "train/epoch": epoch,
                    }
                )

            eval_np_iter = ds.batch_iterator(
                batch_size=min(cfg.batch_size, len(ds)),
                shuffle=False,
                seed=0,
                drop_last=eval_drop_last,
            )
            if use_pmap:
                eval_iter = prefetch_device_batches_pmap(eval_np_iter, devices, cfg.use_bfloat16)
            else:
                eval_iter = prefetch_device_batches(eval_np_iter, device, cfg.use_bfloat16)
            for x, y in eval_iter:
                state, eval_metrics = eval_step(state, x, y)
                eval_loss = float(eval_metrics["loss"][0]) if use_pmap else float(eval_metrics["loss"])
                print(f"[epoch {epoch}/{cfg.epochs}] eval_loss={eval_loss:.6f}")
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "eval/loss": eval_loss,
                            "train/epoch": epoch,
                        }
                    )
                break

            if cfg.save_every_epochs > 0 and (epoch % cfg.save_every_epochs == 0):
                save_state = jax_utils.unreplicate(state) if use_pmap else state
                _save_checkpoint(
                    checkpoint_dir=cfg.checkpoint_dir,
                    epoch=epoch,
                    state=save_state,
                    cfg=cfg,
                    keep_last_k=cfg.keep_last_k,
                )
                print(f"Saved checkpoint for epoch {epoch} to {cfg.checkpoint_dir}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    total_elapsed = time.time() - total_train_start
    total_h = int(total_elapsed // 3600)
    total_m = int((total_elapsed % 3600) // 60)
    total_s = total_elapsed % 60

    print(
        f"Training finished. Total time: "
        f"{total_h:02d}h {total_m:02d}m {total_s:05.2f}s "
        f"({total_elapsed:.2f} seconds)"
    )

    out_state = jax_utils.unreplicate(state) if use_pmap else state
    return out_state, model, schedule, ds


# ============================================================
# Example main
# ============================================================

if __name__ == "__main__":
    cfg = TrainConfig(
        csv_path="databases/3x3_4342_100000.csv",
        grid_size=3,
        # Use full dataset by default.
        # subset_ranges=((1000, 10000), ),
        # row_indices=(1, 5, 9),   # use this instead of subset_ranges if needed

        image_size=3,
        in_channels=3,
        base_channels=160,
        time_emb_dim=128,
        class_cond=True,
        dropout=0.1,

        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        predict_x0=False,

        seed=0,
        epochs=200,
        batch_size=128,
        learning_rate=2e-4,
        weight_decay=1e-4,
        grad_clip_norm=1.0,
        ema_decay=0.999,
        log_every=20,

        prefer_device="auto",
        use_bfloat16=False,

        checkpoint_dir="./models/x3_pixel_checkpoints",
        save_every_epochs=1,
        keep_last_k=5,
        resume_from=None,
    )

    state, model, schedule, ds = train(cfg)

    # Example sampling:
    # rng = jax.random.PRNGKey(123)
    # cond = jnp.zeros((4, len(ds.vocab)), dtype=jnp.float32)
    # samples = p_sample_loop(
    #     state,
    #     model,
    #     schedule,
    #     cfg,
    #     rng,
    #     shape=(4, 3, 3, 3),
    #     y=cond,
    # )
    # print("sample shape:", samples.shape)