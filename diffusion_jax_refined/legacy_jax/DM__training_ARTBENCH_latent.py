"""
Latent diffusion trainer for ArtBench image-folder datasets.

Pipeline
--------
1. Train a convolutional autoencoder on images.
2. Encode train/test images into cached latent tensors.
3. Train the diffusion model in latent space instead of pixel space.

Why this compresses the data
----------------------------
For example, with:
  image_size = 256
  ae_downsample_factor = 8
  latent_channels = 4

the encoder maps:
  (256, 256, 3) -> (32, 32, 4)

This keeps 2D spatial structure, but reduces the total number of values from
256*256*3 = 196608
to
32*32*4 = 4096

which is 48x smaller. The diffusion model then runs on the 32x32x4 latent map,
which is much faster than running on raw 256x256 pixels.
"""

import os
import sys
import pickle
import time
from dataclasses import dataclass, asdict
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from flax.serialization import to_bytes, from_bytes
import optax

import DM__training_CIFAR10_pixel as base
from dataset_loader_artbench_latent import (
    ArtBenchImageFolderLatentDataset,
    ArtBenchLatentDataset,
    save_latent_dataset_npz,
)


class AETrainState(train_state.TrainState):
    rng: jax.Array


@dataclass
class LatentArtBenchConfig:
    # data
    data_root: str = "./databases/artbench-10-imagefolder-split"
    train_split: str = "train"
    test_split: str = "test"
    class_names: Optional[Tuple[str, ...]] = None
    image_size: int = 128
    resize_mode: str = "shortest_center_crop"
    file_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")

    train_exclude_ranges: Optional[Tuple[Tuple[Any, int, int], ...]] = None
    train_exclude_indices: Optional[dict] = None
    train_exclude_files: Optional[Tuple[str, ...]] = None
    test_exclude_ranges: Optional[Tuple[Tuple[Any, int, int], ...]] = None
    test_exclude_indices: Optional[dict] = None
    test_exclude_files: Optional[Tuple[str, ...]] = None

    # autoencoder
    ae_base_channels: int = 64
    latent_channels: int = 4
    ae_downsample_factor: int = 8
    ae_epochs: int = 40
    ae_batch_size: int = 32
    ae_learning_rate: float = 2e-4
    ae_weight_decay: float = 1e-4
    ae_log_every: int = 50

    # diffusion on latents
    dm_model_type: str = "unet"
    dm_class_cond: bool = True
    dm_cond_mode: str = "class_id"  # "class_id" or "multi_hot"
    dm_base_channels: int = 160
    dm_channel_mults: Tuple[int, ...] = (1, 2, 2)
    dm_num_res_blocks: int = 2
    dm_time_emb_dim: int = 128
    dm_dropout: float = 0.1
    dm_epochs: int = 200
    dm_batch_size: int = 128
    dm_learning_rate: float = 2e-4
    dm_weight_decay: float = 1e-4
    dm_adam_b1: float = 0.9
    dm_adam_b2: float = 0.999
    dm_adam_eps: float = 1e-8
    dm_grad_clip_norm: float = 1.0
    dm_ema_decay: float = 0.999
    dm_log_every: int = 100
    dm_timesteps: int = 1000
    dm_beta_start: float = 1e-4
    dm_beta_end: float = 0.02
    dm_predict_x0: bool = False

    # misc
    seed: int = 0
    use_bfloat16: bool = True
    prefer_device: str = "gpu"
    use_tqdm: bool = True
    reuse_autoencoder: bool = True

    # outputs
    cache_dir: str = "./latents/artbench"
    autoencoder_model_dir: str = "./models/artbench_latent_autoencoder"
    dm_checkpoint_dir: str = "./models/artbench_latent_dm_checkpoints"
    keep_last_k: Optional[int] = 5


class ConvBlock(nn.Module):
    out_channels: int
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        in_ch = x.shape[-1]
        h = nn.GroupNorm(num_groups=min(8, max(1, in_ch)), dtype=self.dtype, param_dtype=self.param_dtype)(x)
        h = nn.swish(h)
        h = nn.Conv(self.out_channels, (3, 3), padding="SAME", dtype=self.dtype, param_dtype=self.param_dtype)(h)
        h = nn.GroupNorm(num_groups=min(8, max(1, self.out_channels)), dtype=self.dtype, param_dtype=self.param_dtype)(h)
        h = nn.swish(h)
        h = nn.Conv(self.out_channels, (3, 3), padding="SAME", dtype=self.dtype, param_dtype=self.param_dtype)(h)
        if in_ch != self.out_channels:
            x = nn.Conv(self.out_channels, (1, 1), padding="SAME", dtype=self.dtype, param_dtype=self.param_dtype)(x)
        return x + h


class Encoder(nn.Module):
    base_channels: int
    latent_channels: int
    num_downsamples: int
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        h = nn.Conv(self.base_channels, (3, 3), padding="SAME", dtype=self.dtype, param_dtype=self.param_dtype)(x)
        channels = self.base_channels
        for _ in range(self.num_downsamples):
            h = ConvBlock(channels, dtype=self.dtype, param_dtype=self.param_dtype)(h)
            h = nn.Conv(channels * 2, (4, 4), strides=(2, 2), padding="SAME", dtype=self.dtype, param_dtype=self.param_dtype)(h)
            channels *= 2
        h = ConvBlock(channels, dtype=self.dtype, param_dtype=self.param_dtype)(h)
        z = nn.Conv(self.latent_channels, (3, 3), padding="SAME", dtype=self.dtype, param_dtype=self.param_dtype)(h)
        return z


class Decoder(nn.Module):
    base_channels: int
    latent_channels: int
    num_upsamples: int
    out_channels: int = 3
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, z):
        channels = self.base_channels * (2 ** self.num_upsamples)
        h = nn.Conv(channels, (3, 3), padding="SAME", dtype=self.dtype, param_dtype=self.param_dtype)(z)
        h = ConvBlock(channels, dtype=self.dtype, param_dtype=self.param_dtype)(h)
        for _ in range(self.num_upsamples):
            b, hgt, wid, ch = h.shape
            h = jax.image.resize(h, (b, hgt * 2, wid * 2, ch), method="nearest")
            channels = max(self.base_channels, channels // 2)
            h = nn.Conv(channels, (3, 3), padding="SAME", dtype=self.dtype, param_dtype=self.param_dtype)(h)
            h = ConvBlock(channels, dtype=self.dtype, param_dtype=self.param_dtype)(h)
        h = nn.GroupNorm(num_groups=min(8, max(1, channels)), dtype=self.dtype, param_dtype=self.param_dtype)(h)
        h = nn.swish(h)
        return nn.Conv(self.out_channels, (3, 3), padding="SAME", dtype=self.dtype, param_dtype=self.param_dtype)(h)


class Autoencoder(nn.Module):
    base_channels: int
    latent_channels: int
    num_downsamples: int
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self):
        self.encoder = Encoder(
            base_channels=self.base_channels,
            latent_channels=self.latent_channels,
            num_downsamples=self.num_downsamples,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.decoder = Decoder(
            base_channels=self.base_channels,
            latent_channels=self.latent_channels,
            num_upsamples=self.num_downsamples,
            out_channels=3,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def __call__(self, x):
        return self.decode(self.encode(x))


def _save_autoencoder_checkpoint(path: str, state: AETrainState, cfg: LatentArtBenchConfig):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(
            {
                "config": asdict(cfg),
                "state_bytes": to_bytes(state),
            },
            f,
        )


def _restore_autoencoder_checkpoint(path: str, template: AETrainState) -> AETrainState:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return from_bytes(template, payload["state_bytes"])


def _autoencoder_checkpoint_path(cfg: LatentArtBenchConfig) -> str:
    return os.path.join(cfg.autoencoder_model_dir, "ae_state.ckpt")


def create_autoencoder_state(cfg: LatentArtBenchConfig, model: Autoencoder, rng: jax.Array, device) -> AETrainState:
    compute_dtype = base.resolve_compute_dtype(cfg.use_bfloat16)
    dummy_x = jnp.zeros((1, cfg.image_size, cfg.image_size, 3), dtype=compute_dtype)
    with jax.default_device(device):
        variables = model.init(rng, dummy_x)
        params = variables["params"]

    tx = optax.adamw(learning_rate=cfg.ae_learning_rate, weight_decay=cfg.ae_weight_decay)
    return AETrainState.create(apply_fn=model.apply, params=params, tx=tx, rng=rng)


def reconstruction_loss(x_recon: jnp.ndarray, x_target: jnp.ndarray) -> Tuple[jnp.ndarray, dict]:
    l1 = jnp.mean(jnp.abs(x_recon - x_target))
    l2 = jnp.mean((x_recon - x_target) ** 2)
    total = l1 + 0.5 * l2
    return total, {"loss": total, "l1": l1, "l2": l2}


def make_ae_train_step(model: Autoencoder):
    @jax.jit
    def train_step(state: AETrainState, x: jnp.ndarray):
        def loss_fn(params):
            recon = model.apply({"params": params}, x)
            loss, metrics = reconstruction_loss(recon, x)
            return loss, metrics

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, {"loss": loss, "l1": aux["l1"], "l2": aux["l2"]}

    return train_step


def make_ae_eval_step(model: Autoencoder):
    @jax.jit
    def eval_step(state: AETrainState, x: jnp.ndarray):
        recon = model.apply({"params": state.params}, x)
        loss, metrics = reconstruction_loss(recon, x)
        return {"loss": loss, "l1": metrics["l1"], "l2": metrics["l2"]}

    return eval_step


def _iter_with_tqdm(iterator, total: int, desc: str, use_tqdm: bool):
    if not use_tqdm:
        return iterator
    try:
        from tqdm.auto import tqdm
    except ImportError as e:
        raise ImportError("use_tqdm=True but tqdm is not installed. Install with: pip install tqdm") from e
    return tqdm(iterator, total=total, desc=desc, leave=True, dynamic_ncols=True, file=sys.stdout)


def train_autoencoder(cfg: LatentArtBenchConfig, train_ds, test_ds, device):
    compute_dtype = base.resolve_compute_dtype(cfg.use_bfloat16)
    param_dtype = base.resolve_param_dtype()
    num_downsamples = int(np.log2(cfg.ae_downsample_factor))
    if 2 ** num_downsamples != cfg.ae_downsample_factor:
        raise ValueError("ae_downsample_factor must be a power of two.")

    model = Autoencoder(
        base_channels=cfg.ae_base_channels,
        latent_channels=cfg.latent_channels,
        num_downsamples=num_downsamples,
        dtype=compute_dtype,
        param_dtype=param_dtype,
    )

    rng = jax.random.PRNGKey(cfg.seed)
    state = create_autoencoder_state(cfg, model, rng, device)
    ae_ckpt_path = _autoencoder_checkpoint_path(cfg)
    if cfg.reuse_autoencoder and os.path.exists(ae_ckpt_path):
        try:
            state = _restore_autoencoder_checkpoint(ae_ckpt_path, state)
            print(f"Restored autoencoder checkpoint: {ae_ckpt_path}")
            return state, model
        except Exception as e:
            print(f"Warning: failed to restore autoencoder checkpoint, retraining from scratch: {e}")

    train_step = make_ae_train_step(model)
    eval_step = make_ae_eval_step(model)

    steps_per_epoch = len(train_ds) // cfg.ae_batch_size
    global_step = 0

    for epoch in range(1, cfg.ae_epochs + 1):
        epoch_start = time.time()
        train_iter = train_ds.batch_iterator(cfg.ae_batch_size, shuffle=True, seed=cfg.seed + epoch, drop_last=True)
        train_iter = _iter_with_tqdm(train_iter, steps_per_epoch, f"AE Epoch {epoch}/{cfg.ae_epochs}", cfg.use_tqdm)
        loss_sum = 0.0
        n_steps = 0

        for x, _ in train_iter:
            state, metrics = train_step(state, x)
            loss_sum += float(metrics["loss"])
            n_steps += 1
            global_step += 1
            if global_step % cfg.ae_log_every == 0:
                print(
                    f"[ae] epoch={epoch}/{cfg.ae_epochs} step={global_step} "
                    f"loss={float(metrics['loss']):.6f} l1={float(metrics['l1']):.6f} l2={float(metrics['l2']):.6f}"
                )

        eval_iter = test_ds.batch_iterator(cfg.ae_batch_size, shuffle=False, seed=0, drop_last=False)
        eval_metrics = None
        for x, _ in eval_iter:
            eval_metrics = eval_step(state, x)
            break

        epoch_elapsed = time.time() - epoch_start
        mean_loss = loss_sum / max(1, n_steps)
        if eval_metrics is not None:
            print(
                f"[ae epoch {epoch}/{cfg.ae_epochs}] "
                f"train_loss={mean_loss:.6f} "
                f"eval_loss={float(eval_metrics['loss']):.6f} "
                f"time={epoch_elapsed:.1f}s"
            )
        else:
            print(f"[ae epoch {epoch}/{cfg.ae_epochs}] train_loss={mean_loss:.6f} time={epoch_elapsed:.1f}s")

    _save_autoencoder_checkpoint(ae_ckpt_path, state, cfg)
    print(f"Saved autoencoder checkpoint to {ae_ckpt_path}")
    return state, model


def encode_split_to_npz(
    cfg: LatentArtBenchConfig,
    state: AETrainState,
    model: Autoencoder,
    dataset: ArtBenchImageFolderLatentDataset,
    out_path: str,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    encode_fn = jax.jit(lambda params, x: model.apply({"params": params}, x, method=Autoencoder.encode))

    latents = []
    labels = []
    batch_iter = dataset.batch_iterator(cfg.ae_batch_size, shuffle=False, seed=0, drop_last=False)
    for x, y in batch_iter:
        z = encode_fn(state.params, x)
        latents.append(np.asarray(z, dtype=np.float32))
        labels.append(np.asarray(y, dtype=np.int32))

    latents_np = np.concatenate(latents, axis=0)
    labels_np = np.concatenate(labels, axis=0)
    save_latent_dataset_npz(
        out_path=out_path,
        latents=latents_np,
        labels=labels_np,
        class_names=dataset.label_names,
        relpaths=dataset.relpaths(),
    )
    print(f"Saved latent cache: {out_path} with shape={latents_np.shape}")


def train_latent_diffusion(cfg: LatentArtBenchConfig, train_latent_ds: ArtBenchLatentDataset, test_latent_ds: ArtBenchLatentDataset, device):
    latent_shape = train_latent_ds.latents.shape[1:]
    latent_h, latent_w, latent_c = latent_shape
    if latent_h != latent_w:
        raise ValueError(f"Expected square latents, got shape {latent_shape}")

    dm_cfg = base.TrainConfig(
        data_root="",
        model_type=cfg.dm_model_type,
        image_size=int(latent_h),
        in_channels=int(latent_c),
        base_channels=cfg.dm_base_channels,
        channel_mults=cfg.dm_channel_mults,
        num_res_blocks=cfg.dm_num_res_blocks,
        time_emb_dim=cfg.dm_time_emb_dim,
        num_classes=train_latent_ds.num_classes,
        class_cond=cfg.dm_class_cond,
        cond_mode=cfg.dm_cond_mode,
        dropout=cfg.dm_dropout,
        seed=cfg.seed,
        epochs=cfg.dm_epochs,
        batch_size=cfg.dm_batch_size,
        learning_rate=cfg.dm_learning_rate,
        weight_decay=cfg.dm_weight_decay,
        adam_b1=cfg.dm_adam_b1,
        adam_b2=cfg.dm_adam_b2,
        adam_eps=cfg.dm_adam_eps,
        grad_clip_norm=cfg.dm_grad_clip_norm,
        ema_decay=cfg.dm_ema_decay,
        log_every=cfg.dm_log_every,
        timesteps=cfg.dm_timesteps,
        beta_start=cfg.dm_beta_start,
        beta_end=cfg.dm_beta_end,
        predict_x0=cfg.dm_predict_x0,
        prefer_device=cfg.prefer_device,
        use_bfloat16=cfg.use_bfloat16,
        use_data_parallel=False,
        checkpoint_dir=cfg.dm_checkpoint_dir,
        save_every_epochs=1,
        keep_last_k=cfg.keep_last_k,
        resume_from=None,
        num_workers=0,
        use_tqdm=cfg.use_tqdm,
        use_wandb=False,
        wandb_project="",
        wandb_entity=None,
        wandb_run_name=None,
        wandb_mode="disabled",
        wandb_log_step_metrics=False,
    )

    schedule = base.make_diffusion_schedule(dm_cfg.timesteps, dm_cfg.beta_start, dm_cfg.beta_end)
    model = base.build_model(dm_cfg)
    rng = jax.random.PRNGKey(dm_cfg.seed)
    state = base.create_train_state(dm_cfg, model, rng, device)
    train_step = base.make_train_step(schedule, dm_cfg)
    eval_step = base.make_eval_step(schedule, dm_cfg)

    steps_per_epoch = len(train_latent_ds) // dm_cfg.batch_size
    global_step = 0

    for epoch in range(1, dm_cfg.epochs + 1):
        epoch_start = time.time()
        train_iter = train_latent_ds.batch_iterator(dm_cfg.batch_size, shuffle=True, seed=dm_cfg.seed + epoch, drop_last=True)
        train_iter = _iter_with_tqdm(train_iter, steps_per_epoch, f"DM Epoch {epoch}/{dm_cfg.epochs}", dm_cfg.use_tqdm)
        loss_sum = 0.0
        n_steps = 0

        for z, y in train_iter:
            state, metrics = train_step(state, z, y)
            loss_val = float(metrics["loss"])
            loss_sum += loss_val
            n_steps += 1
            global_step += 1
            if global_step % dm_cfg.log_every == 0:
                print(
                    f"[dm] epoch={epoch}/{dm_cfg.epochs} step={global_step} "
                    f"loss={loss_val:.6f}"
                )

        eval_iter = test_latent_ds.batch_iterator(dm_cfg.batch_size, shuffle=False, seed=0, drop_last=False)
        eval_loss = None
        for z, y in eval_iter:
            state, eval_metrics = eval_step(state, z, y)
            eval_loss = float(eval_metrics["loss"])
            break

        mean_loss = loss_sum / max(1, n_steps)
        elapsed = time.time() - epoch_start
        if eval_loss is None:
            print(f"[dm epoch {epoch}/{dm_cfg.epochs}] train_loss={mean_loss:.6f} time={elapsed:.1f}s")
        else:
            print(
                f"[dm epoch {epoch}/{dm_cfg.epochs}] "
                f"train_loss={mean_loss:.6f} eval_loss={eval_loss:.6f} time={elapsed:.1f}s"
            )

        base._save_checkpoint(
            checkpoint_dir=dm_cfg.checkpoint_dir,
            epoch=epoch,
            state=state,
            cfg=dm_cfg,
            keep_last_k=dm_cfg.keep_last_k,
        )

    return state, model, schedule, dm_cfg


def available_artbench_labels(class_names: Sequence[str]) -> list[str]:
    return [str(x) for x in class_names]


def encode_artbench_prompt(
    prompt,
    label_names: Sequence[str],
    cond_mode: str = "class_id",
) -> np.ndarray:
    name_to_id = {name: i for i, name in enumerate(label_names)}

    if cond_mode == "class_id":
        if isinstance(prompt, str):
            prompt = prompt.strip()
            if "," in prompt:
                raise ValueError(
                    "cond_mode='class_id' accepts exactly one class. "
                    "Use cond_mode='multi_hot' for multi-style prompts."
                )
            if prompt.isdigit():
                cid = int(prompt)
            else:
                if prompt not in name_to_id:
                    raise ValueError(f"Unknown ArtBench label: {prompt}. Available labels: {list(label_names)}")
                cid = name_to_id[prompt]
        else:
            cid = int(prompt)

        if cid < 0 or cid >= len(label_names):
            raise ValueError(f"class id {cid} is out of range [0, {len(label_names) - 1}]")
        return np.array(cid, dtype=np.int32)

    if cond_mode == "multi_hot":
        if isinstance(prompt, str):
            tokens = [tok.strip() for tok in prompt.split(",") if tok.strip()]
        else:
            tokens = [str(tok).strip() for tok in prompt]

        if len(tokens) == 0:
            raise ValueError("Empty prompt provided for multi_hot conditioning.")

        vec = np.zeros((len(label_names),), dtype=np.float32)
        for tok in tokens:
            if tok.isdigit():
                cid = int(tok)
                if cid < 0 or cid >= len(label_names):
                    raise ValueError(f"class id {cid} is out of range [0, {len(label_names) - 1}]")
            else:
                if tok not in name_to_id:
                    raise ValueError(f"Unknown ArtBench label: {tok}. Available labels: {list(label_names)}")
                cid = name_to_id[tok]
            vec[cid] = 1.0
        return vec

    raise ValueError("cond_mode must be 'class_id' or 'multi_hot'")


def train(cfg: LatentArtBenchConfig):
    total_train_start = time.time()
    device = base.choose_device(cfg.prefer_device)
    print(f"Using backend={jax.default_backend()}, device={device}")
    print("Config:", asdict(cfg))

    train_ds = ArtBenchImageFolderLatentDataset(
        root=cfg.data_root,
        split=cfg.train_split,
        class_names=cfg.class_names,
        normalize="minus_one_to_one",
        channels_last=True,
        one_hot_labels=False,
        image_size=cfg.image_size,
        resize_mode=cfg.resize_mode,
        file_extensions=cfg.file_extensions,
        exclude_ranges=cfg.train_exclude_ranges,
        exclude_indices=cfg.train_exclude_indices,
        exclude_files=cfg.train_exclude_files,
    )
    test_ds = ArtBenchImageFolderLatentDataset(
        root=cfg.data_root,
        split=cfg.test_split,
        class_names=train_ds.label_names,
        normalize="minus_one_to_one",
        channels_last=True,
        one_hot_labels=False,
        image_size=cfg.image_size,
        resize_mode=cfg.resize_mode,
        file_extensions=cfg.file_extensions,
        exclude_ranges=cfg.test_exclude_ranges,
        exclude_indices=cfg.test_exclude_indices,
        exclude_files=cfg.test_exclude_files,
    )

    print(f"Train images: {len(train_ds)}")
    print(f"Test images: {len(test_ds)}")
    print(f"Class counts: {train_ds.class_counts()}")

    ae_state, ae_model = train_autoencoder(cfg, train_ds, test_ds, device)

    os.makedirs(cfg.cache_dir, exist_ok=True)
    train_cache_path = os.path.join(cfg.cache_dir, "train_latents.npz")
    test_cache_path = os.path.join(cfg.cache_dir, "test_latents.npz")

    encode_split_to_npz(cfg, ae_state, ae_model, train_ds, train_cache_path)
    encode_split_to_npz(cfg, ae_state, ae_model, test_ds, test_cache_path)

    use_one_hot_labels = (cfg.dm_cond_mode == "multi_hot")
    train_latent_ds = ArtBenchLatentDataset(train_cache_path, one_hot_labels=use_one_hot_labels)
    test_latent_ds = ArtBenchLatentDataset(test_cache_path, one_hot_labels=use_one_hot_labels)
    print(f"Latent train shape: {train_latent_ds.latents.shape}")
    print(f"Latent test shape: {test_latent_ds.latents.shape}")

    dm_state, dm_model, schedule, dm_cfg = train_latent_diffusion(cfg, train_latent_ds, test_latent_ds, device)

    total_elapsed = time.time() - total_train_start
    total_h = int(total_elapsed // 3600)
    total_m = int((total_elapsed % 3600) // 60)
    total_s = total_elapsed % 60
    print(
        f"Training finished. Total time: "
        f"{total_h:02d}h {total_m:02d}m {total_s:05.2f}s "
        f"({total_elapsed:.2f} seconds)"
    )

    return {
        "autoencoder_state": ae_state,
        "autoencoder_model": ae_model,
        "dm_state": dm_state,
        "dm_model": dm_model,
        "dm_schedule": schedule,
        "dm_cfg": dm_cfg,
        "train_latent_ds": train_latent_ds,
        "test_latent_ds": test_latent_ds,
    }


if __name__ == "__main__":
    train_exclude_ranges = (
        ("art_nouveau", 0, 100),
        (1, 50, 20),
    )

    train_exclude_indices = {
        "baroque": [0, 10, 25],
        3: [1, 2, 3],
    }

    train_exclude_files = (
        "art_nouveau/a-y-jackson_algoma-in-november-1935.jpg",
        "baroque/some_painting.jpg",
    )

    test_exclude_ranges = None
    test_exclude_indices = None
    test_exclude_files = (
        "expressionism/example.jpg",
    )

    cfg = LatentArtBenchConfig(
        data_root="./databases/artbench-10-imagefolder-split",
        train_split="train",
        test_split="test",
        class_names=None,

        image_size=256,
        resize_mode="shortest_center_crop",
        file_extensions=(".jpg", ".jpeg", ".png", ".webp"),

        train_exclude_ranges=None,  # train_exclude_ranges
        train_exclude_indices=None,  # train_exclude_indices
        train_exclude_files=None,  # train_exclude_files
        test_exclude_ranges=None,  # test_exclude_ranges
        test_exclude_indices=None,  # test_exclude_indices
        test_exclude_files=None,  # test_exclude_files

        ae_base_channels=64,
        ae_epochs=20,
        ae_batch_size=32,
        ae_learning_rate=2e-4,
        ae_weight_decay=1e-4,
        ae_log_every=50,
        latent_channels=4,
        ae_downsample_factor=4,

        dm_model_type="unet",
        dm_cond_mode="multi_hot",#"class_id",
        dm_epochs=100,
        dm_batch_size=128,
        dm_learning_rate=2e-4,
        dm_weight_decay=1e-4,
        dm_grad_clip_norm=1.0,
        dm_ema_decay=0.999,
        dm_log_every=100,
        dm_timesteps=1000,
        dm_beta_start=1e-4,
        dm_beta_end=0.02,
        dm_predict_x0=False,
        dm_base_channels=160,
        dm_channel_mults=(1, 2, 2),
        dm_num_res_blocks=2,
        dm_time_emb_dim=128,
        dm_dropout=0.1,

        seed=0,
        use_bfloat16=True,
        prefer_device="auto",
        use_tqdm=True,
        reuse_autoencoder=True,

        cache_dir="./latents/artbench256",
        autoencoder_model_dir="./models/artbench_latent_autoencoder256",
        dm_checkpoint_dir="./models/artbench_latent_dm_checkpoints256",
        keep_last_k=None,
    )
    train(cfg)
