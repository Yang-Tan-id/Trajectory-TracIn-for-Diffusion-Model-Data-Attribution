import os
import sys
import time
import math
import json
import pickle
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from flax.serialization import from_bytes


# ============================================================
# Utilities
# ============================================================

def tree_scalar_mul(tree, c):
    return jax.tree_util.tree_map(lambda x: x * c, tree)


def tree_add(a, b):
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def tree_sub(a, b):
    return jax.tree_util.tree_map(lambda x, y: x - y, a, b)


def tree_zeros_like(tree):
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def tree_vdot(a, b):
    leaves_a, _ = jax.tree_util.tree_flatten(a)
    leaves_b, _ = jax.tree_util.tree_flatten(b)
    out = jnp.array(0.0, dtype=jnp.float32)
    for x, y in zip(leaves_a, leaves_b):
        out = out + jnp.vdot(x.astype(jnp.float32), y.astype(jnp.float32))
    return out


def tree_mask(tree, mask_tree):
    return jax.tree_util.tree_map(lambda x, m: x if m else jnp.zeros_like(x), tree, mask_tree)


def tree_any(mask_tree):
    leaves, _ = jax.tree_util.tree_flatten(mask_tree)
    return any(bool(x) for x in leaves)


def format_seconds(sec: float) -> str:
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def iter_with_tqdm(iterator, total: int, desc: str, use_tqdm: bool):
    if not use_tqdm:
        return iterator
    try:
        from tqdm.auto import tqdm
    except ImportError as e:
        raise ImportError("use_tqdm=True but tqdm is not installed. Install with: pip install tqdm") from e
    leave = os.environ.get("ATTRIBUTION_TQDM_LEAVE", "1") not in ("0", "false", "False")
    mininterval = float(os.environ.get("ATTRIBUTION_TQDM_MININTERVAL", "1"))
    return tqdm(
        iterator,
        total=total,
        desc=desc,
        leave=leave,
        dynamic_ncols=True,
        file=sys.stdout,
        mininterval=mininterval,
        maxinterval=max(30.0, mininterval),
    )


def tree_to_device(tree, device):
    return jax.tree_util.tree_map(lambda x: jax.device_put(x, device), tree)


def schedule_to_device(schedule: "DiffusionSchedule", device) -> "DiffusionSchedule":
    return DiffusionSchedule(
        betas=jax.device_put(schedule.betas, device),
        alphas=jax.device_put(schedule.alphas, device),
        alphas_cumprod=jax.device_put(schedule.alphas_cumprod, device),
        sqrt_alphas_cumprod=jax.device_put(schedule.sqrt_alphas_cumprod, device),
        sqrt_one_minus_alphas_cumprod=jax.device_put(schedule.sqrt_one_minus_alphas_cumprod, device),
    )


def array_to_device(x, device):
    if x is None:
        return None
    return jax.device_put(x, device)


def array_device_str(x) -> str:
    if x is None:
        return "None"
    try:
        dev = x.device
        if callable(dev):
            dev = dev()
        return str(dev)
    except Exception:
        pass
    try:
        return str(x.devices())
    except Exception:
        return "unknown"


def first_leaf_device_str(tree) -> str:
    leaves, _ = jax.tree_util.tree_flatten(tree)
    if not leaves:
        return "empty"
    return array_device_str(leaves[0])


def list_checkpoints_sorted(checkpoint_dir: str, suffix: str = ".ckpt") -> List[str]:
    paths = []
    if not os.path.isdir(checkpoint_dir):
        return paths
    for name in os.listdir(checkpoint_dir):
        if name.endswith(suffix):
            paths.append(os.path.join(checkpoint_dir, name))
    paths.sort()
    return paths


def latest_checkpoint_in_dir(checkpoint_dir: str, suffix: str = ".ckpt") -> Optional[str]:
    paths = list_checkpoints_sorted(checkpoint_dir, suffix=suffix)
    return paths[-1] if paths else None


def normalize_query_tokens(query_spec) -> List[str]:
    if isinstance(query_spec, str):
        return [tok.strip() for tok in query_spec.split(",") if tok.strip()]
    if isinstance(query_spec, (list, tuple)):
        return [str(tok).strip() for tok in query_spec if str(tok).strip()]
    return [str(query_spec).strip()]


def encode_cifar_query(
    query,
    label_names: Sequence[str],
    cond_mode: str = "class_id",
) -> np.ndarray:
    name_to_id = {name: i for i, name in enumerate(label_names)}

    if cond_mode == "class_id":
        if isinstance(query, str):
            q = query.strip()
            if "," in q:
                raise ValueError(
                    "cond_mode='class_id' accepts exactly one class. "
                    "Use cond_mode='multi_hot' for multi-label queries."
                )
            if q.isdigit():
                cid = int(q)
            else:
                if q not in name_to_id:
                    raise ValueError(
                        f"Unknown CIFAR label: {q}. Available labels: {list(label_names)}"
                    )
                cid = name_to_id[q]
        else:
            cid = int(query)

        if cid < 0 or cid >= len(label_names):
            raise ValueError(f"class id {cid} is out of range [0, {len(label_names) - 1}]")
        return np.array(cid, dtype=np.int32)

    if cond_mode == "multi_hot":
        if isinstance(query, str):
            tokens = [tok.strip() for tok in query.split(",") if tok.strip()]
        else:
            tokens = [str(tok).strip() for tok in query]

        if len(tokens) == 0:
            raise ValueError("Empty query provided for multi_hot conditioning.")

        vec = np.zeros((len(label_names),), dtype=np.float32)
        for tok in tokens:
            if tok.isdigit():
                cid = int(tok)
                if cid < 0 or cid >= len(label_names):
                    raise ValueError(f"class id {cid} is out of range [0, {len(label_names) - 1}]")
            else:
                if tok not in name_to_id:
                    raise ValueError(
                        f"Unknown CIFAR label: {tok}. Available labels: {list(label_names)}"
                    )
                cid = name_to_id[tok]
            vec[cid] = 1.0
        return vec

    raise ValueError("cond_mode must be 'class_id' or 'multi_hot'")


# ============================================================
# Shared diffusion helpers
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


def predict_x0_from_eps(schedule: DiffusionSchedule, xt: jnp.ndarray, t: jnp.ndarray, eps: jnp.ndarray):
    return (
        xt - extract(schedule.sqrt_one_minus_alphas_cumprod, t, xt.shape) * eps
    ) / extract(schedule.sqrt_alphas_cumprod, t, xt.shape)


def ddim_step_from_eps(
    eps_fn,
    params,
    schedule: DiffusionSchedule,
    x_t: jnp.ndarray,
    t_idx: int,
    t_prev_idx: int,
    cond,
):
    t = jnp.full((x_t.shape[0],), t_idx, dtype=jnp.int32)
    eps = eps_fn(params, x_t, t, cond)

    abar_t = schedule.alphas_cumprod[t_idx]
    abar_prev = jnp.array(1.0, dtype=jnp.float32) if t_prev_idx < 0 else schedule.alphas_cumprod[t_prev_idx]

    x0_pred = (x_t - jnp.sqrt(1.0 - abar_t) * eps) / jnp.sqrt(abar_t)
    x_prev = jnp.sqrt(abar_prev) * x0_pred + jnp.sqrt(1.0 - abar_prev) * eps
    return x_prev, x0_pred, eps


def compute_reference_endpoint_ddim(
    eps_fn,
    params,
    schedule: DiffusionSchedule,
    cond,
    shape: Tuple[int, ...],
    seed: int,
    ddim_steps: int,
    device=None,
):
    rng = jax.random.PRNGKey(seed)
    if device is not None:
        rng = jax.device_put(rng, device)
    x = jax.random.normal(rng, shape, dtype=jnp.float32)
    if device is not None:
        x = jax.device_put(x, device)

    T = int(schedule.betas.shape[0])
    ddim_ts = np.linspace(T - 1, 0, ddim_steps, dtype=np.int32)

    for pos, t_idx in enumerate(ddim_ts):
        t_prev_idx = int(ddim_ts[pos + 1]) if pos + 1 < len(ddim_ts) else -1
        x, _, _ = ddim_step_from_eps(eps_fn, params, schedule, x, int(t_idx), t_prev_idx, cond)

    return x


# ============================================================
# Parameter mask helpers (baseline vs LoRA)
# ============================================================

def flatten_keys(tree, prefix=()):
    out = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            out.update(flatten_keys(v, prefix + (str(k),)))
    else:
        out[prefix] = tree
    return out


def build_param_mask(params, mode: str):
    """
    mode:
      - 'all'       : all params active
      - 'baseline'  : non-LoRA params active
      - 'lora'      : only LoRA params active
    """
    flat = flatten_keys(params)
    out = {}

    def insert(d, key_tuple, value):
        cur = d
        for k in key_tuple[:-1]:
            if k not in cur:
                cur[k] = {}
            cur = cur[k]
        cur[key_tuple[-1]] = value

    for key in flat.keys():
        key_str = "/".join(key).lower()
        is_lora = ("lora" in key_str)

        if mode == "all":
            val = True
        elif mode == "baseline":
            val = not is_lora
        elif mode == "lora":
            val = is_lora
        else:
            raise ValueError("mode must be 'all', 'baseline', or 'lora'")

        insert(out, key, val)

    return out


# ============================================================
# Task adapters
# ============================================================

class BaseTaskAdapter:
    def __init__(self, module):
        self.m = module

    def choose_device(self, prefer_device: str):
        if hasattr(self.m, "choose_device"):
            return self.m.choose_device(prefer_device)
        if hasattr(self.m, "base") and hasattr(self.m.base, "choose_device"):
            return self.m.base.choose_device(prefer_device)
        raise AttributeError(f"Module '{self.m.__name__}' does not provide choose_device(...).")

    def build_state_template(self, cfg, model, device):
        raise NotImplementedError

    def restore_state(self, ckpt_path: str, state_template):
        with open(ckpt_path, "rb") as f:
            payload = pickle.load(f)
        state = from_bytes(state_template, payload["state_bytes"])
        return state, payload

    def build_model(self, cfg):
        return self.m.build_model(cfg)

    def iter_dataset(self, cfg):
        raise NotImplementedError

    def get_example_batch(self, ds):
        raise NotImplementedError

    def get_item(self, ds, idx):
        raise NotImplementedError

    def eps_apply(self, model, params, x, t, cond):
        raise NotImplementedError

    def make_query_cond(self, ds, query_spec, cfg):
        raise NotImplementedError

    def train_loss_mc(self, model, params, schedule, x0, cond, num_mc_samples: int, rng):
        raise NotImplementedError


class X3TaskAdapter(BaseTaskAdapter):
    def iter_dataset(self, cfg):
        ds = self.m.ColorGridDatasetJAX(
            csv_path=cfg.csv_path,
            grid_size=cfg.grid_size,
            fixed_s=cfg.fixed_s,
            fixed_v=cfg.fixed_v,
            label_start=cfg.label_start,
            row_indices=cfg.row_indices,
            subset_ranges=cfg.subset_ranges,
        )
        return ds

    def get_example_batch(self, ds):
        x, y = ds[0]
        return x[None, ...], y[None, ...]

    def get_item(self, ds, idx):
        x, y = ds[idx]
        x = jnp.array(x[None, ...], dtype=jnp.float32)
        cond = jnp.array(y[None, ...], dtype=jnp.float32)
        return x, cond

    def build_state_template(self, cfg, model, device):
        cond_dim = len(self.iter_dataset(cfg).vocab)
        rng = jax.random.PRNGKey(cfg.seed)
        return self.m.create_train_state(cfg, model, rng, device, cond_dim)

    def eps_apply(self, model, params, x, t, cond):
        return model.apply({"params": params}, x, t, cond, train=False)

    def make_query_cond(self, ds, query_spec, cfg):
        vec = np.zeros((len(ds.vocab),), dtype=np.float32)
        tokens = normalize_query_tokens(query_spec)
        missing = [lab for lab in tokens if lab not in ds.vocab]
        if missing:
            raise KeyError(f"Missing labels in x3 vocab: {missing}")
        for lab in tokens:
            vec[ds.vocab[lab]] = 1.0
        return jnp.array(vec[None, :], dtype=jnp.float32)

    def train_loss_mc(self, model, params, schedule, x0, cond, num_mc_samples: int, rng):
        losses = []
        local_rng = rng
        for _ in range(int(num_mc_samples)):
            local_rng, noise_rng, t_rng = jax.random.split(local_rng, 3)
            t = jax.random.randint(t_rng, (x0.shape[0],), 0, schedule.betas.shape[0], dtype=jnp.int32)
            noise = jax.random.normal(noise_rng, x0.shape, dtype=x0.dtype)
            xt = q_sample(schedule, x0, t, noise)
            pred = model.apply({"params": params}, xt, t, cond, train=False)
            losses.append(jnp.mean((pred - noise) ** 2))
        return jnp.mean(jnp.stack(losses))


class CIFAR10TaskAdapter(BaseTaskAdapter):
    def iter_dataset(self, cfg):
        ds = self.m.CIFAR10Dataset(
            root=cfg.data_root,
            batch_names=cfg.batch_names,
            use_test=cfg.use_test,
            class_names=cfg.class_names,
            normalize="minus_one_to_one",
            channels_last=True,
            exclude_ranges=cfg.exclude_ranges,
            exclude_indices=cfg.exclude_indices,
            cond_mode=cfg.cond_mode,
        )
        return ds

    def get_example_batch(self, ds):
        x = jnp.array(ds.images[0:1], dtype=jnp.float32)
        if ds.labels.ndim == 1:
            y = jnp.array(ds.labels[0:1], dtype=jnp.int32)
        else:
            y = jnp.array(ds.labels[0:1], dtype=jnp.float32)
        return x, y

    def get_item(self, ds, idx):
        x = jnp.array(ds.images[idx:idx + 1], dtype=jnp.float32)
        if ds.labels.ndim == 1:
            cond = jnp.array(ds.labels[idx:idx + 1], dtype=jnp.int32)
        else:
            cond = jnp.array(ds.labels[idx:idx + 1], dtype=jnp.float32)
        return x, cond

    def build_state_template(self, cfg, model, device):
        rng = jax.random.PRNGKey(cfg.seed)
        return self.m.create_train_state(cfg, model, rng, device)

    def eps_apply(self, model, params, x, t, cond):
        return model.apply({"params": params}, x, t, cond, train=False)

    def make_query_cond(self, ds, query_spec, cfg):
        if not cfg.class_cond:
            if cfg.cond_mode == "class_id":
                return jnp.zeros((1,), dtype=jnp.int32)
            return jnp.zeros((1, cfg.num_classes), dtype=jnp.float32)
        q = encode_cifar_query(
            query=query_spec,
            label_names=ds.label_names,
            cond_mode=cfg.cond_mode,
        )
        if cfg.cond_mode == "class_id":
            return jnp.array([int(q)], dtype=jnp.int32)
        if cfg.cond_mode == "multi_hot":
            return jnp.array(q[None, :], dtype=jnp.float32)
        raise ValueError("cond_mode must be 'class_id' or 'multi_hot'")

    def train_loss_mc(self, model, params, schedule, x0, cond, num_mc_samples: int, rng):
        losses = []
        local_rng = rng
        for _ in range(int(num_mc_samples)):
            local_rng, noise_rng, t_rng = jax.random.split(local_rng, 3)
            t = jax.random.randint(t_rng, (x0.shape[0],), 0, schedule.betas.shape[0], dtype=jnp.int32)
            noise = jax.random.normal(noise_rng, x0.shape, dtype=x0.dtype)
            xt = q_sample(schedule, x0, t, noise)
            pred = model.apply({"params": params}, xt, t, cond, train=False)
            losses.append(jnp.mean((pred - noise) ** 2))
        return jnp.mean(jnp.stack(losses))


class ArtBenchLatentTaskAdapter(BaseTaskAdapter):
    def iter_dataset(self, cfg):
        use_one_hot = (cfg.cond_mode == "multi_hot")
        npz_path = cfg.latent_npz_path
        if npz_path is None:
            npz_path = os.path.join(cfg.cache_dir, "train_latents.npz")
        return self.m.ArtBenchLatentDataset(
            npz_path=npz_path,
            one_hot_labels=use_one_hot,
            class_names=cfg.class_names,
            exclude_indices=cfg.latent_exclude_indices,
        )

    def get_example_batch(self, ds):
        x, y = ds[0]
        x = jnp.array(x[None, ...], dtype=jnp.float32)
        if y.ndim == 0:
            y = jnp.array(y[None], dtype=jnp.int32)
        else:
            y = jnp.array(y[None, ...], dtype=jnp.float32)
        return x, y

    def get_item(self, ds, idx):
        x, y = ds[idx]
        x = jnp.array(x[None, ...], dtype=jnp.float32)
        if y.ndim == 0:
            cond = jnp.array(y[None], dtype=jnp.int32)
        else:
            cond = jnp.array(y[None, ...], dtype=jnp.float32)
        return x, cond

    def build_state_template(self, cfg, model, device):
        rng = jax.random.PRNGKey(cfg.seed)
        return self.m.base.create_train_state(cfg, model, rng, device)

    def build_model(self, cfg):
        return self.m.base.build_model(cfg)

    def eps_apply(self, model, params, x, t, cond):
        return model.apply({"params": params}, x, t, cond, train=False)

    def make_query_cond(self, ds, query_spec, cfg):
        if not cfg.class_cond:
            if cfg.cond_mode == "class_id":
                return jnp.zeros((1,), dtype=jnp.int32)
            return jnp.zeros((1, cfg.num_classes), dtype=jnp.float32)
        q = self.m.encode_artbench_prompt(
            prompt=query_spec,
            label_names=ds.label_names,
            cond_mode=cfg.cond_mode,
        )
        if cfg.cond_mode == "class_id":
            return jnp.array([int(q)], dtype=jnp.int32)
        if cfg.cond_mode == "multi_hot":
            return jnp.array(q[None, :], dtype=jnp.float32)
        raise ValueError("cond_mode must be 'class_id' or 'multi_hot'")

    def train_loss_mc(self, model, params, schedule, x0, cond, num_mc_samples: int, rng):
        losses = []
        local_rng = rng
        for _ in range(int(num_mc_samples)):
            local_rng, noise_rng, t_rng = jax.random.split(local_rng, 3)
            t = jax.random.randint(t_rng, (x0.shape[0],), 0, schedule.betas.shape[0], dtype=jnp.int32)
            noise = jax.random.normal(noise_rng, x0.shape, dtype=x0.dtype)
            xt = q_sample(schedule, x0, t, noise)
            pred = model.apply({"params": params}, xt, t, cond, train=False)
            losses.append(jnp.mean((pred - noise) ** 2))
        return jnp.mean(jnp.stack(losses))


# ============================================================
# Attribution config
# ============================================================

@dataclass
class EndpointTraceInConfig:
    task_type: str                   # 'x3', 'cifar10', or 'artbench_latent'
    module_name: str                 # e.g. 'x3_training_jax' or 'cifar10_training_jax'

    # model dirs
    baseline_dir: Optional[str] = None
    lora_update_dir: Optional[str] = None
    reference_ckpt: Optional[str] = None

    # which ckpts to use
    use_baseline_ckpts: bool = True
    use_lora_ckpts: bool = False
    checkpoint_limit: int = -1

    # query
    query: Any = None
    seed: int = 0

    # optional precomputed sampler input
    # Accepts either:
    #   - a seed directory containing final_state.npy / seed_info.json, or
    #   - a sampler run root containing manifest.json and seed_* directories.
    attribution_sample_dir: Optional[str] = None
    attribution_sample_seed: Optional[int] = None
    attribution_sample_index: int = 0
    attribution_use_trajectory_endpoint: bool = True

    # endpoint generation
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    ddim_steps: int = 1000
    eta: float = 0.0  # kept for compatibility, not used in deterministic DDIM here

    # endpoint-anchored loss window
    t_min_end: int = 0
    t_max_end_frac: float = 0.2

    # Monte Carlo controls
    endpoint_mc_samples: int = 10
    train_mc_samples: int = 10

    # scoring set
    max_train_points: int = 1024
    random_subset: bool = True
    score_index_ranges: Optional[Tuple[Tuple[int, int], ...]] = None
    score_index_base: int = 1
    score_batch_size: int = 32
    topk: int = 100

    # output
    out_dir: str = "./endpoint_tracein_out"
    use_tqdm: bool = True

    # x3 fields
    csv_path: Optional[str] = None
    grid_size: int = 3
    fixed_s: float = 0.9
    fixed_v: float = 0.9
    label_start: Optional[int] = None
    row_indices: Optional[Tuple[int, ...]] = None
    subset_ranges: Optional[Tuple[Tuple[int, int], ...]] = None
    image_size: int = 3
    in_channels: int = 3
    base_channels: int = 160
    time_emb_dim: int = 128
    class_cond: bool = True
    dropout: float = 0.1
    predict_x0: bool = False
    prefer_device: str = "gpu"
    epochs: int = 1
    batch_size: int = 128
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    ema_decay: float = 0.999
    log_every: int = 100
    use_bfloat16: bool = False
    keep_last_k: Optional[int] = None
    resume_from: Optional[str] = None

    # cifar fields
    data_root: Optional[str] = None
    batch_names: Optional[Tuple[str, ...]] = None
    class_names: Optional[Tuple[str, ...]] = None
    use_test: bool = False
    exclude_ranges: Optional[Tuple[Tuple[int, int, int], ...]] = None
    exclude_indices: Optional[Dict[int, Tuple[int, ...]]] = None
    model_type: str = "unet"
    channel_mults: Tuple[int, ...] = (1, 2, 2)
    num_res_blocks: int = 2
    num_classes: int = 10
    num_workers: int = 0
    cond_mode: str = "class_id"      # for CIFAR: 'class_id' or 'multi_hot'

    # artbench_latent fields
    latent_npz_path: Optional[str] = None
    cache_dir: str = "./latents/artbench256"
    latent_exclude_indices: Optional[Tuple[int, ...]] = None


def get_adapter(cfg: EndpointTraceInConfig):
    module = __import__(cfg.module_name)
    if cfg.task_type == "x3":
        return X3TaskAdapter(module)
    if cfg.task_type in ("cifar10", "cifar"):
        return CIFAR10TaskAdapter(module)
    if cfg.task_type == "artbench_latent":
        return ArtBenchLatentTaskAdapter(module)
    raise ValueError("task_type must be 'x3', 'cifar10', or 'artbench_latent'")


CHECKPOINT_CONFIG_FIELDS = (
    "model_type",
    "image_size",
    "in_channels",
    "base_channels",
    "channel_mults",
    "num_res_blocks",
    "time_emb_dim",
    "num_classes",
    "class_cond",
    "cond_mode",
    "dropout",
    "timesteps",
    "beta_start",
    "beta_end",
    "predict_x0",
    "use_bfloat16",
)


def load_checkpoint_config(ckpt_path: Optional[str]) -> Dict[str, Any]:
    if not ckpt_path:
        return {}
    ckpt_path = os.path.abspath(os.path.expanduser(str(ckpt_path)))
    if not os.path.isfile(ckpt_path):
        return {}
    with open(ckpt_path, "rb") as f:
        payload = pickle.load(f)
    cfg_dict = payload.get("config", {})
    return dict(cfg_dict) if isinstance(cfg_dict, dict) else {}


def apply_checkpoint_config(cfg: EndpointTraceInConfig, ckpt_path: Optional[str]):
    ckpt_cfg = load_checkpoint_config(ckpt_path)
    if not ckpt_cfg:
        return

    changed = []
    for name in CHECKPOINT_CONFIG_FIELDS:
        if name not in ckpt_cfg:
            continue
        old = getattr(cfg, name, None)
        new = ckpt_cfg[name]
        if old != new:
            setattr(cfg, name, new)
            changed.append(name)

    if changed:
        print(
            f"[setup] synced model config from checkpoint {os.path.basename(str(ckpt_path))}: "
            f"{', '.join(changed)}"
        )


def _find_attribution_seed_dir(sample_dir: str, seed: Optional[int]) -> str:
    sample_dir = os.path.abspath(os.path.expanduser(sample_dir))
    if (
        os.path.isfile(os.path.join(sample_dir, "final_state.npy"))
        or os.path.isfile(os.path.join(sample_dir, "trajectory_xt.npy"))
    ):
        return sample_dir

    if seed is not None:
        candidate = os.path.join(sample_dir, f"seed_{int(seed):06d}")
        if (
            os.path.isfile(os.path.join(candidate, "final_state.npy"))
            or os.path.isfile(os.path.join(candidate, "trajectory_xt.npy"))
        ):
            return candidate
        raise FileNotFoundError(f"No final_state.npy or trajectory_xt.npy found for seed {seed}: {candidate}")

    seed_dirs = []
    if os.path.isdir(sample_dir):
        for name in os.listdir(sample_dir):
            path = os.path.join(sample_dir, name)
            if name.startswith("seed_") and (
                os.path.isfile(os.path.join(path, "final_state.npy"))
                or os.path.isfile(os.path.join(path, "trajectory_xt.npy"))
            ):
                seed_dirs.append(path)
    seed_dirs.sort()
    if not seed_dirs:
        raise FileNotFoundError(
            "Could not find final_state.npy or trajectory_xt.npy. "
            "Pass either a seed_* directory or a sampler run root."
        )
    return seed_dirs[0]


def _load_json_if_exists(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def load_attribution_endpoint(cfg: EndpointTraceInConfig) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    if cfg.attribution_sample_dir is None:
        raise ValueError("attribution_sample_dir is required.")

    seed_dir = _find_attribution_seed_dir(cfg.attribution_sample_dir, cfg.attribution_sample_seed)
    final_path = os.path.join(seed_dir, "final_state.npy")
    trajectory_path = os.path.join(seed_dir, "trajectory_xt.npy")
    t_path = os.path.join(seed_dir, "trajectory_t.npy")

    source = "final_state.npy"
    if os.path.isfile(final_path):
        final_state = np.load(final_path)
    elif cfg.attribution_use_trajectory_endpoint and os.path.isfile(trajectory_path):
        trajectory = np.load(trajectory_path)
        if trajectory.ndim != 5:
            raise ValueError(
                f"Expected trajectory_xt.npy to have shape (K,B,H,W,C), got {trajectory.shape}"
            )
        if os.path.isfile(t_path):
            t_seq = np.load(t_path)
            matches = np.where(t_seq == 0)[0]
            traj_idx = int(matches[-1]) if len(matches) else -1
        else:
            t_seq = None
            traj_idx = -1
        final_state = trajectory[traj_idx]
        source = f"trajectory_xt.npy[{traj_idx}]"
    else:
        raise FileNotFoundError(
            f"No final_state.npy found at {final_path}. "
            "Set attribution_use_trajectory_endpoint=True to fall back to trajectory_xt.npy."
        )

    if final_state.ndim != 4:
        raise ValueError(f"Expected endpoint state to have shape (B,H,W,C), got {final_state.shape}")

    sample_idx = int(cfg.attribution_sample_index)
    if sample_idx < 0 or sample_idx >= final_state.shape[0]:
        raise IndexError(
            f"attribution_sample_index={sample_idx} is out of range for batch size {final_state.shape[0]}"
        )

    x0_ref_np = final_state[sample_idx:sample_idx + 1].astype(np.float32)
    seed_info = _load_json_if_exists(os.path.join(seed_dir, "seed_info.json"))
    run_root = os.path.dirname(seed_dir)
    manifest = _load_json_if_exists(os.path.join(run_root, "manifest.json"))

    meta = {
        "seed_dir": seed_dir,
        "final_state_path": final_path,
        "trajectory_xt_path": trajectory_path if os.path.isfile(trajectory_path) else None,
        "source": source,
        "sample_index": sample_idx,
        "final_state_shape": list(final_state.shape),
        "loaded_x0_ref_shape": list(x0_ref_np.shape),
        "seed_info": seed_info,
        "manifest": manifest,
    }
    return jnp.asarray(x0_ref_np, dtype=jnp.float32), meta


def infer_query_from_attribution_meta(meta: Dict[str, Any]) -> Optional[Any]:
    seed_info = meta.get("seed_info") or {}
    if seed_info.get("prompt") is not None:
        return seed_info["prompt"]
    manifest = meta.get("manifest") or {}
    if manifest.get("prompt") is not None:
        return manifest["prompt"]
    return None


def repeat_condition_to_batch(cond, batch_size: int):
    if cond is None:
        return None
    batch_size = int(batch_size)
    if batch_size <= 1 or cond.shape[0] == batch_size:
        return cond
    if cond.shape[0] != 1:
        raise ValueError(f"Cannot repeat condition with leading dim {cond.shape[0]} to batch {batch_size}")
    reps = (batch_size,) + (1,) * (cond.ndim - 1)
    return jnp.tile(cond, reps)


def resolve_checkpoint_path_from_manifest(manifest_ckpt: Optional[str]) -> Optional[str]:
    if not manifest_ckpt:
        return None

    ckpt_path = os.path.abspath(os.path.expanduser(str(manifest_ckpt)))
    if os.path.isfile(ckpt_path):
        return ckpt_path

    basename = os.path.basename(str(manifest_ckpt))
    if not basename:
        return None

    search_roots = [
        os.getcwd(),
        os.path.join(os.getcwd(), "diffusion model_jax", "models"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
    ]

    matches = []
    seen_roots = set()
    for root in search_roots:
        root = os.path.abspath(root)
        if root in seen_roots or not os.path.isdir(root):
            continue
        seen_roots.add(root)
        for dirpath, _, filenames in os.walk(root):
            if basename in filenames:
                matches.append(os.path.join(dirpath, basename))

    matches = sorted(set(matches))
    if not matches:
        return None
    return matches[0]


# ============================================================
# Endpoint anchored loss + score
# ============================================================

def denoising_loss_mc_vectorized(
    adapter,
    model,
    params,
    schedule,
    x0,
    cond,
    *,
    num_mc_samples: int,
    rng,
    t_min: int = 0,
    t_max: Optional[int] = None,
):
    T = int(schedule.betas.shape[0])
    if t_max is None:
        t_max = T - 1
    t_min = max(0, min(T - 1, int(t_min)))
    t_max = max(0, min(T - 1, int(t_max)))
    if t_max < t_min:
        t_max = t_min

    num_mc_samples = int(num_mc_samples)
    if num_mc_samples <= 0:
        raise ValueError(f"num_mc_samples must be positive, got {num_mc_samples}")

    B = int(x0.shape[0])
    rng, noise_rng, t_rng = jax.random.split(rng, 3)
    t = jax.random.randint(
        t_rng,
        (num_mc_samples, B),
        minval=t_min,
        maxval=t_max + 1,
        dtype=jnp.int32,
    )
    noise = jax.random.normal(noise_rng, (num_mc_samples,) + tuple(x0.shape), dtype=x0.dtype)

    x0_rep = jnp.broadcast_to(x0[None, ...], (num_mc_samples,) + tuple(x0.shape))
    x0_flat = x0_rep.reshape((num_mc_samples * B,) + tuple(x0.shape[1:]))
    noise_flat = noise.reshape((num_mc_samples * B,) + tuple(x0.shape[1:]))
    t_flat = t.reshape((num_mc_samples * B,))

    if cond is None:
        cond_flat = None
    else:
        cond_rep = jnp.broadcast_to(cond[None, ...], (num_mc_samples,) + tuple(cond.shape))
        cond_flat = cond_rep.reshape((num_mc_samples * B,) + tuple(cond.shape[1:]))

    xt = q_sample(schedule, x0_flat, t_flat, noise_flat)
    pred = adapter.eps_apply(model, params, xt, t_flat, cond_flat)
    return jnp.mean((pred - noise_flat) ** 2)

def endpoint_anchored_loss_mc(
    adapter,
    model,
    params,
    schedule,
    x0_ref,
    cond,
    *,
    t_min: int = 0,
    t_max: Optional[int] = None,
    num_mc_samples: int = 8,
    rng,
):
    return denoising_loss_mc_vectorized(
        adapter=adapter,
        model=model,
        params=params,
        schedule=schedule,
        x0=x0_ref,
        cond=cond,
        num_mc_samples=num_mc_samples,
        rng=rng,
        t_min=t_min,
        t_max=t_max,
    )


def compute_g_end(
    adapter,
    model,
    params,
    active_mask,
    schedule,
    x0_ref,
    query_cond,
    t_min_end,
    t_max_end,
    endpoint_mc_samples,
    rng,
):
    def loss_fn(p):
        return endpoint_anchored_loss_mc(
            adapter=adapter,
            model=model,
            params=p,
            schedule=schedule,
            x0_ref=x0_ref,
            cond=query_cond,
            t_min=t_min_end,
            t_max=t_max_end,
            num_mc_samples=endpoint_mc_samples,
            rng=rng,
        )

    value_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    L_end, g_end = value_and_grad_fn(params)
    g_end = tree_mask(g_end, active_mask)
    return g_end, L_end


def score_one_trainpoint_given_gend(
    adapter,
    model,
    params,
    active_mask,
    schedule,
    g_end,
    x0_train,
    train_cond,
    *,
    eta_k=1.0,
    train_mc_samples=10,
    rng,
):
    def loss_fn(p):
        return denoising_loss_mc_vectorized(
            adapter=adapter,
            model=model,
            params=p,
            schedule=schedule,
            x0=x0_train,
            cond=train_cond,
            num_mc_samples=train_mc_samples,
            rng=rng,
        )

    value_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    L_tr, g_tr = value_and_grad_fn(params)
    g_tr = tree_mask(g_tr, active_mask)

    sc = eta_k * tree_vdot(g_end, g_tr)
    return sc, L_tr


def make_train_batch(adapter, ds, indices: Sequence[int], device):
    xs = []
    conds = []
    for idx in indices:
        x, cond = adapter.get_item(ds, int(idx))
        xs.append(np.asarray(x[0], dtype=np.float32))
        cond_np = np.asarray(cond)
        conds.append(cond_np[0])

    x_batch = jnp.asarray(np.stack(xs, axis=0), dtype=jnp.float32)
    cond_arr = np.stack(conds, axis=0)
    if cond_arr.ndim == 1:
        cond_batch = jnp.asarray(cond_arr, dtype=jnp.int32)
    else:
        cond_batch = jnp.asarray(cond_arr, dtype=jnp.float32)
    return array_to_device(x_batch, device), array_to_device(cond_batch, device)


def pad_indices_to_batch(indices: Sequence[int], batch_size: int) -> List[int]:
    out = [int(i) for i in indices]
    if not out:
        return out
    while len(out) < int(batch_size):
        out.append(out[-1])
    return out


def make_score_train_batch_fn(
    adapter,
    model,
    active_mask,
    schedule,
    g_end,
    *,
    eta_k=1.0,
    train_mc_samples=10,
):
    def one_score(params, x0_one, cond_one, rng_one):
        x0_one = x0_one[None, ...]
        if cond_one.ndim == 0:
            cond_one = cond_one[None]
        else:
            cond_one = cond_one[None, ...]

        def loss_fn(p):
            return denoising_loss_mc_vectorized(
                adapter=adapter,
                model=model,
                params=p,
                schedule=schedule,
                x0=x0_one,
                cond=cond_one,
                num_mc_samples=train_mc_samples,
                rng=rng_one,
            )

        L_tr, g_tr = jax.value_and_grad(loss_fn)(params)
        g_tr = tree_mask(g_tr, active_mask)
        sc = eta_k * tree_vdot(g_end, g_tr)
        return sc, L_tr

    score_fn = jax.jit(jax.vmap(one_score, in_axes=(None, 0, 0, 0)))

    def score_train_batch(params, x0_batch, cond_batch, rng):
        keys = jax.random.split(rng, x0_batch.shape[0])
        return score_fn(params, x0_batch, cond_batch, keys)

    return score_train_batch


# ============================================================
# Candidate selection
# ============================================================

def build_candidate_items(cfg, N: int) -> List[int]:
    if cfg.score_index_ranges is not None:
        if int(cfg.score_index_base) not in (0, 1):
            raise ValueError("score_index_base must be 0 or 1.")

        picked = []
        seen = set()
        offset = int(cfg.score_index_base)
        for start, end in cfg.score_index_ranges:
            start = int(start)
            end = int(end)
            if end < start:
                raise ValueError(f"Invalid score index range ({start}, {end}): end < start.")
            start0 = start - offset
            end0 = end - offset
            if start0 < 0 or end0 >= N:
                raise ValueError(
                    f"Score index range ({start}, {end}) with base={offset} is out of dataset bounds. "
                    f"Valid 0-based indices are [0, {N - 1}]."
                )
            for idx in range(start0, end0 + 1):
                if idx not in seen:
                    picked.append(idx)
                    seen.add(idx)
        if not picked:
            raise ValueError("score_index_ranges produced no candidate indices.")
        return [int(i) for i in picked]

    M_req = min(int(cfg.max_train_points), N)

    if cfg.random_subset:
        rng = np.random.default_rng(cfg.seed)
        picked = rng.choice(N, size=M_req, replace=False).tolist()
    else:
        picked = list(range(M_req))

    return [int(i) for i in picked]


def score_subset_suffix(cfg) -> str:
    if cfg.score_index_ranges is not None:
        parts = []
        for start, end in cfg.score_index_ranges:
            parts.append(f"{int(start)}_{int(end)}")
        return "range_" + "__".join(parts)
    if cfg.random_subset:
        return f"random_seed{int(cfg.seed)}_n{int(cfg.max_train_points)}"
    return f"first_n{int(cfg.max_train_points)}"


def apply_score_subset_suffix_to_out_dir(cfg):
    suffix = score_subset_suffix(cfg)
    base = os.path.normpath(cfg.out_dir)
    tail = os.path.basename(base)
    if tail == suffix or tail.endswith("_" + suffix):
        return suffix
    cfg.out_dir = f"{cfg.out_dir}_{suffix}"
    return suffix


# ============================================================
# Main run
# ============================================================

def run_endpoint_tracein(cfg: EndpointTraceInConfig):
    stage_mode = os.environ.get("END_TRACIN_STAGE_MODE", "").strip().lower()
    stage_artifact_path = os.environ.get("END_TRACIN_STAGE_ARTIFACT_PATH")
    if stage_mode and stage_mode not in ("train", "query"):
        raise ValueError("END_TRACIN_STAGE_MODE must be unset, 'train', or 'query'.")
    if stage_mode and not stage_artifact_path:
        raise ValueError("END_TRACIN_STAGE_ARTIFACT_PATH is required when END_TRACIN_STAGE_MODE is set.")
    out_suffix = apply_score_subset_suffix_to_out_dir(cfg)
    ensure_dir(cfg.out_dir)
    t0 = time.perf_counter()

    precomputed_sample_meta = None
    x0_ref = None
    manifest_ckpt = None
    resolved_manifest_ckpt = None
    if cfg.attribution_sample_dir is not None:
        print(f"[setup] loading precomputed attribution sample: {cfg.attribution_sample_dir}")
        x0_ref, precomputed_sample_meta = load_attribution_endpoint(cfg)
        inferred_query = infer_query_from_attribution_meta(precomputed_sample_meta)
        if inferred_query is None:
            if cfg.query is None:
                raise ValueError(
                    "query is None and no prompt was found in seed_info.json or manifest.json."
                )
        else:
            if cfg.query is not None and sorted(normalize_query_tokens(cfg.query)) != sorted(
                normalize_query_tokens(inferred_query)
            ):
                print(
                    "[setup] configured query differs from saved sample prompt; "
                    f"using sample prompt: configured={cfg.query!r}, sample={inferred_query!r}"
                )
            cfg.query = inferred_query
            print(f"[setup] query loaded from attribution sample: {cfg.query}")
        print(
            f"[setup] loaded x0_ref shape={tuple(x0_ref.shape)} "
            f"from {precomputed_sample_meta.get('source')}"
        )
        manifest = precomputed_sample_meta.get("manifest") or {}
        manifest_ckpt = manifest.get("checkpoint")
        resolved_manifest_ckpt = resolve_checkpoint_path_from_manifest(manifest_ckpt)
        manifest_adapter = manifest.get("adapter")
        expected = "cifar" if cfg.task_type in ("cifar10", "cifar") else cfg.task_type
        if manifest_adapter is not None and manifest_adapter != expected:
            print(
                f"[warning] sample manifest adapter={manifest_adapter!r}, "
                f"but cfg.task_type={cfg.task_type!r}."
            )

    config_ckpt = cfg.reference_ckpt
    if config_ckpt is None and resolved_manifest_ckpt is not None:
        config_ckpt = resolved_manifest_ckpt
    if config_ckpt is None and cfg.baseline_dir is not None:
        config_ckpt = latest_checkpoint_in_dir(cfg.baseline_dir)
    if config_ckpt is None and cfg.lora_update_dir is not None:
        config_ckpt = latest_checkpoint_in_dir(cfg.lora_update_dir)
    apply_checkpoint_config(cfg, config_ckpt)

    print("=" * 90)
    print("Starting endpoint TracIn attribution run")
    print(f"task_type            : {cfg.task_type}")
    print(f"module_name          : {cfg.module_name}")
    print(f"baseline_dir         : {cfg.baseline_dir}")
    print(f"lora_update_dir      : {cfg.lora_update_dir}")
    print(f"reference_ckpt       : {cfg.reference_ckpt}")
    print(f"config_ckpt          : {config_ckpt}")
    print(f"attribution_sample   : {cfg.attribution_sample_dir}")
    if precomputed_sample_meta is not None:
        print(f"sample_source        : {precomputed_sample_meta.get('source')}")
        print(f"sample_shape         : {precomputed_sample_meta.get('loaded_x0_ref_shape')}")
    print(f"query                : {cfg.query}")
    print(f"seed                 : {cfg.seed}")
    print(f"timesteps            : {cfg.timesteps}")
    print(f"ddim_steps           : {cfg.ddim_steps}")
    print(f"t_min_end            : {cfg.t_min_end}")
    print(f"t_max_end_frac       : {cfg.t_max_end_frac}")
    print(f"endpoint_mc_samples  : {cfg.endpoint_mc_samples}")
    print(f"train_mc_samples     : {cfg.train_mc_samples}")
    print(f"max_train_points     : {cfg.max_train_points}")
    print(f"random_subset        : {cfg.random_subset}")
    print(f"score_index_ranges   : {cfg.score_index_ranges}")
    print(f"score_batch_size     : {cfg.score_batch_size}")
    print(f"topk                 : {cfg.topk}")
    print(f"out_dir              : {cfg.out_dir}")
    print(f"subset_suffix        : {out_suffix}")
    print("=" * 90)

    print("[setup] importing adapter and selecting device...")
    adapter = get_adapter(cfg)
    device = adapter.choose_device(cfg.prefer_device)

    print(f"[device] backend={jax.default_backend()} | device={device}")
    print(f"[device] available={jax.devices()}")
    print(f"[device] selected_platform={getattr(device, 'platform', 'unknown')}")

    print("[setup] loading dataset...")
    ds = adapter.iter_dataset(cfg)
    print(f"[setup] dataset loaded | size={len(ds)}")
    example_x, _ = adapter.get_example_batch(ds)
    print(f"[setup] example input shape={tuple(example_x.shape)}")
    if x0_ref is not None and tuple(x0_ref.shape[1:]) != tuple(example_x.shape[1:]):
        raise ValueError(
            "Loaded endpoint shape does not match the current model/data shape: "
            f"x0_ref={tuple(x0_ref.shape)}, example={tuple(example_x.shape)}. "
            "Check task_type/module_name and the checkpoint recorded by the sample manifest."
        )

    print("[setup] building model...")
    model = adapter.build_model(cfg)
    print("[setup] model built")

    print("[setup] building state template...")
    state_template = adapter.build_state_template(cfg, model, device)
    print("[setup] state template ready")

    print("[setup] building diffusion schedule...")
    schedule = schedule_to_device(
        make_diffusion_schedule(cfg.timesteps, cfg.beta_start, cfg.beta_end),
        device,
    )
    print("[setup] diffusion schedule ready")

    print("[setup] building query conditioning...")
    query_cond = adapter.make_query_cond(ds, cfg.query, cfg)
    if x0_ref is not None:
        query_cond = repeat_condition_to_batch(query_cond, int(x0_ref.shape[0]))
        x0_ref = array_to_device(x0_ref, device)
    query_cond = array_to_device(query_cond, device)
    print(f"[setup] query conditioning shape={tuple(query_cond.shape)}")
    print(
        "[device-check] "
        f"x0_ref={array_device_str(x0_ref)} | "
        f"query_cond={array_device_str(query_cond)} | "
        f"schedule_betas={array_device_str(schedule.betas)}"
    )

    # reference checkpoint / endpoint
    ref_ckpt = cfg.reference_ckpt
    if x0_ref is None:
        if ref_ckpt is None:
            if cfg.baseline_dir is None:
                raise ValueError("reference_ckpt is None and baseline_dir is also None.")
            ref_ckpt = latest_checkpoint_in_dir(cfg.baseline_dir)

        if ref_ckpt is None:
            raise FileNotFoundError("No reference checkpoint found.")

        print(f"[setup] reference_ckpt={ref_ckpt}")

        ref_state, ref_payload = adapter.restore_state(ref_ckpt, state_template)
        ref_params = tree_to_device(ref_state.ema_params, device)

        eps_fn = lambda p, x, t, c: adapter.eps_apply(model, p, x, t, c)
        x0_ref = compute_reference_endpoint_ddim(
            eps_fn=eps_fn,
            params=ref_params,
            schedule=schedule,
            cond=query_cond,
            shape=tuple(example_x.shape),
            seed=int(cfg.seed),
            ddim_steps=int(cfg.ddim_steps),
            device=device,
        )
    else:
        print("[setup] using precomputed sampler endpoint as x0_ref")

    # checkpoints to score
    baseline_ckpts = list_checkpoints_sorted(cfg.baseline_dir) if cfg.use_baseline_ckpts and cfg.baseline_dir else []
    lora_ckpts = list_checkpoints_sorted(cfg.lora_update_dir) if cfg.use_lora_ckpts and cfg.lora_update_dir else []

    if (
        cfg.use_baseline_ckpts
        and not baseline_ckpts
        and precomputed_sample_meta is not None
    ):
        if resolved_manifest_ckpt is not None:
            baseline_ckpts = [resolved_manifest_ckpt]
            print(f"[setup] using sampler manifest checkpoint for scoring: {resolved_manifest_ckpt}")

    if (
        cfg.use_baseline_ckpts
        and not baseline_ckpts
        and cfg.reference_ckpt is not None
    ):
        baseline_ckpts = [cfg.reference_ckpt]
        print(f"[setup] using reference_ckpt for scoring: {cfg.reference_ckpt}")

    if cfg.checkpoint_limit is not None and cfg.checkpoint_limit > 0:
        if len(baseline_ckpts) > cfg.checkpoint_limit:
            idx = np.linspace(0, len(baseline_ckpts) - 1, cfg.checkpoint_limit, dtype=np.int32)
            baseline_ckpts = [baseline_ckpts[i] for i in idx]
        if len(lora_ckpts) > cfg.checkpoint_limit:
            idx = np.linspace(0, len(lora_ckpts) - 1, cfg.checkpoint_limit, dtype=np.int32)
            lora_ckpts = [lora_ckpts[i] for i in idx]

    print(f"[setup] baseline_ckpts={len(baseline_ckpts)} | lora_ckpts={len(lora_ckpts)}")
    if not baseline_ckpts and not lora_ckpts:
        raise FileNotFoundError(
            "No checkpoints selected for scoring. Set baseline_dir/lora_update_dir, "
            "or pass an attribution sample run root whose manifest.json contains a valid checkpoint."
        )

    # candidate set
    N = len(ds)
    picked = build_candidate_items(cfg, N)
    M = len(picked)
    if M == 0:
        raise RuntimeError("No training points selected for scoring.")

    print(f"[candidate-set] N={N} | M_selected={M}")

    if stage_mode:
        from dtrak.algorithm import build_countsketch_projector_jax

        proj_dim = int(getattr(cfg, "proj_dim", int(os.environ.get("END_TRACIN_PROJ_DIM", "4096"))))
        stage_features = []
        stage_ckpt_indices = []
        stage_ckpt_paths = []
        stage_losses = []
        stage_checkpoint_paths = baseline_ckpts + lora_ckpts
        if not stage_checkpoint_paths:
            raise FileNotFoundError("No checkpoints selected for EndTracIn stage artifact.")

        for ckpt_idx, ckpt_path in enumerate(stage_checkpoint_paths):
            print(f"[stage:{stage_mode}] EndTracIn checkpoint {ckpt_idx + 1}/{len(stage_checkpoint_paths)} | {os.path.basename(ckpt_path)}")
            state_k, _ = adapter.restore_state(ckpt_path, state_template)
            params_k = tree_to_device(state_k.ema_params, device)
            active_mask = build_param_mask(
                params_k,
                mode="lora" if ckpt_path in lora_ckpts else "baseline",
            )
            projector = build_countsketch_projector_jax(
                params_k,
                proj_dim,
                seed_parts=(cfg.seed, "end_tracin_projection", ckpt_idx),
                device=device,
            )

            if stage_mode == "query":
                g_rng = array_to_device(jax.random.PRNGKey(cfg.seed + 10_000 + ckpt_idx), device)
                g_end, L_end = compute_g_end(
                    adapter=adapter,
                    model=model,
                    params=params_k,
                    active_mask=active_mask,
                    schedule=schedule,
                    x0_ref=x0_ref,
                    query_cond=query_cond,
                    t_min_end=int(cfg.t_min_end),
                    t_max_end=int(t_max_end),
                    endpoint_mc_samples=int(cfg.endpoint_mc_samples),
                    rng=g_rng,
                )
                stage_features.append(np.asarray(projector(g_end), dtype=np.float32))
                stage_losses.append(float(L_end))
            else:
                point_features = np.empty((M, proj_dim), dtype=np.float32)

                def train_phi_one(p, x0_one, cond_one, rng_one):
                    x0_one = x0_one[None, ...]
                    if cond_one.ndim == 0:
                        cond_one = cond_one[None]
                    else:
                        cond_one = cond_one[None, ...]

                    def loss_fn(pp):
                        return denoising_loss_mc_vectorized(
                            adapter=adapter,
                            model=model,
                            params=pp,
                            schedule=schedule,
                            x0=x0_one,
                            cond=cond_one,
                            num_mc_samples=int(cfg.train_mc_samples),
                            rng=rng_one,
                        )

                    _loss, grads = jax.value_and_grad(loss_fn)(p)
                    return projector(tree_mask(grads, active_mask))

                train_phi_batch = jax.jit(jax.vmap(train_phi_one, in_axes=(None, 0, 0, 0)))
                bs_stage = max(1, int(cfg.score_batch_size))
                for start in range(0, M, bs_stage):
                    end = min(M, start + bs_stage)
                    batch_indices = pad_indices_to_batch(picked[start:end], bs_stage)
                    x0_train, cond_train = make_train_batch(adapter, ds, batch_indices, device)
                    rngs = array_to_device(
                        jnp.stack(
                            [jax.random.PRNGKey(cfg.seed + 500_000 * (ckpt_idx + 1) + start + j) for j in range(bs_stage)],
                            axis=0,
                        ),
                        device,
                    )
                    phi_batch = train_phi_batch(params_k, x0_train, cond_train, rngs)
                    phi_batch.block_until_ready()
                    point_features[start:end] = np.asarray(phi_batch[: end - start], dtype=np.float32)
                    print(f"[stage:train] EndTracIn {end}/{M} train features")
                stage_features.append(point_features)
            stage_ckpt_indices.append(int(ckpt_idx))
            stage_ckpt_paths.append(str(ckpt_path))

        ensure_dir(os.path.dirname(stage_artifact_path))
        if stage_mode == "query":
            np.savez_compressed(
                stage_artifact_path,
                query_features=np.stack(stage_features, axis=0).astype(np.float32),
                query_losses=np.asarray(stage_losses, dtype=np.float32),
                term_weights=np.ones((len(stage_features),), dtype=np.float32),
                ckpt_indices=np.asarray(stage_ckpt_indices, dtype=np.int32),
                ckpt_paths=np.asarray(stage_ckpt_paths),
                proj_dim=np.asarray(proj_dim, dtype=np.int32),
            )
        else:
            np.savez_compressed(
                stage_artifact_path,
                train_features=np.stack(stage_features, axis=0).astype(np.float32),
                score_indices=np.asarray(picked, dtype=np.int64),
                term_weights=np.ones((len(stage_features),), dtype=np.float32),
                ckpt_indices=np.asarray(stage_ckpt_indices, dtype=np.int32),
                ckpt_paths=np.asarray(stage_ckpt_paths),
                proj_dim=np.asarray(proj_dim, dtype=np.int32),
            )
        print(f"[saved] EndTracIn {stage_mode} artifact: {stage_artifact_path}")
        return

    scores = np.zeros((M,), dtype=np.float64)

    T = int(schedule.betas.shape[0])
    t_max_end = int(float(cfg.t_max_end_frac) * T)
    t_max_end = max(0, min(T - 1, t_max_end))

    run_info = {
        "ref_ckpt": ref_ckpt,
        "attribution_sample_dir": cfg.attribution_sample_dir,
        "attribution_sample_meta": precomputed_sample_meta,
        "T": int(T),
        "ddim_steps": int(cfg.ddim_steps),
        "M_scored": int(M),
        "device": str(device),
        "seed": int(cfg.seed),
        "endpoint_mc_samples": int(cfg.endpoint_mc_samples),
        "train_mc_samples": int(cfg.train_mc_samples),
        "score_batch_size": int(cfg.score_batch_size),
        "score_index_ranges": cfg.score_index_ranges,
        "score_index_base": int(cfg.score_index_base),
        "score_subset_suffix": out_suffix,
        "baseline_ckpts": len(baseline_ckpts),
        "lora_ckpts": len(lora_ckpts),
    }

    # ---- baseline checkpoints ----
    for ckpt_idx, ckpt_path in enumerate(baseline_ckpts):
        ckpt_t0 = time.perf_counter()
        print(f"\n[baseline checkpoint] {ckpt_idx + 1}/{len(baseline_ckpts)} | {os.path.basename(ckpt_path)}")

        print("    [progress] restoring checkpoint...", flush=True)
        state_k, payload_k = adapter.restore_state(ckpt_path, state_template)
        params_k = tree_to_device(state_k.ema_params, device)
        print(f"    [device-check] params={first_leaf_device_str(params_k)}")

        active_mask = build_param_mask(params_k, mode="baseline")
        eta_k = 1.0

        g_rng = array_to_device(jax.random.PRNGKey(cfg.seed + 10_000 + ckpt_idx), device)
        print("    [progress] computing endpoint gradient g_end...", flush=True)
        g_t0 = time.perf_counter()
        g_end, L_end = compute_g_end(
            adapter=adapter,
            model=model,
            params=params_k,
            active_mask=active_mask,
            schedule=schedule,
            x0_ref=x0_ref,
            query_cond=query_cond,
            t_min_end=int(cfg.t_min_end),
            t_max_end=int(t_max_end),
            endpoint_mc_samples=int(cfg.endpoint_mc_samples),
            rng=g_rng,
        )
        print(
            f"    [progress] g_end ready | L_end_mc={float(L_end):.6f} | "
            f"elapsed={format_seconds(time.perf_counter() - g_t0)}",
            flush=True,
        )
        print("    [progress] preparing jitted batched scorer...", flush=True)
        score_fn = make_score_train_batch_fn(
            adapter=adapter,
            model=model,
            active_mask=active_mask,
            schedule=schedule,
            g_end=g_end,
            eta_k=eta_k,
            train_mc_samples=int(cfg.train_mc_samples),
        )

        score_batch_size = max(1, int(cfg.score_batch_size))
        batch_starts = list(range(0, M, score_batch_size))
        print(
            f"    [progress] scoring {M} train points in {len(batch_starts)} batches "
            f"(score_batch_size={score_batch_size})",
            flush=True,
        )
        score_iter = iter_with_tqdm(
            enumerate(batch_starts),
            total=len(batch_starts),
            desc=f"Baseline {ckpt_idx + 1}/{len(baseline_ckpts)}",
            use_tqdm=bool(cfg.use_tqdm),
        )
        for batch_i, start in score_iter:
            end = min(M, start + score_batch_size)
            batch_indices = picked[start:end]
            padded_batch_indices = pad_indices_to_batch(batch_indices, score_batch_size)
            if (not cfg.use_tqdm) and (start == 0 or end % 100 == 0 or end == M):
                print(f"    [baseline score] {end}/{M} | ckpt={ckpt_idx + 1}/{len(baseline_ckpts)}")

            x0_train, cond_train = make_train_batch(adapter, ds, padded_batch_indices, device)
            if batch_i == 0:
                print(
                    "    [device-check] "
                    f"x0_train={array_device_str(x0_train)} | "
                    f"cond_train={array_device_str(cond_train)}"
                )
            tr_rng = array_to_device(jax.random.PRNGKey(cfg.seed + 100_000 * (ckpt_idx + 1) + batch_i), device)

            sc_batch, _ = score_fn(params_k, x0_train, cond_train, tr_rng)
            scores[start:end] += np.asarray(sc_batch[: end - start], dtype=np.float64)
            if cfg.use_tqdm:
                score_iter.set_postfix(samples=f"{end}/{M}")

        print(
            f"[baseline] done: {os.path.basename(ckpt_path)} | "
            f"L_end_mc={float(L_end):.6f} | elapsed={format_seconds(time.perf_counter() - ckpt_t0)}"
        )

    # ---- LoRA checkpoints ----
    for ckpt_idx, ckpt_path in enumerate(lora_ckpts):
        ckpt_t0 = time.perf_counter()
        print(f"\n[lora checkpoint] {ckpt_idx + 1}/{len(lora_ckpts)} | {os.path.basename(ckpt_path)}")

        print("    [progress] restoring checkpoint...", flush=True)
        state_k, payload_k = adapter.restore_state(ckpt_path, state_template)
        params_k = tree_to_device(state_k.ema_params, device)
        print(f"    [device-check] params={first_leaf_device_str(params_k)}")

        active_mask = build_param_mask(params_k, mode="lora")
        if not tree_any(active_mask):
            print("    [warning] no LoRA parameters found by key name match 'lora'; scores for this ckpt will be zero.")

        eta_k = 1.0

        g_rng = array_to_device(jax.random.PRNGKey(cfg.seed + 20_000 + ckpt_idx), device)
        print("    [progress] computing endpoint gradient g_end...", flush=True)
        g_t0 = time.perf_counter()
        g_end, L_end = compute_g_end(
            adapter=adapter,
            model=model,
            params=params_k,
            active_mask=active_mask,
            schedule=schedule,
            x0_ref=x0_ref,
            query_cond=query_cond,
            t_min_end=int(cfg.t_min_end),
            t_max_end=int(t_max_end),
            endpoint_mc_samples=int(cfg.endpoint_mc_samples),
            rng=g_rng,
        )
        print(
            f"    [progress] g_end ready | L_end_mc={float(L_end):.6f} | "
            f"elapsed={format_seconds(time.perf_counter() - g_t0)}",
            flush=True,
        )
        print("    [progress] preparing jitted batched scorer...", flush=True)
        score_fn = make_score_train_batch_fn(
            adapter=adapter,
            model=model,
            active_mask=active_mask,
            schedule=schedule,
            g_end=g_end,
            eta_k=eta_k,
            train_mc_samples=int(cfg.train_mc_samples),
        )

        score_batch_size = max(1, int(cfg.score_batch_size))
        batch_starts = list(range(0, M, score_batch_size))
        print(
            f"    [progress] scoring {M} train points in {len(batch_starts)} batches "
            f"(score_batch_size={score_batch_size})",
            flush=True,
        )
        score_iter = iter_with_tqdm(
            enumerate(batch_starts),
            total=len(batch_starts),
            desc=f"LoRA {ckpt_idx + 1}/{len(lora_ckpts)}",
            use_tqdm=bool(cfg.use_tqdm),
        )
        for batch_i, start in score_iter:
            end = min(M, start + score_batch_size)
            batch_indices = picked[start:end]
            padded_batch_indices = pad_indices_to_batch(batch_indices, score_batch_size)
            if (not cfg.use_tqdm) and (start == 0 or end % 100 == 0 or end == M):
                print(f"    [lora score] {end}/{M} | ckpt={ckpt_idx + 1}/{len(lora_ckpts)}")

            x0_train, cond_train = make_train_batch(adapter, ds, padded_batch_indices, device)
            if batch_i == 0:
                print(
                    "    [device-check] "
                    f"x0_train={array_device_str(x0_train)} | "
                    f"cond_train={array_device_str(cond_train)}"
                )
            tr_rng = array_to_device(jax.random.PRNGKey(cfg.seed + 200_000 * (ckpt_idx + 1) + batch_i), device)

            sc_batch, _ = score_fn(params_k, x0_train, cond_train, tr_rng)
            scores[start:end] += np.asarray(sc_batch[: end - start], dtype=np.float64)
            if cfg.use_tqdm:
                score_iter.set_postfix(samples=f"{end}/{M}")

        print(
            f"[lora] done: {os.path.basename(ckpt_path)} | "
            f"L_end_mc={float(L_end):.6f} | elapsed={format_seconds(time.perf_counter() - ckpt_t0)}"
        )

    # ---- top-k ----
    topk = min(int(cfg.topk), M)
    order = np.argsort(-scores)[:topk]

    top = []
    for r in range(topk):
        j = int(order[r])
        train_idx = int(picked[j])
        top.append({
            "idx": train_idx,
            "score": float(scores[j]),
        })

    run_info["elapsed_sec"] = float(time.perf_counter() - t0)

    save_json(os.path.join(cfg.out_dir, "run_config.json"), asdict(cfg))
    save_json(os.path.join(cfg.out_dir, "run_info.json"), run_info)
    save_json(
        os.path.join(cfg.out_dir, "score_indices.json"),
        {
            "N_eff": int(M),
            "score_index_ranges": cfg.score_index_ranges,
            "score_index_base": int(cfg.score_index_base),
            "picked_indices": [int(i) for i in picked],
            "picked_indices_base1": [int(i) + 1 for i in picked],
        },
    )
    save_json(
        os.path.join(cfg.out_dir, "result_topk.json"),
        {
            "N_eff": int(M),
            "topk": int(topk),
            "top": top,
        },
    )
    np.save(os.path.join(cfg.out_dir, "scores.npy"), scores)

    print(f"\n[saved] {cfg.out_dir}/run_config.json")
    print(f"[saved] {cfg.out_dir}/run_info.json")
    print(f"[saved] {cfg.out_dir}/score_indices.json")
    print(f"[saved] {cfg.out_dir}/result_topk.json")
    print(f"[saved] {cfg.out_dir}/scores.npy")
    print(f"\n(done) total elapsed={format_seconds(time.perf_counter() - t0)}")

    return {
        "scores": scores,
        "top": top,
        "run_info": run_info,
    }


# ============================================================
# Example main
# ============================================================

# Usage examples
#
# 1) From the repo root, generate CIFAR attribution samples first:
#
#    cd "diffusion model_jax"
#    python3 DM___data_attribution_sampler.py \
#      --adapter cifar \
#      --code-file DM__training_CIFAR10_pixel.py \
#      --checkpoint models/cifar10_checkpoints/seed_0_epoch_0200.ckpt \
#      --prompt truck \
#      --seeds 0,1,2,3 \
#      --batch-size 1 \
#      --num-trajectory-steps 100 \
#      --outdir ./attribution_samples \
#      --prefer-device gpu \
#      --cifar-data-root ./databases/cifar-10-batches-py
#
# 2) Then run endpoint TracIn using one sampled CIFAR endpoint.
#    Set mode = "cifar10_sample" below, or construct the same config in a
#    separate driver script and call run_endpoint_tracein(cfg).
#
#    cd "diffusion model_jax"
#    python3 -m end_tracin.algorithm
#
#    The input sample is final_state.npy from:
#      attribution_samples/cifar/prompt_truck/ckpt_seed_0_epoch_0200/seed_000000/
#
#    The output scores are written to:
#      endpoint_tracein_cifar10_from_sample/
#
#    Key outputs:
#      scores.npy
#      result_topk.json
#      score_indices.json
#      run_info.json


if __name__ == "__main__":
    mode = "cifar10_sample"   # choose from: "x3", "cifar10_single", "cifar10_multi", "cifar10_sample", "artbench_latent_sample"

    if mode == "x3":
        cfg = EndpointTraceInConfig(
            task_type="x3",
            module_name="x3_training_jax",
            baseline_dir="./models/x3_checkpoints/baseline",
            lora_update_dir="./models/x3_checkpoints/lora",
            use_baseline_ckpts=True,
            use_lora_ckpts=False,
            csv_path="databases/3x3_4342_100000.csv",
            query=["background_color_red", "shape_color_blue", "shape_ring"],
            ddim_steps=1000,
            endpoint_mc_samples=10,
            train_mc_samples=10,
            max_train_points=2000,
            random_subset=True,
            topk=2000,
            out_dir="./attribution_results/endpoint_tracein/endpoint_tracein_x3",
        )

    elif mode == "cifar10_single":
        cfg = EndpointTraceInConfig(
            task_type="cifar10",
            module_name="cifar10_training_jax",
            baseline_dir="./models/cifar10_checkpoints/baseline",
            lora_update_dir="./models/cifar10_checkpoints/lora",
            use_baseline_ckpts=True,
            use_lora_ckpts=False,
            data_root="./databases/cifar-10-batches-py",
            model_type="unet",
            cond_mode="class_id",
            query="airplane",
            ddim_steps=1000,
            endpoint_mc_samples=10,
            train_mc_samples=10,
            max_train_points=2000,
            random_subset=True,
            topk=2000,
            out_dir="./attribution_results/endpoint_tracein/endpoint_tracein_cifar10_single",
        )

    elif mode == "cifar10_multi":
        cfg = EndpointTraceInConfig(
            task_type="cifar10",
            module_name="DM__training_CIFAR10_pixel",
            baseline_dir="./models/artbench_latent_dm_checkpoints256",
            lora_update_dir="./models/cifar10_checkpoints",
            use_baseline_ckpts=True,
            use_lora_ckpts=False,
            data_root="./databases/cifar-10-batches-py",
            model_type="unet",
                  
            image_size=32,
            in_channels=3,
            
            cond_mode="multi_hot",
            query=["airplane", "ship"],   # or "airplane,ship"
            ddim_steps=1000,
            t_min_end=0,
            t_max_end_frac=0.2,
            endpoint_mc_samples=3,
            train_mc_samples=3,
            score_index_ranges=((1, 10000),),
            # score_index_ranges=((10001, 20000),)
            # score_index_ranges=((20001, 30000),)
            # score_index_ranges=((30001, 40000),)
            # score_index_ranges=((40001, 50000),)
            score_index_base=1,
            max_train_points=10000,
            random_subset=False,
            topk=10000,
            out_dir="./attribution_results/endpoint_tracein/endpoint_tracein_cifar10_multi",
        )

    elif mode == "cifar10_sample":
        cfg = EndpointTraceInConfig(
            task_type="cifar10",
            module_name="DM__training_CIFAR10_pixel",
            # Optional: leave baseline_dir unset to use the checkpoint recorded
            # in the sampler manifest. If that path is stale, it is resolved by
            # checkpoint filename under ./models.
            baseline_dir="./models/cifar10_checkpoints",
            use_baseline_ckpts=True,
            use_lora_ckpts=False,
            attribution_sample_dir=(
                "./attribution_samples/cifar/prompt_truck/"
                "ckpt_seed_0_epoch_0200"
            ),
            attribution_sample_seed=0,
            attribution_sample_index=0,
            # Leave query=None to infer "truck" from seed_info.json/manifest.json.
            query=None,
            data_root="./databases/cifar-10-batches-py",
            model_type="unet",
            image_size=32,
            in_channels=3,
            cond_mode="multi_hot",
            t_min_end=0,
            t_max_end_frac=0.2,
            endpoint_mc_samples=3,
            train_mc_samples=3,
            score_index_ranges=((1, 10000),),
            score_index_base=1,
            max_train_points=1000,
            random_subset=False,
            topk=10000,
            out_dir="./attribution_results/endpoint_tracein/endpoint_tracein_cifar10_from_sample",
        )

    elif mode == "artbench_latent_sample":
        cfg = EndpointTraceInConfig(
            task_type="artbench_latent",
            module_name="DM__training_ARTBENCH_latent",
            # Leave baseline_dir unset to use the checkpoint recorded in the
            # sampler manifest. The checkpoint config supplies the current
            # latent shape, channels, schedule, and conditioning mode.
            baseline_dir="./models/cifar10_checkpoints",
            use_baseline_ckpts=True,
            use_lora_ckpts=False,
            attribution_sample_dir=(
                "./attribution_samples/artbench_latent/prompt_baroque/"
                "ckpt_seed_0_epoch_0100"
            ),
            attribution_sample_seed=0,
            attribution_sample_index=0,
            # Leave query=None to infer the prompt from seed_info.json/manifest.json.
            query=None,
            latent_npz_path="./latents/artbench256/train_latents.npz",
            t_min_end=0,
            t_max_end_frac=0.2,
            endpoint_mc_samples=3,
            train_mc_samples=3,
            score_index_ranges=((1, 10000),),
            score_index_base=1,
            max_train_points=1000,
            random_subset=False,
            topk=10000,
            out_dir="./attribution_results/endpoint_tracein/endpoint_tracein_artbench_latent_from_sample",
        )

    else:
        raise ValueError(
            f"Unknown mode: {mode}. "
            "Expected one of: 'x3', 'cifar10_single', 'cifar10_multi', "
            "'cifar10_sample', 'artbench_latent_sample'."
        )

    run_endpoint_tracein(cfg)
