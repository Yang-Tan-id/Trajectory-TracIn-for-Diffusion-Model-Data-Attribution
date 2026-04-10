import os
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
):
    rng = jax.random.PRNGKey(seed)
    x = jax.random.normal(rng, shape, dtype=jnp.float32)

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


# ============================================================
# Attribution config
# ============================================================

@dataclass
class EndpointTraceInConfig:
    task_type: str                   # 'x3' or 'cifar10'
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
    endpoint_mc_samples: int = 8
    train_mc_samples: int = 8

    # scoring set
    max_train_points: int = 1024
    random_subset: bool = True
    topk: int = 100

    # output
    out_dir: str = "./endpoint_tracein_out"

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
    prefer_device: str = "auto"
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


def get_adapter(cfg: EndpointTraceInConfig):
    module = __import__(cfg.module_name)
    if cfg.task_type == "x3":
        return X3TaskAdapter(module)
    if cfg.task_type == "cifar10":
        return CIFAR10TaskAdapter(module)
    raise ValueError("task_type must be 'x3' or 'cifar10'")


# ============================================================
# Endpoint anchored loss + score
# ============================================================

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
    T = int(schedule.betas.shape[0])
    if t_max is None:
        t_max = T - 1
    t_min = max(0, min(T - 1, int(t_min)))
    t_max = max(0, min(T - 1, int(t_max)))
    if t_max < t_min:
        t_max = t_min

    losses = []
    local_rng = rng
    for _ in range(int(num_mc_samples)):
        local_rng, noise_rng, t_rng = jax.random.split(local_rng, 3)
        t = jax.random.randint(t_rng, (x0_ref.shape[0],), minval=t_min, maxval=t_max + 1, dtype=jnp.int32)
        noise = jax.random.normal(noise_rng, x0_ref.shape, dtype=x0_ref.dtype)
        xt_ref = q_sample(schedule, x0_ref, t, noise)
        eps_pred = adapter.eps_apply(model, params, xt_ref, t, cond)
        losses.append(jnp.mean((eps_pred - noise) ** 2))

    return jnp.mean(jnp.stack(losses))


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

    L_end = loss_fn(params)
    g_end = jax.grad(loss_fn)(params)
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
    train_mc_samples=8,
    rng,
):
    def loss_fn(p):
        return adapter.train_loss_mc(
            model=model,
            params=p,
            schedule=schedule,
            x0=x0_train,
            cond=train_cond,
            num_mc_samples=train_mc_samples,
            rng=rng,
        )

    L_tr = loss_fn(params)
    g_tr = jax.grad(loss_fn)(params)
    g_tr = tree_mask(g_tr, active_mask)

    sc = eta_k * tree_vdot(g_end, g_tr)
    return sc, L_tr


# ============================================================
# Candidate selection
# ============================================================

def build_candidate_items(cfg, N: int) -> List[int]:
    M_req = min(int(cfg.max_train_points), N)

    if cfg.random_subset:
        rng = np.random.default_rng(cfg.seed)
        picked = rng.choice(N, size=M_req, replace=False).tolist()
    else:
        picked = list(range(M_req))

    return [int(i) for i in picked]


# ============================================================
# Main run
# ============================================================

def run_endpoint_tracein(cfg: EndpointTraceInConfig):
    ensure_dir(cfg.out_dir)
    t0 = time.perf_counter()

    adapter = get_adapter(cfg)
    device = adapter.m.choose_device(cfg.prefer_device)

    print(f"[device] backend={jax.default_backend()} | device={device}")

    ds = adapter.iter_dataset(cfg)
    example_x, _ = adapter.get_example_batch(ds)
    model = adapter.build_model(cfg)
    state_template = adapter.build_state_template(cfg, model, device)

    schedule = make_diffusion_schedule(cfg.timesteps, cfg.beta_start, cfg.beta_end)
    query_cond = adapter.make_query_cond(ds, cfg.query, cfg)

    # reference checkpoint
    ref_ckpt = cfg.reference_ckpt
    if ref_ckpt is None:
        if cfg.baseline_dir is None:
            raise ValueError("reference_ckpt is None and baseline_dir is also None.")
        ref_ckpt = latest_checkpoint_in_dir(cfg.baseline_dir)

    if ref_ckpt is None:
        raise FileNotFoundError("No reference checkpoint found.")

    print(f"[setup] reference_ckpt={ref_ckpt}")

    ref_state, ref_payload = adapter.restore_state(ref_ckpt, state_template)
    ref_params = ref_state.ema_params

    eps_fn = lambda p, x, t, c: adapter.eps_apply(model, p, x, t, c)
    x0_ref = compute_reference_endpoint_ddim(
        eps_fn=eps_fn,
        params=ref_params,
        schedule=schedule,
        cond=query_cond,
        shape=tuple(example_x.shape),
        seed=int(cfg.seed),
        ddim_steps=int(cfg.ddim_steps),
    )

    # checkpoints to score
    baseline_ckpts = list_checkpoints_sorted(cfg.baseline_dir) if cfg.use_baseline_ckpts and cfg.baseline_dir else []
    lora_ckpts = list_checkpoints_sorted(cfg.lora_update_dir) if cfg.use_lora_ckpts and cfg.lora_update_dir else []

    if cfg.checkpoint_limit is not None and cfg.checkpoint_limit > 0:
        if len(baseline_ckpts) > cfg.checkpoint_limit:
            idx = np.linspace(0, len(baseline_ckpts) - 1, cfg.checkpoint_limit, dtype=np.int32)
            baseline_ckpts = [baseline_ckpts[i] for i in idx]
        if len(lora_ckpts) > cfg.checkpoint_limit:
            idx = np.linspace(0, len(lora_ckpts) - 1, cfg.checkpoint_limit, dtype=np.int32)
            lora_ckpts = [lora_ckpts[i] for i in idx]

    print(f"[setup] baseline_ckpts={len(baseline_ckpts)} | lora_ckpts={len(lora_ckpts)}")

    # candidate set
    N = len(ds)
    picked = build_candidate_items(cfg, N)
    M = len(picked)
    if M == 0:
        raise RuntimeError("No training points selected for scoring.")

    print(f"[candidate-set] N={N} | M_selected={M}")

    scores = np.zeros((M,), dtype=np.float64)

    T = int(schedule.betas.shape[0])
    t_max_end = int(float(cfg.t_max_end_frac) * T)
    t_max_end = max(0, min(T - 1, t_max_end))

    run_info = {
        "ref_ckpt": ref_ckpt,
        "T": int(T),
        "ddim_steps": int(cfg.ddim_steps),
        "M_scored": int(M),
        "device": str(device),
        "seed": int(cfg.seed),
        "endpoint_mc_samples": int(cfg.endpoint_mc_samples),
        "train_mc_samples": int(cfg.train_mc_samples),
        "baseline_ckpts": len(baseline_ckpts),
        "lora_ckpts": len(lora_ckpts),
    }

    # ---- baseline checkpoints ----
    for ckpt_idx, ckpt_path in enumerate(baseline_ckpts):
        ckpt_t0 = time.perf_counter()
        print(f"\n[baseline checkpoint] {ckpt_idx + 1}/{len(baseline_ckpts)} | {os.path.basename(ckpt_path)}")

        state_k, payload_k = adapter.restore_state(ckpt_path, state_template)
        params_k = state_k.ema_params

        active_mask = build_param_mask(params_k, mode="baseline")
        eta_k = 1.0

        g_rng = jax.random.PRNGKey(cfg.seed + 10_000 + ckpt_idx)
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

        for j, idx in enumerate(picked):
            if j == 0 or (j + 1) % 100 == 0 or (j + 1) == M:
                print(f"    [baseline score] {j + 1}/{M} | ckpt={ckpt_idx + 1}/{len(baseline_ckpts)}")

            x0_train, cond_train = adapter.get_item(ds, idx)
            tr_rng = jax.random.PRNGKey(cfg.seed + 100_000 * (ckpt_idx + 1) + j)

            sc, _ = score_one_trainpoint_given_gend(
                adapter=adapter,
                model=model,
                params=params_k,
                active_mask=active_mask,
                schedule=schedule,
                g_end=g_end,
                x0_train=x0_train,
                train_cond=cond_train,
                eta_k=eta_k,
                train_mc_samples=int(cfg.train_mc_samples),
                rng=tr_rng,
            )
            scores[j] += float(sc)

        print(
            f"[baseline] done: {os.path.basename(ckpt_path)} | "
            f"L_end_mc={float(L_end):.6f} | elapsed={format_seconds(time.perf_counter() - ckpt_t0)}"
        )

    # ---- LoRA checkpoints ----
    for ckpt_idx, ckpt_path in enumerate(lora_ckpts):
        ckpt_t0 = time.perf_counter()
        print(f"\n[lora checkpoint] {ckpt_idx + 1}/{len(lora_ckpts)} | {os.path.basename(ckpt_path)}")

        state_k, payload_k = adapter.restore_state(ckpt_path, state_template)
        params_k = state_k.ema_params

        active_mask = build_param_mask(params_k, mode="lora")
        if not tree_any(active_mask):
            print("    [warning] no LoRA parameters found by key name match 'lora'; scores for this ckpt will be zero.")

        eta_k = 1.0

        g_rng = jax.random.PRNGKey(cfg.seed + 20_000 + ckpt_idx)
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

        for j, idx in enumerate(picked):
            if j == 0 or (j + 1) % 100 == 0 or (j + 1) == M:
                print(f"    [lora score] {j + 1}/{M} | ckpt={ckpt_idx + 1}/{len(lora_ckpts)}")

            x0_train, cond_train = adapter.get_item(ds, idx)
            tr_rng = jax.random.PRNGKey(cfg.seed + 200_000 * (ckpt_idx + 1) + j)

            sc, _ = score_one_trainpoint_given_gend(
                adapter=adapter,
                model=model,
                params=params_k,
                active_mask=active_mask,
                schedule=schedule,
                g_end=g_end,
                x0_train=x0_train,
                train_cond=cond_train,
                eta_k=eta_k,
                train_mc_samples=int(cfg.train_mc_samples),
                rng=tr_rng,
            )
            scores[j] += float(sc)

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
            "picked_indices": [int(i) for i in picked],
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


if __name__ == "__main__":
    mode = "cifar10_multi"   # choose from: "x3", "cifar10_single", "cifar10_multi"

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
            endpoint_mc_samples=8,
            train_mc_samples=8,
            max_train_points=2000,
            random_subset=True,
            topk=2000,
            out_dir="./endpoint_tracein_x3",
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
            endpoint_mc_samples=8,
            train_mc_samples=8,
            max_train_points=2000,
            random_subset=True,
            topk=2000,
            out_dir="./endpoint_tracein_cifar10_single",
        )

    elif mode == "cifar10_multi":
        cfg = EndpointTraceInConfig(
            task_type="cifar10",
            module_name="DM__training_CIFAR10_pixel",
            baseline_dir="./models/cifar10_checkpoints",
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
            max_train_points=1000,
            random_subset=True,
            topk=1000,
            out_dir="./endpoint_tracein_cifar10_multi",
        )

    else:
        raise ValueError(
            f"Unknown mode: {mode}. "
            "Expected one of: 'x3', 'cifar10_single', 'cifar10_multi'."
        )

    run_endpoint_tracein(cfg)