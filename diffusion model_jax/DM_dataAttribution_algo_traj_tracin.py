import os
import time
import math
import pickle
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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


def tree_vdot(a, b):
    leaves_a, _ = jax.tree_util.tree_flatten(a)
    leaves_b, _ = jax.tree_util.tree_flatten(b)
    out = jnp.array(0.0, dtype=jnp.float32)
    for x, y in zip(leaves_a, leaves_b):
        out = out + jnp.vdot(x.astype(jnp.float32), y.astype(jnp.float32))
    return out


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


def cosine_select_indices(n_total: int, n_keep: int) -> List[int]:
    if n_keep >= n_total:
        return list(range(n_total))
    return np.linspace(0, n_total - 1, n_keep, dtype=np.int32).tolist()


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
    """
    Examples
    --------
    class_id mode:
        encode_cifar_query("airplane", label_names, cond_mode="class_id")
        encode_cifar_query(0, label_names, cond_mode="class_id")

    multi_hot mode:
        encode_cifar_query("airplane,ship", label_names, cond_mode="multi_hot")
        encode_cifar_query(["airplane", "ship"], label_names, cond_mode="multi_hot")
    """
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


def select_snapshot_positions(
    ddim_steps: int,
    num_keep: int,
    snapshot_positions: Optional[Sequence[int]] = None,
) -> List[int]:
    if snapshot_positions is not None:
        pos = sorted(set(int(v) for v in snapshot_positions))
        bad = [v for v in pos if v < 0 or v >= ddim_steps]
        if bad:
            raise ValueError(f"snapshot_positions contains invalid entries: {bad} for ddim_steps={ddim_steps}")
        return pos
    return cosine_select_indices(ddim_steps, num_keep)


def compute_reference_trajectory_ddim(
    eps_fn,
    params,
    schedule: DiffusionSchedule,
    cond,
    shape: Tuple[int, ...],
    seed: int,
    ddim_steps: int,
    num_keep: int,
    snapshot_positions: Optional[Sequence[int]] = None,
):
    print("[trajectory] preparing reference trajectory")
    rng = jax.random.PRNGKey(seed)
    x = jax.random.normal(rng, shape, dtype=jnp.float32)

    T = int(schedule.betas.shape[0])
    ddim_ts = np.linspace(T - 1, 0, ddim_steps, dtype=np.int32)
    keep_pos = set(select_snapshot_positions(ddim_steps, num_keep, snapshot_positions))

    print(
        f"[trajectory] total_ddim_steps={ddim_steps} | "
        f"keep_snapshots={len(keep_pos)} | "
        f"first_t={int(ddim_ts[0])} | last_t={int(ddim_ts[-1])}"
    )

    saved_xt = []
    saved_t = []
    saved_pos = []

    traj_start = time.time()
    report_every = max(1, ddim_steps // 10)

    for pos, t_idx in enumerate(ddim_ts):
        if pos in keep_pos:
            saved_xt.append(x)
            saved_t.append(int(t_idx))
            saved_pos.append(int(pos))

        t_prev_idx = int(ddim_ts[pos + 1]) if pos + 1 < len(ddim_ts) else -1
        x, _, _ = ddim_step_from_eps(eps_fn, params, schedule, x, int(t_idx), t_prev_idx, cond)

        done = pos + 1
        if done == 1 or done % report_every == 0 or done == len(ddim_ts):
            elapsed = time.time() - traj_start
            avg = elapsed / done
            remain = avg * (len(ddim_ts) - done)
            print(
                f"[trajectory] step {done}/{len(ddim_ts)} | "
                f"saved={len(saved_t)} | "
                f"elapsed={format_seconds(elapsed)} | "
                f"eta={format_seconds(remain)}"
            )

    print(
        f"[trajectory] done | saved_snapshots={len(saved_t)} | "
        f"elapsed={format_seconds(time.time() - traj_start)}"
    )

    return saved_xt, np.array(saved_t, dtype=np.int32), np.array(saved_pos, dtype=np.int32)


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

    def train_loss_at_t(self, model, params, schedule, x0, cond, t, rng):
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

    def train_loss_at_t(self, model, params, schedule, x0, cond, t, rng):
        noise = jax.random.normal(rng, x0.shape, dtype=x0.dtype)
        xt = q_sample(schedule, x0, t, noise)
        pred = model.apply({"params": params}, xt, t, cond, train=False)
        return jnp.mean((pred - noise) ** 2)


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

    def train_loss_at_t(self, model, params, schedule, x0, cond, t, rng):
        noise = jax.random.normal(rng, x0.shape, dtype=x0.dtype)
        xt = q_sample(schedule, x0, t, noise)
        pred = model.apply({"params": params}, xt, t, cond, train=False)
        return jnp.mean((pred - noise) ** 2)


# ============================================================
# Checkpoint helpers
# ============================================================

def list_checkpoints_sorted(checkpoint_dir: str, suffix: str = ".ckpt") -> List[str]:
    paths = []
    if not os.path.isdir(checkpoint_dir):
        return paths
    for name in os.listdir(checkpoint_dir):
        if name.endswith(suffix):
            paths.append(os.path.join(checkpoint_dir, name))
    paths.sort()
    return paths


# ============================================================
# Attribution core
# ============================================================

@dataclass
class TrajAttributionConfig:
    task_type: str  # 'x3' or 'cifar10'
    module_name: str  # e.g. 'x3_training_jax' or 'cifar10_training_jax'
    checkpoint_dir: str
    checkpoint_limit: int = -1

    # query
    query: Any = None
    seed: int = 0

    # diffusion / trajectory
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    ddim_steps: int = 1000
    num_traj_snapshots: int = 100
    traj_snapshot_positions: Optional[Tuple[int, ...]] = None
    # if provided, this overrides num_traj_snapshots selection on DDIM positions
    # positions are in [0, ddim_steps-1], not raw diffusion t

    # scoring
    train_mc_samples: int = 2
    m_proj: int = 2   # number of random r projections for query scalarization
    max_train_points: int = 1024
    random_subset: bool = True
    topk: int = 100
    progress_every: int = 50

    # save
    out_dir: str = "./traj_attr_out"

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
    cond_mode: str = "multi_hot"  # "class_id" or "multi_hot"


def get_adapter(cfg: TrajAttributionConfig):
    module = __import__(cfg.module_name)
    if cfg.task_type == "x3":
        return X3TaskAdapter(module)
    if cfg.task_type == "cifar10":
        return CIFAR10TaskAdapter(module)
    raise ValueError("task_type must be 'x3' or 'cifar10'")


def rand_rademacher_like(rng, x):
    bits = jax.random.randint(rng, x.shape, 0, 2, dtype=jnp.int32)
    return (bits * 2 - 1).astype(x.dtype)


def query_scalar(adapter, model, params, xt_ref, t, cond, rng, m_proj: int):
    eps = adapter.eps_apply(model, params, xt_ref, t, cond)
    acc = jnp.array(0.0, dtype=jnp.float32)

    proj_rngs = jax.random.split(rng, m_proj)
    for rr in proj_rngs:
        r = rand_rademacher_like(rr, eps)
        acc = acc + jnp.sum(eps * r)

    return acc / float(m_proj)


def compute_query_grads(adapter, model, params, xt_refs, t_seq, cond, cfg, base_rng):
    out = []
    total = len(t_seq)
    print(f"[query-grad] computing gradients for {total} trajectory snapshots")
    qg_start = time.time()

    for snap_i, (xt_ref, t_int) in enumerate(zip(xt_refs, t_seq)):
        base_rng, use_rng = jax.random.split(base_rng)
        t = jnp.array([int(t_int)], dtype=jnp.int32)

        def f(p):
            return query_scalar(
                adapter=adapter,
                model=model,
                params=p,
                xt_ref=xt_ref,
                t=t,
                cond=cond,
                rng=use_rng,
                m_proj=cfg.m_proj,
            )

        g = jax.grad(f)(params)
        out.append(g)

        done = snap_i + 1
        if done == 1 or done % max(1, min(10, total)) == 0 or done == total:
            elapsed = time.time() - qg_start
            avg = elapsed / done
            remain = avg * (total - done)
            print(
                f"[query-grad] {done}/{total} done | "
                f"t={int(t_int)} | "
                f"elapsed={format_seconds(elapsed)} | "
                f"eta={format_seconds(remain)}"
            )

    print(f"[query-grad] done | elapsed={format_seconds(time.time() - qg_start)}")
    return out


def score_one_point(adapter, model, params, schedule, x0, cond, query_grads, t_seq, cfg, point_seed: int):
    total = jnp.array(0.0, dtype=jnp.float32)
    w = 1.0 / float(len(t_seq))

    rng = jax.random.PRNGKey(point_seed)
    for snap_id, t_int in enumerate(t_seq):
        grads_this_t = []

        for _ in range(cfg.train_mc_samples):
            rng, step_rng = jax.random.split(rng)
            t = jnp.array([int(t_int)], dtype=jnp.int32)

            def loss_fn(p):
                return adapter.train_loss_at_t(model, p, schedule, x0, cond, t, step_rng)

            g = jax.grad(loss_fn)(params)
            grads_this_t.append(g)

        g_acc = grads_this_t[0]
        for g in grads_this_t[1:]:
            g_acc = tree_add(g_acc, g)
        g_acc = tree_scalar_mul(g_acc, 1.0 / float(cfg.train_mc_samples))

        total = total + w * tree_vdot(query_grads[snap_id], g_acc)

    return total


def run_attribution(cfg: TrajAttributionConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)
    t_start = time.time()

    print("=" * 90)
    print("Starting trajectory attribution run")
    print(f"task_type            : {cfg.task_type}")
    print(f"module_name          : {cfg.module_name}")
    print(f"checkpoint_dir       : {cfg.checkpoint_dir}")
    print(f"query                : {cfg.query}")
    print(f"seed                 : {cfg.seed}")
    print(f"timesteps            : {cfg.timesteps}")
    print(f"ddim_steps           : {cfg.ddim_steps}")
    print(f"num_traj_snapshots   : {cfg.num_traj_snapshots}")
    print(f"train_mc_samples     : {cfg.train_mc_samples}")
    print(f"m_proj               : {cfg.m_proj}")
    print(f"max_train_points     : {cfg.max_train_points}")
    print(f"random_subset        : {cfg.random_subset}")
    print(f"topk                 : {cfg.topk}")
    print(f"progress_every       : {cfg.progress_every}")
    print(f"out_dir              : {cfg.out_dir}")
    print("=" * 90)

    print("[setup] importing adapter and selecting device...")
    adapter = get_adapter(cfg)
    device = adapter.m.choose_device(cfg.prefer_device)
    print(f"[setup] using device: {device}")

    print("[setup] loading dataset...")
    ds = adapter.iter_dataset(cfg)
    print(f"[setup] dataset loaded | size={len(ds)}")

    print("[setup] building model...")
    model = adapter.build_model(cfg)
    print("[setup] model built")

    print("[setup] building state template...")
    state_template = adapter.build_state_template(cfg, model, device)
    print("[setup] state template ready")

    example_x, _ = adapter.get_example_batch(ds)
    print(f"[setup] example input shape={tuple(example_x.shape)}")

    print("[setup] building diffusion schedule...")
    schedule = make_diffusion_schedule(cfg.timesteps, cfg.beta_start, cfg.beta_end)
    print("[setup] diffusion schedule ready")

    print("[setup] building query conditioning...")
    query_cond = adapter.make_query_cond(ds, cfg.query, cfg)
    print(f"[setup] query conditioning shape={tuple(query_cond.shape)}")

    print("[setup] searching for checkpoints...")
    ckpts = list_checkpoints_sorted(cfg.checkpoint_dir)
    if cfg.checkpoint_limit is not None and cfg.checkpoint_limit > 0:
        idx = np.linspace(0, len(ckpts) - 1, min(cfg.checkpoint_limit, len(ckpts)), dtype=np.int32)
        ckpts = [ckpts[i] for i in idx]

    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {cfg.checkpoint_dir}")

    print(f"[setup] found {len(ckpts)} checkpoint(s)")
    for i, ck in enumerate(ckpts[:10]):
        print(f"  - ckpt[{i}] = {os.path.basename(ck)}")
    if len(ckpts) > 10:
        print(f"  ... and {len(ckpts) - 10} more")

    N = len(ds)
    print("[setup] selecting train points to score...")
    if cfg.random_subset:
        rng = np.random.default_rng(cfg.seed)
        picked = rng.choice(N, size=min(cfg.max_train_points, N), replace=False).tolist()
        print(f"[setup] randomly selected {len(picked)} / {N} train points")
    else:
        picked = list(range(min(cfg.max_train_points, N)))
        print(f"[setup] selected first {len(picked)} / {N} train points")

    if len(picked) > 0:
        preview = picked[: min(10, len(picked))]
        print(f"[setup] first picked indices: {preview}")

    scores = np.zeros((len(picked),), dtype=np.float64)

    snapshot_positions_used = None
    timestep_values_used = None

    total_points_all_ckpts = len(ckpts) * len(picked)
    processed_points_all_ckpts = 0

    for ckpt_i, ckpt_path in enumerate(ckpts):
        ckpt_start = time.time()
        ckpt_name = os.path.basename(ckpt_path)

        print("\n" + "-" * 90)
        print(f"[checkpoint {ckpt_i + 1}/{len(ckpts)}] starting {ckpt_name}")
        print(f"[checkpoint {ckpt_i + 1}/{len(ckpts)}] restoring state...")

        state, payload = adapter.restore_state(ckpt_path, state_template)
        params = state.ema_params

        print(f"[checkpoint {ckpt_i + 1}/{len(ckpts)}] state restored")
        print(f"[checkpoint {ckpt_i + 1}/{len(ckpts)}] building eps function")

        eps_fn = lambda p, x, t, c: adapter.eps_apply(model, p, x, t, c)

        print(f"[checkpoint {ckpt_i + 1}/{len(ckpts)}] computing reference trajectory...")
        traj_start = time.time()
        xt_refs, t_seq, pos_seq = compute_reference_trajectory_ddim(
            eps_fn=eps_fn,
            params=params,
            schedule=schedule,
            cond=query_cond,
            shape=tuple(example_x.shape),
            seed=cfg.seed,
            ddim_steps=cfg.ddim_steps,
            num_keep=cfg.num_traj_snapshots,
            snapshot_positions=cfg.traj_snapshot_positions,
        )
        print(
            f"[checkpoint {ckpt_i + 1}/{len(ckpts)}] reference trajectory ready | "
            f"num_snapshots={len(t_seq)} | "
            f"elapsed={format_seconds(time.time() - traj_start)}"
        )

        snapshot_positions_used = pos_seq.tolist()
        timestep_values_used = t_seq.tolist()

        if timestep_values_used is not None and len(timestep_values_used) > 0:
            print(
                f"[checkpoint {ckpt_i + 1}/{len(ckpts)}] snapshot timestep preview: "
                f"{timestep_values_used[:min(10, len(timestep_values_used))]}"
            )

        q_rng = jax.random.PRNGKey(cfg.seed + 1000 + ckpt_i)
        print(f"[checkpoint {ckpt_i + 1}/{len(ckpts)}] computing query gradients...")
        query_grads = compute_query_grads(adapter, model, params, xt_refs, t_seq, query_cond, cfg, q_rng)
        print(f"[checkpoint {ckpt_i + 1}/{len(ckpts)}] query gradients computed")

        print(f"[checkpoint {ckpt_i + 1}/{len(ckpts)}] scoring {len(picked)} training points...")
        score_loop_start = time.time()

        running_sum = 0.0
        running_min = None
        running_max = None

        for j, idx in enumerate(picked):
            point_start = time.time()

            x0, cond = adapter.get_item(ds, idx)
            sc = score_one_point(
                adapter=adapter,
                model=model,
                params=params,
                schedule=schedule,
                x0=x0,
                cond=cond,
                query_grads=query_grads,
                t_seq=t_seq,
                cfg=cfg,
                point_seed=cfg.seed + 100000 * ckpt_i + j,
            )
            sc_float = float(sc)
            scores[j] += sc_float

            running_sum += sc_float
            running_min = sc_float if running_min is None else min(running_min, sc_float)
            running_max = sc_float if running_max is None else max(running_max, sc_float)

            processed_points_all_ckpts += 1
            done = j + 1

            if done == 1 or done % cfg.progress_every == 0 or done == len(picked):
                elapsed_ckpt = time.time() - score_loop_start
                avg_ckpt = elapsed_ckpt / done
                remain_ckpt = avg_ckpt * (len(picked) - done)

                elapsed_total = time.time() - t_start
                avg_total = elapsed_total / processed_points_all_ckpts
                remain_total = avg_total * (total_points_all_ckpts - processed_points_all_ckpts)

                print(
                    f"[checkpoint {ckpt_i + 1}/{len(ckpts)}] "
                    f"point {done}/{len(picked)} | "
                    f"data_idx={idx} | "
                    f"last_score={sc_float:.6f} | "
                    f"mean_score={running_sum / done:.6f} | "
                    f"min_score={running_min:.6f} | "
                    f"max_score={running_max:.6f} | "
                    f"ckpt_elapsed={format_seconds(elapsed_ckpt)} | "
                    f"ckpt_eta={format_seconds(remain_ckpt)} | "
                    f"total_eta={format_seconds(remain_total)}"
                )

                if done <= 3:
                    print(
                        f"[checkpoint {ckpt_i + 1}/{len(ckpts)}] "
                        f"point runtime={format_seconds(time.time() - point_start)}"
                    )

        print(
            f"[checkpoint {ckpt_i + 1}/{len(ckpts)}] {ckpt_name} done | "
            f"num_snapshots={len(t_seq)} | "
            f"elapsed={format_seconds(time.time() - ckpt_start)}"
        )

    print("[final] computing top-k results...")
    topk = min(cfg.topk, len(picked))
    order = np.argsort(-scores)[:topk]
    top = [{"idx": int(picked[i]), "score": float(scores[i])} for i in order]

    if len(top) > 0:
        print("[final] top result preview:")
        for rank, item in enumerate(top[: min(10, len(top))], start=1):
            print(f"  rank={rank:02d} | idx={item['idx']} | score={item['score']:.6f}")

    out = {
        "config": asdict(cfg),
        "num_scored": len(picked),
        "num_snapshots_used": 0 if timestep_values_used is None else len(timestep_values_used),
        "snapshot_positions_used": snapshot_positions_used,
        "snapshot_timesteps_used": timestep_values_used,
        "topk": top,
        "elapsed_sec": time.time() - t_start,
    }

    print("[save] writing traj_attr_result.json ...")
    with open(os.path.join(cfg.out_dir, "traj_attr_result.json"), "w") as f:
        import json
        json.dump(out, f, indent=2)
    print("[save] traj_attr_result.json written")

    print("[save] writing scores.npy ...")
    np.save(os.path.join(cfg.out_dir, "scores.npy"), scores)
    print("[save] scores.npy written")

    print("=" * 90)
    print(f"Saved to {cfg.out_dir}")
    print(f"Total elapsed: {format_seconds(time.time() - t_start)}")
    print("=" * 90)
    return out


# ============================================================
# Example main
# ============================================================

if __name__ == "__main__":
    EXAMPLE = "cifar10_multi"   # choose from: "x3", "cifar10_single", "cifar10_multi"

    if EXAMPLE == "x3":
        cfg = TrajAttributionConfig(
            task_type="x3",
            module_name="x3_training_jax",
            checkpoint_dir="./models/x3_checkpoints",
            csv_path="databases/3x3_4342_100000.csv",
            query=["background_color_red", "shape_color_blue", "shape_ring"],

            # trajectory construction
            ddim_steps=1000,
            num_traj_snapshots=100,
            # traj_snapshot_positions=(0, 50, 100, 200, 400, 600, 800, 999),

            # Monte Carlo approximation controls
            train_mc_samples=2,
            m_proj=2,   # number of random r projections for query scalarization

            # how many training points to score
            max_train_points=2000,
            random_subset=True,

            # how many highest-scoring points to save
            topk=2000,

            progress_every=50,
            out_dir="./traj_attr_x3",
        )

    elif EXAMPLE == "cifar10_single":
        cfg = TrajAttributionConfig(
            task_type="cifar10",
            module_name="DM__training_CIFAR10_pixel",
            checkpoint_dir="./models/cifar10_checkpoints",
            data_root="./databases/cifar-10-batches-py",
            batch_names=("data_batch_1", "data_batch_3"),
            model_type="unet",
            image_size=32,
            in_channels=3,
            cond_mode="class_id",
            query="airplane",

            ddim_steps=1000,
            num_traj_snapshots=100,

            train_mc_samples=2,
            m_proj=2,

            # how many training points to score
            max_train_points=100,
            random_subset=True,

            # how many top results to save
            topk=100,

            progress_every=10,
            out_dir="./traj_attr_cifar10_single",
        )

    elif EXAMPLE == "cifar10_multi":
        cfg = TrajAttributionConfig(
            task_type="cifar10",
            module_name="DM__training_CIFAR10_pixel",
            checkpoint_dir="./models/cifar10_checkpoints",
            data_root="./databases/cifar-10-batches-py",
            batch_names=("data_batch_1", "data_batch_3"),
            model_type="unet",
            image_size=32,
            in_channels=3,
            cond_mode="multi_hot",
            query=["airplane", "ship"],

            ddim_steps=1000,
            num_traj_snapshots=100,

            train_mc_samples=2,
            m_proj=2,

            max_train_points=1000,
            random_subset=True,

            topk=1000,

            progress_every=25,
            out_dir="./traj_attr_cifar10_multi",
        )

    else:
        raise ValueError(
            "EXAMPLE must be one of: 'x3', 'cifar10_single', 'cifar10_multi'"
        )

    run_attribution(cfg)