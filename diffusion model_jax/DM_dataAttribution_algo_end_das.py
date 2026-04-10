import os
import time
import json
import pickle
import hashlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Optional, Sequence

import numpy as np
import jax
import jax.numpy as jnp
from flax.serialization import from_bytes


# ============================================================
# Utilities
# ============================================================

def tree_vdot(a, b):
    leaves_a, _ = jax.tree_util.tree_flatten(a)
    leaves_b, _ = jax.tree_util.tree_flatten(b)
    out = jnp.array(0.0, dtype=jnp.float32)
    for x, y in zip(leaves_a, leaves_b):
        out = out + jnp.vdot(x.astype(jnp.float32), y.astype(jnp.float32))
    return out


def format_seconds(sec: float) -> str:
    sec = max(0, int(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def format_eta(done: int, total: int, elapsed: float, warmup_steps: int = 3) -> str:
    if done <= 0 or done < warmup_steps:
        return "warming up..."
    avg = elapsed / float(done)
    remain = avg * float(max(0, total - done))
    return format_seconds(remain)


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


def normalize_query_tokens(query_spec) -> List[str]:
    if isinstance(query_spec, str):
        return [tok.strip() for tok in query_spec.split(",") if tok.strip()]
    if isinstance(query_spec, (list, tuple)):
        return [str(tok).strip() for tok in query_spec if str(tok).strip()]
    return [str(query_spec).strip()]


def _stable_int_seed(*parts: Any) -> int:
    s = "|".join(map(str, parts)).encode("utf-8")
    h = hashlib.sha256(s).digest()
    v = int.from_bytes(h[:8], "little", signed=False)
    return int(v % (2**31 - 1))


def make_jax_key(*parts: Any):
    return jax.random.PRNGKey(_stable_int_seed(*parts))


def should_print_progress(done: int, total: int, every: int) -> bool:
    return done == 1 or done % every == 0 or done == total


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
    progress_every: int = 100,
):
    print("[reference] building reference endpoint with DDIM...")
    rng = jax.random.PRNGKey(seed)
    x = jax.random.normal(rng, shape, dtype=jnp.float32)

    T = int(schedule.betas.shape[0])
    ddim_ts = np.linspace(T - 1, 0, ddim_steps, dtype=np.int32)

    loop_t0 = time.perf_counter()

    for pos, t_idx in enumerate(ddim_ts):
        t_prev_idx = int(ddim_ts[pos + 1]) if pos + 1 < len(ddim_ts) else -1
        x, _, _ = ddim_step_from_eps(eps_fn, params, schedule, x, int(t_idx), t_prev_idx, cond)

        done = pos + 1
        if should_print_progress(done, len(ddim_ts), progress_every):
            elapsed = time.perf_counter() - loop_t0
            eta = format_eta(done, len(ddim_ts), elapsed, warmup_steps=5)
            print(
                f"[reference] step {done}/{len(ddim_ts)} | "
                f"t={int(t_idx)} | "
                f"elapsed={format_seconds(elapsed)} | "
                f"eta={eta}"
            )

    total_elapsed = time.perf_counter() - loop_t0
    print(f"[reference] done | elapsed={format_seconds(total_elapsed)}")
    return x


# ============================================================
# Adapters
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


class X3TaskAdapter(BaseTaskAdapter):
    def iter_dataset(self, cfg):
        return self.m.ColorGridDatasetJAX(
            csv_path=cfg.csv_path,
            grid_size=cfg.grid_size,
            fixed_s=cfg.fixed_s,
            fixed_v=cfg.fixed_v,
            label_start=cfg.label_start,
            row_indices=cfg.row_indices,
            subset_ranges=cfg.subset_ranges,
        )

    def get_example_batch(self, ds):
        x, y = ds[0]
        return x[None, ...], y[None, ...]

    def get_item(self, ds, idx):
        x, y = ds[idx]
        return jnp.array(x[None, ...], dtype=jnp.float32), jnp.array(y[None, ...], dtype=jnp.float32)

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


class CIFAR10TaskAdapter(BaseTaskAdapter):
    def iter_dataset(self, cfg):
        return self.m.CIFAR10Dataset(
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
        q = encode_cifar_query(query=query_spec, label_names=ds.label_names, cond_mode=cfg.cond_mode)
        if cfg.cond_mode == "class_id":
            return jnp.array([int(q)], dtype=jnp.int32)
        return jnp.array(q[None, :], dtype=jnp.float32)


@dataclass
class EndpointProjectedDASJAXConfig:
    task_type: str
    module_name: str
    baseline_dir: str
    reference_ckpt: Optional[str] = None

    seed: int = 808
    query: Any = None

    timesteps_total: int = 2000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    ddim_steps: int = 2000
    eta: float = 0.0

    timesteps: Tuple[int, ...] = (0, 400, 800, 1200, 1600, 1999)
    num_mc_noise: int = 8
    damping: float = 1e-3
    proj_dim: int = 4096

    ckpt_stride: int = 1
    max_num_ckpts: Optional[int] = 1

    batch_size: int = 64
    max_train_points: int = 1024
    random_subset: bool = True
    progress_every_batches: int = 10

    topk: int = 2000
    out_dir: str = "./endpoint_das_projected_jax_out"

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
    batch_size_train: int = 128
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
    cond_mode: str = "class_id"


def get_adapter(cfg: EndpointProjectedDASJAXConfig):
    module = __import__(cfg.module_name)
    if cfg.task_type == "x3":
        return X3TaskAdapter(module)
    if cfg.task_type == "cifar10":
        return CIFAR10TaskAdapter(module)
    raise ValueError("task_type must be 'x3' or 'cifar10'")


# ============================================================
# Projected DAS helpers
# ============================================================

@dataclass
class ProjectedDASBundleJAX:
    B: jnp.ndarray
    BtB: jnp.ndarray
    Btr: jnp.ndarray
    resid_norm2: float


def _countsketch_project_grad_jax(
    grads,
    d: int,
    *,
    seed_parts: Tuple[Any, ...],
) -> jnp.ndarray:
    out = np.zeros((d,), dtype=np.float32)

    leaves, _ = jax.tree_util.tree_flatten(grads)
    for t_idx, g in enumerate(leaves):
        if g is None:
            continue

        flat = np.asarray(g, dtype=np.float32).reshape(-1)
        n = flat.size
        if n == 0:
            continue

        rng = np.random.default_rng(_stable_int_seed(*seed_parts, "tensor", t_idx, n, d))
        idx = rng.integers(0, d, size=n, dtype=np.int64)
        sign = rng.integers(0, 2, size=n, dtype=np.int8).astype(np.float32) * 2.0 - 1.0

        np.add.at(out, idx, sign * flat)

    out /= np.sqrt(float(d))
    return jnp.array(out, dtype=jnp.float32)


def sample_xt_and_noise_jax(
    schedule: DiffusionSchedule,
    x0: jnp.ndarray,
    *,
    t: jnp.ndarray,
    rng,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    noise = jax.random.normal(rng, x0.shape, dtype=x0.dtype)
    xt = q_sample(schedule, x0, t, noise)
    return xt, noise


def compute_projected_jacobian_bundle_jax(
    adapter,
    model,
    params,
    xt: jnp.ndarray,
    t: jnp.ndarray,
    cond,
    noise: jnp.ndarray,
    *,
    proj_dim: int,
    seed_parts: Tuple[Any, ...],
) -> ProjectedDASBundleJAX:
    pred = adapter.eps_apply(model, params, xt, t, cond)
    pred_flat = pred.reshape(-1)
    noise_flat = noise.reshape(-1)
    resid = (pred_flat - noise_flat).astype(jnp.float32)

    rows = []
    for k in range(int(pred_flat.shape[0])):
        def scalar_fn(p):
            pred_k = adapter.eps_apply(model, p, xt, t, cond).reshape(-1)[k]
            return pred_k

        grads = jax.grad(scalar_fn)(params)
        bk = _countsketch_project_grad_jax(
            grads,
            proj_dim,
            seed_parts=(*seed_parts, "row", k),
        )
        rows.append(bk)

    B = jnp.stack(rows, axis=0)
    BtB = B.T @ B
    Btr = B.T @ resid

    return ProjectedDASBundleJAX(
        B=B,
        BtB=BtB,
        Btr=Btr,
        resid_norm2=float(jnp.sum(resid ** 2)),
    )


# ============================================================
# Checkpoint selection
# ============================================================

def filter_checkpoints(ckpts: List[str], ckpt_stride: int, max_num_ckpts: Optional[int]) -> List[str]:
    if ckpt_stride <= 0:
        raise ValueError(f"ckpt_stride must be >= 1, got {ckpt_stride}")

    out = ckpts[::ckpt_stride]
    if max_num_ckpts is not None:
        if int(max_num_ckpts) <= 0:
            raise ValueError(f"max_num_ckpts must be positive or None, got {max_num_ckpts}")
        out = out[-int(max_num_ckpts):]

    if not out:
        raise RuntimeError("Checkpoint filtering removed all checkpoints.")
    return out


# ============================================================
# Main
# ============================================================

def run_endpoint_das_projected_jax(cfg: EndpointProjectedDASJAXConfig):
    ensure_dir(cfg.out_dir)
    t0 = time.perf_counter()

    print("=" * 90)
    print("Starting endpoint projected DAS JAX run")
    print(f"task_type               : {cfg.task_type}")
    print(f"module_name             : {cfg.module_name}")
    print(f"baseline_dir            : {cfg.baseline_dir}")
    print(f"reference_ckpt          : {cfg.reference_ckpt}")
    print(f"seed                    : {cfg.seed}")
    print(f"query                   : {cfg.query}")
    print(f"timesteps_total         : {cfg.timesteps_total}")
    print(f"ddim_steps              : {cfg.ddim_steps}")
    print(f"timesteps               : {cfg.timesteps}")
    print(f"num_mc_noise            : {cfg.num_mc_noise}")
    print(f"proj_dim                : {cfg.proj_dim}")
    print(f"damping                 : {cfg.damping}")
    print(f"batch_size              : {cfg.batch_size}")
    print(f"max_train_points        : {cfg.max_train_points}")
    print(f"random_subset           : {cfg.random_subset}")
    print(f"progress_every_batches  : {cfg.progress_every_batches}")
    print(f"topk                    : {cfg.topk}")
    print(f"out_dir                 : {cfg.out_dir}")
    print("=" * 90)

    adapter = get_adapter(cfg)
    device = adapter.m.choose_device(cfg.prefer_device)
    print(f"[device] backend={jax.default_backend()} | device={device}")

    print("[setup] loading dataset...")
    ds = adapter.iter_dataset(cfg)
    print(f"[setup] dataset ready | size={len(ds)}")

    example_x, _ = adapter.get_example_batch(ds)
    print(f"[setup] example shape={tuple(example_x.shape)}")

    print("[setup] building model...")
    model = adapter.build_model(cfg)
    print("[setup] model ready")

    print("[setup] building state template...")
    state_template = adapter.build_state_template(cfg, model, device)
    print("[setup] state template ready")

    print("[setup] building diffusion schedule...")
    schedule = make_diffusion_schedule(cfg.timesteps_total, cfg.beta_start, cfg.beta_end)
    print(f"[setup] schedule ready | T={int(schedule.betas.shape[0])}")

    print("[setup] building query conditioning...")
    query_cond = adapter.make_query_cond(ds, cfg.query, cfg)
    print(f"[setup] query conditioning ready | shape={tuple(query_cond.shape)}")

    baseline_ckpts = list_checkpoints_sorted(cfg.baseline_dir)
    if not baseline_ckpts:
        raise FileNotFoundError(f"No baseline checkpoints found in: {cfg.baseline_dir}")

    baseline_ckpts = filter_checkpoints(
        baseline_ckpts,
        ckpt_stride=int(cfg.ckpt_stride),
        max_num_ckpts=cfg.max_num_ckpts,
    )

    print(f"[checkpoints] selected={len(baseline_ckpts)}")
    for p in baseline_ckpts:
        print("  ", os.path.basename(p))

    ref_ckpt = cfg.reference_ckpt or baseline_ckpts[-1]
    print(f"[setup] reference checkpoint: {os.path.basename(ref_ckpt)}")

    print("[setup] restoring reference checkpoint...")
    ref_state, _ = adapter.restore_state(ref_ckpt, state_template)
    ref_params = ref_state.ema_params
    print("[setup] reference checkpoint restored")

    eps_fn = lambda p, x, t, c: adapter.eps_apply(model, p, x, t, c)
    x0_ref = compute_reference_endpoint_ddim(
        eps_fn=eps_fn,
        params=ref_params,
        schedule=schedule,
        cond=query_cond,
        shape=tuple(example_x.shape),
        seed=int(cfg.seed),
        ddim_steps=int(cfg.ddim_steps),
        progress_every=max(1, int(cfg.ddim_steps) // 10),
    )
    print("[setup] reference endpoint ready.")

    N_total = len(ds)
    if cfg.random_subset:
        rng = np.random.default_rng(cfg.seed)
        picked = rng.choice(
            N_total,
            size=min(int(cfg.max_train_points), N_total),
            replace=False,
        ).tolist()
    else:
        picked = list(range(min(int(cfg.max_train_points), N_total)))

    M = len(picked)
    if M == 0:
        raise RuntimeError("No training points selected for scoring.")

    print(
        f"[candidate-set] N_total={N_total} | "
        f"M_selected={M} | random_subset={cfg.random_subset}"
    )
    print(f"[candidate-set] first indices={picked[:min(10, len(picked))]}")

    scores = np.zeros((M,), dtype=np.float64)

    timesteps = [int(t) for t in cfg.timesteps]
    num_mc_noise = int(cfg.num_mc_noise)
    proj_dim = int(cfg.proj_dim)
    damping = float(cfg.damping)
    bs = int(cfg.batch_size)
    num_batches = (M + bs - 1) // bs
    total_terms = 0

    total_mc_terms = len(baseline_ckpts) * len(timesteps) * num_mc_noise
    processed_mc_terms = 0

    for ckpt_i, ckpt_path in enumerate(baseline_ckpts):
        ckpt_t0 = time.perf_counter()
        ckpt_name = os.path.basename(ckpt_path)

        print(f"\n[checkpoint] {ckpt_i+1}/{len(baseline_ckpts)} | {ckpt_name}")
        print(f"[checkpoint] restoring {ckpt_name}...")

        state_k, _ = adapter.restore_state(ckpt_path, state_template)
        params_k = state_k.ema_params

        print(f"[checkpoint] restored {ckpt_name}")

        for t_idx, t_value in enumerate(timesteps):
            T = int(schedule.betas.shape[0])
            if not (0 <= t_value < T):
                raise ValueError(f"Invalid timestep {t_value}; T={T}")

            print(
                f"[timestep] ckpt {ckpt_i+1}/{len(baseline_ckpts)} | "
                f"timestep {t_idx+1}/{len(timesteps)} | t={t_value}"
            )

            t_tensor = jnp.array([t_value], dtype=jnp.int32)

            for mc_i in range(num_mc_noise):
                total_terms += 1
                processed_mc_terms += 1
                term_t0 = time.perf_counter()

                total_elapsed = time.perf_counter() - t0
                total_eta = format_eta(processed_mc_terms, total_mc_terms, total_elapsed, warmup_steps=2)

                print(
                    f"[mc] ckpt {ckpt_i+1}/{len(baseline_ckpts)} | "
                    f"t={t_value} | mc {mc_i+1}/{num_mc_noise} | "
                    f"global_term={processed_mc_terms}/{total_mc_terms} | "
                    f"total_elapsed={format_seconds(total_elapsed)} | "
                    f"total_eta={total_eta} | "
                    f"building query bundle"
                )

                rng_q = make_jax_key(cfg.seed, "pdas_q", ckpt_i, t_value, mc_i)
                xt_q, noise_q = sample_xt_and_noise_jax(schedule, x0_ref, t=t_tensor, rng=rng_q)
                bundle_q = compute_projected_jacobian_bundle_jax(
                    adapter=adapter,
                    model=model,
                    params=params_k,
                    xt=xt_q,
                    t=t_tensor,
                    cond=query_cond,
                    noise=noise_q,
                    proj_dim=proj_dim,
                    seed_parts=(cfg.seed, "pdas_q_bundle", ckpt_i, t_value, mc_i),
                )

                print(f"[mc] query bundle ready | query_resid2={bundle_q.resid_norm2:.6f}")

                H_proj = damping * np.eye(proj_dim, dtype=np.float32)
                bundle_cache: Dict[int, ProjectedDASBundleJAX] = {}

                pass1_t0 = time.perf_counter()
                for batch_idx in range(num_batches):
                    start = batch_idx * bs
                    batch = picked[start:start + bs]
                    done_batches = batch_idx + 1

                    if should_print_progress(done_batches, num_batches, max(1, int(cfg.progress_every_batches))):
                        elapsed = time.perf_counter() - pass1_t0
                        eta = format_eta(done_batches, num_batches, elapsed, warmup_steps=2)
                        print(
                            f"    [PASS1/H-build] batch {done_batches}/{num_batches} | "
                            f"ckpt={ckpt_i+1}/{len(baseline_ckpts)}, t={t_value}, mc={mc_i+1}/{num_mc_noise} | "
                            f"elapsed={format_seconds(elapsed)} | eta={eta}"
                        )

                    for idx in batch:
                        x0_i, cond_i = adapter.get_item(ds, idx)
                        rng_i = make_jax_key(cfg.seed, "pdas_tr", ckpt_i, t_value, mc_i, idx)
                        xt_i, noise_i = sample_xt_and_noise_jax(schedule, x0_i, t=t_tensor, rng=rng_i)

                        bundle_i = compute_projected_jacobian_bundle_jax(
                            adapter=adapter,
                            model=model,
                            params=params_k,
                            xt=xt_i,
                            t=t_tensor,
                            cond=cond_i,
                            noise=noise_i,
                            proj_dim=proj_dim,
                            seed_parts=(cfg.seed, "pdas_tr_bundle", ckpt_i, t_value, mc_i, idx),
                        )

                        H_proj += np.asarray(bundle_i.BtB, dtype=np.float32)
                        bundle_cache[idx] = bundle_i

                print("[mc] PASS1 complete. Starting PASS2 scoring...")

                batch_losses_dbg = []
                pass2_t0 = time.perf_counter()
                for batch_idx in range(num_batches):
                    start = batch_idx * bs
                    batch = picked[start:start + bs]
                    done_batches = batch_idx + 1

                    if should_print_progress(done_batches, num_batches, max(1, int(cfg.progress_every_batches))):
                        elapsed = time.perf_counter() - pass2_t0
                        eta = format_eta(done_batches, num_batches, elapsed, warmup_steps=2)
                        print(
                            f"    [PASS2/Score] batch {done_batches}/{num_batches} | "
                            f"ckpt={ckpt_i+1}/{len(baseline_ckpts)}, t={t_value}, mc={mc_i+1}/{num_mc_noise} | "
                            f"elapsed={format_seconds(elapsed)} | eta={eta}"
                        )

                    batch_scores = []
                    for idx in batch:
                        bundle_i = bundle_cache[idx]

                        H_loo = H_proj - np.asarray(bundle_i.BtB, dtype=np.float32)
                        delta_z_i = np.linalg.solve(H_loo, np.asarray(bundle_i.Btr, dtype=np.float32))
                        delta_eps_q = np.asarray(bundle_q.B, dtype=np.float32) @ delta_z_i
                        score_i = float(np.sum(delta_eps_q ** 2))

                        batch_scores.append(score_i)
                        batch_losses_dbg.append(bundle_i.resid_norm2)

                    scores[start:start + len(batch)] += np.asarray(batch_scores, dtype=np.float64)

                avg_resid = float(np.mean(batch_losses_dbg)) if batch_losses_dbg else 0.0
                term_elapsed = time.perf_counter() - term_t0
                term_eta = format_eta(processed_mc_terms, total_mc_terms, time.perf_counter() - t0, warmup_steps=2)

                print(
                    f"[done] ckpt {ckpt_i+1}/{len(baseline_ckpts)} | "
                    f"t={t_value} | mc {mc_i+1}/{num_mc_noise} | "
                    f"query_resid2={bundle_q.resid_norm2:.6f} | "
                    f"avg_train_resid2={avg_resid:.6f} | "
                    f"term_elapsed={format_seconds(term_elapsed)} | "
                    f"total_eta={term_eta}"
                )

        ckpt_elapsed = time.perf_counter() - ckpt_t0
        ckpt_eta = format_eta(ckpt_i + 1, len(baseline_ckpts), time.perf_counter() - t0, warmup_steps=1)
        print(
            f"[checkpoint done] {ckpt_name} | "
            f"elapsed={format_seconds(ckpt_elapsed)} | "
            f"remaining_eta={ckpt_eta}"
        )

    if total_terms > 0:
        scores /= float(total_terms)

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

    run_info = {
        "ref_ckpt": ref_ckpt,
        "num_ckpts": int(len(baseline_ckpts)),
        "ckpt_stride": int(cfg.ckpt_stride),
        "max_num_ckpts": cfg.max_num_ckpts,
        "used_ckpts": [os.path.basename(p) for p in baseline_ckpts],
        "T": int(schedule.betas.shape[0]),
        "ddim_steps": int(cfg.ddim_steps),
        "M_scored": int(M),
        "device": str(device),
        "seed": int(cfg.seed),
        "timesteps": timesteps,
        "num_mc_noise": int(num_mc_noise),
        "damping": float(damping),
        "proj_dim": int(proj_dim),
        "num_terms_total": int(total_terms),
        "elapsed_sec": float(time.perf_counter() - t0),
        "solver": "projected_das_countsketch_jax",
    }

    save_json(os.path.join(cfg.out_dir, "run_config.json"), asdict(cfg))
    save_json(os.path.join(cfg.out_dir, "run_info.json"), run_info)
    save_json(
        os.path.join(cfg.out_dir, "score_indices.json"),
        {
            "N_total": int(N_total),
            "M_selected": int(M),
            "picked_indices": [int(i) for i in picked],
        },
    )
    save_json(
        os.path.join(cfg.out_dir, "result_topk.json"),
        {
            "N_total": int(N_total),
            "M_selected": int(M),
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


# ============================================================
# Example main
# ============================================================

if __name__ == "__main__":
    mode = "cifar10_multi"

    if mode == "x3":
        cfg = EndpointProjectedDASJAXConfig(
            task_type="x3",
            module_name="x3_training_jax",
            baseline_dir="models_checkpoints/r/model_109900/baseline",
            query=["background_color_red", "background_color_blue", "background_color_yellow"],
            seed=808,
            csv_path="generated_database/49_100000.csv",

            image_size=3,
            in_channels=3,
            grid_size=3,

            ddim_steps=2000,
            eta=0.0,
            timesteps=(0, 400, 800, 1200, 1600, 1999),
            num_mc_noise=8,
            damping=1e-3,
            proj_dim=4069,
            ckpt_stride=1,
            max_num_ckpts=1,
            batch_size=64,
            max_train_points=2000,
            random_subset=True,
            progress_every_batches=5,
            topk=2000,
            out_dir="tracein_end_runs/model_109900_r_endpoint_das_projected_jax/baseline",
        )

    elif mode == "cifar10_single":
        cfg = EndpointProjectedDASJAXConfig(
            task_type="cifar10",
            module_name="cifar10_training_jax",
            baseline_dir="./models/cifar10_checkpoints/baseline",
            query="airplane",
            seed=808,
            data_root="./databases/cifar-10-batches-py",

            image_size=32,
            in_channels=3,

            cond_mode="class_id",
            model_type="unet",
            timesteps_total=2000,
            ddim_steps=2000,
            timesteps=(0, 400, 800, 1200, 1600, 1999),
            num_mc_noise=8,
            damping=1e-3,
            proj_dim=4069,
            ckpt_stride=1,
            max_num_ckpts=1,
            batch_size=64,
            max_train_points=1024,
            random_subset=True,
            progress_every_batches=5,
            topk=100,
            out_dir="./endpoint_das_projected_cifar10_single_jax",
        )

    elif mode == "cifar10_multi":
        cfg = EndpointProjectedDASJAXConfig(
            task_type="cifar10",
            module_name="DM__training_CIFAR10_pixel",
            baseline_dir="./models/cifar10_checkpoints",
            query=["airplane", "ship"],
            seed=1,
            data_root="./databases/cifar-10-batches-py",

            image_size=32,
            in_channels=3,

            cond_mode="multi_hot",
            model_type="unet",
            timesteps_total=1000,
            ddim_steps=1000,
            timesteps=(0,),
            # timesteps=(0, 111, 222, 333, 444, 555, 666, 777, 888, 999),
            num_mc_noise=2,
            damping=1e-3,
            proj_dim=4069,
            ckpt_stride=1,
            max_num_ckpts=1,
            batch_size=64,
            max_train_points=10,
            # max_train_points=1000,
            random_subset=True,
            progress_every_batches=1,
            topk=10,
            # topk=1000,
            out_dir="./endpoint_das_projected_cifar10_multi_jax",
        )

    else:
        raise ValueError(
            f"Unknown mode: {mode}. "
            "Expected one of: 'x3', 'cifar10_single', 'cifar10_multi'."
        )

    run_endpoint_das_projected_jax(cfg)