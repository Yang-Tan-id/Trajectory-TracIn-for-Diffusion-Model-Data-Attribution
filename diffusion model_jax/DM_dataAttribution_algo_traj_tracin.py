import os
import time
import math
import json
import pickle
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import jax
import jax.numpy as jnp
from flax.serialization import from_bytes

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


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


def iter_with_tqdm(iterable, total: Optional[int], desc: str, enabled: bool = True):
    if enabled and tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=True)
    return iterable


def tree_to_device(tree, device):
    return jax.tree_util.tree_map(lambda x: jax.device_put(x, device), tree)


def array_to_device(x, device):
    return jax.device_put(x, device)


def schedule_to_device(schedule: "DiffusionSchedule", device) -> "DiffusionSchedule":
    return DiffusionSchedule(
        betas=array_to_device(schedule.betas, device),
        alphas=array_to_device(schedule.alphas, device),
        alphas_cumprod=array_to_device(schedule.alphas_cumprod, device),
        sqrt_alphas_cumprod=array_to_device(schedule.sqrt_alphas_cumprod, device),
        sqrt_one_minus_alphas_cumprod=array_to_device(schedule.sqrt_one_minus_alphas_cumprod, device),
    )


def array_device_str(x) -> str:
    try:
        dev = x.device
        return str(dev() if callable(dev) else dev)
    except Exception:
        return "unknown"


def first_leaf_device_str(tree) -> str:
    leaves, _ = jax.tree_util.tree_flatten(tree)
    if not leaves:
        return "empty"
    return array_device_str(leaves[0])


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


def apply_checkpoint_config(cfg: "TrajAttributionConfig", ckpt_path: Optional[str]):
    if not cfg.sync_config_from_checkpoint:
        return
    try:
        ckpt_cfg = load_checkpoint_config(ckpt_path)
    except Exception as exc:
        print(f"[setup] could not read checkpoint config from {ckpt_path}: {exc}")
        return
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
    if os.path.isfile(os.path.join(sample_dir, "trajectory_xt.npy")):
        return sample_dir

    if seed is not None:
        candidate = os.path.join(sample_dir, f"seed_{int(seed):06d}")
        if os.path.isfile(os.path.join(candidate, "trajectory_xt.npy")):
            return candidate
        raise FileNotFoundError(f"No trajectory_xt.npy found for seed {seed}: {candidate}")

    seed_dirs = []
    if os.path.isdir(sample_dir):
        for name in os.listdir(sample_dir):
            path = os.path.join(sample_dir, name)
            if name.startswith("seed_") and os.path.isfile(os.path.join(path, "trajectory_xt.npy")):
                seed_dirs.append(path)
    seed_dirs.sort()
    if not seed_dirs:
        raise FileNotFoundError(
            "Could not find trajectory_xt.npy. Pass either a seed_* directory or a sampler run root."
        )
    return seed_dirs[0]


def _load_json_if_exists(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def infer_query_from_attribution_meta(meta: Dict[str, Any]) -> Optional[Any]:
    seed_info = meta.get("seed_info") or {}
    if seed_info.get("prompt") is not None:
        return seed_info["prompt"]
    manifest = meta.get("manifest") or {}
    if manifest.get("prompt") is not None:
        return manifest["prompt"]
    return None


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


def load_attribution_trajectory(cfg: "TrajAttributionConfig") -> Tuple[List[jnp.ndarray], np.ndarray, np.ndarray, Dict[str, Any]]:
    if cfg.attribution_sample_dir is None:
        raise ValueError("attribution_sample_dir is required.")

    seed_dir = _find_attribution_seed_dir(cfg.attribution_sample_dir, cfg.attribution_sample_seed)
    trajectory_path = os.path.join(seed_dir, "trajectory_xt.npy")
    t_path = os.path.join(seed_dir, "trajectory_t.npy")
    pos_path = os.path.join(seed_dir, "trajectory_pos.npy")

    trajectory = np.load(trajectory_path)
    if trajectory.ndim != 5:
        raise ValueError(f"Expected trajectory_xt.npy to have shape (K,B,H,W,C), got {trajectory.shape}")

    sample_idx = int(cfg.attribution_sample_index)
    if sample_idx < 0 or sample_idx >= trajectory.shape[1]:
        raise IndexError(
            f"attribution_sample_index={sample_idx} is out of range for trajectory batch size {trajectory.shape[1]}"
        )

    if not os.path.isfile(t_path):
        raise FileNotFoundError(f"Missing trajectory_t.npy next to saved trajectory: {t_path}")
    t_seq = np.load(t_path).astype(np.int32)
    if t_seq.ndim != 1 or t_seq.shape[0] != trajectory.shape[0]:
        raise ValueError(
            f"trajectory_t.npy shape {t_seq.shape} does not match trajectory snapshots {trajectory.shape[0]}"
        )

    if os.path.isfile(pos_path):
        pos_seq = np.load(pos_path).astype(np.int32)
    else:
        pos_seq = np.arange(trajectory.shape[0], dtype=np.int32)

    xt_refs = [
        jnp.asarray(trajectory[k, sample_idx:sample_idx + 1].astype(np.float32), dtype=jnp.float32)
        for k in range(trajectory.shape[0])
    ]

    seed_info = _load_json_if_exists(os.path.join(seed_dir, "seed_info.json"))
    run_root = os.path.dirname(seed_dir)
    manifest = _load_json_if_exists(os.path.join(run_root, "manifest.json"))

    meta = {
        "seed_dir": seed_dir,
        "trajectory_xt_path": trajectory_path,
        "trajectory_t_path": t_path,
        "trajectory_pos_path": pos_path if os.path.isfile(pos_path) else None,
        "sample_index": sample_idx,
        "trajectory_shape": list(trajectory.shape),
        "loaded_xt_shape": list(xt_refs[0].shape) if xt_refs else None,
        "seed_info": seed_info,
        "manifest": manifest,
    }
    return xt_refs, t_seq, pos_seq, meta


def make_train_batch(adapter, ds, indices: Sequence[int], device):
    xs = []
    conds = []
    for idx in indices:
        x, cond = adapter.get_item(ds, int(idx))
        xs.append(np.asarray(x[0], dtype=np.float32))
        conds.append(np.asarray(cond)[0])

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
# Attribution core
# ============================================================

@dataclass
class TrajAttributionConfig:
    task_type: str  # 'x3', 'cifar10', or 'artbench_latent'
    module_name: str  # e.g. 'x3_training_jax' or 'cifar10_training_jax'
    checkpoint_dir: str
    checkpoint_limit: int = -1

    # query
    query: Any = None
    seed: int = 0

    # optional precomputed sampler trajectory
    # Accepts either:
    #   - a seed directory containing trajectory_xt.npy / seed_info.json, or
    #   - a sampler run root containing manifest.json and seed_* directories.
    attribution_sample_dir: Optional[str] = None
    attribution_sample_seed: Optional[int] = None
    attribution_sample_index: int = 0
    use_saved_trajectory: bool = True
    sync_config_from_checkpoint: bool = True

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
    score_index_ranges: Optional[Tuple[Tuple[int, int], ...]] = None
    score_index_base: int = 1
    score_batch_size: int = 2
    topk: int = 100
    progress_every: int = 50
    skip_unreadable_checkpoints: bool = True

    # save
    out_dir: str = "./traj_attr_out"
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
    cond_mode: str = "multi_hot"  # "class_id" or "multi_hot"

    # artbench_latent fields
    latent_npz_path: Optional[str] = None
    cache_dir: str = "./latents/artbench256"
    latent_exclude_indices: Optional[Tuple[int, ...]] = None


def get_adapter(cfg: TrajAttributionConfig):
    module = __import__(cfg.module_name)
    if cfg.task_type == "x3":
        return X3TaskAdapter(module)
    if cfg.task_type in ("cifar10", "cifar"):
        return CIFAR10TaskAdapter(module)
    if cfg.task_type == "artbench_latent":
        return ArtBenchLatentTaskAdapter(module)
    raise ValueError("task_type must be 'x3', 'cifar10', or 'artbench_latent'")


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


def make_query_grad_fn(adapter, model, m_proj: int):
    def scalar_fn(params, xt_ref, t_scalar, cond, rng):
        t = jnp.full((xt_ref.shape[0],), t_scalar, dtype=jnp.int32)
        return query_scalar(
            adapter=adapter,
            model=model,
            params=params,
            xt_ref=xt_ref,
            t=t,
            cond=cond,
            rng=rng,
            m_proj=m_proj,
        )

    return jax.jit(jax.grad(scalar_fn))


def compute_query_grads(adapter, model, params, xt_refs, t_seq, cond, cfg, base_rng, device):
    out = []
    total = len(t_seq)
    print(f"[query-grad] computing gradients for {total} trajectory snapshots")
    qg_start = time.time()
    grad_fn = make_query_grad_fn(adapter, model, int(cfg.m_proj))

    for snap_i, (xt_ref, t_int) in enumerate(zip(xt_refs, t_seq)):
        base_rng, use_rng = jax.random.split(base_rng)
        t_scalar = array_to_device(jnp.asarray(int(t_int), dtype=jnp.int32), device)
        use_rng = array_to_device(use_rng, device)
        g = grad_fn(params, xt_ref, t_scalar, cond, use_rng)
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


def train_loss_at_fixed_t_mc_vectorized(
    adapter,
    model,
    params,
    schedule,
    x0,
    cond,
    *,
    t_int: int,
    num_mc_samples: int,
    rng,
):
    S = int(num_mc_samples)
    t_scalar = int(t_int)
    x0_rep = jnp.repeat(x0, repeats=S, axis=0)
    if cond.ndim == 1:
        cond_rep = jnp.repeat(cond, repeats=S, axis=0)
    else:
        cond_rep = jnp.repeat(cond, repeats=S, axis=0)
    t = jnp.full((x0_rep.shape[0],), t_scalar, dtype=jnp.int32)
    noise = jax.random.normal(rng, x0_rep.shape, dtype=x0_rep.dtype)
    xt = q_sample(schedule, x0_rep, t, noise)
    pred = adapter.eps_apply(model, params, xt, t, cond_rep)
    return jnp.mean((pred - noise) ** 2)


def train_loss_at_dynamic_t_mc_vectorized(
    adapter,
    model,
    params,
    schedule,
    x0,
    cond,
    *,
    t_scalar,
    num_mc_samples: int,
    rng,
):
    S = int(num_mc_samples)
    x0_rep = jnp.repeat(x0, repeats=S, axis=0)
    cond_rep = jnp.repeat(cond, repeats=S, axis=0)
    t = jnp.full((x0_rep.shape[0],), t_scalar, dtype=jnp.int32)
    noise = jax.random.normal(rng, x0_rep.shape, dtype=x0_rep.dtype)
    xt = q_sample(schedule, x0_rep, t, noise)
    pred = adapter.eps_apply(model, params, xt, t, cond_rep)
    return jnp.mean((pred - noise) ** 2)


def train_losses_at_dynamic_t_mc_vectorized(
    adapter,
    model,
    params,
    schedule,
    x0_batch,
    cond_batch,
    *,
    t_scalar,
    num_mc_samples: int,
    rng,
):
    S = int(num_mc_samples)
    B = x0_batch.shape[0]
    x0_rep = jnp.repeat(x0_batch, repeats=S, axis=0)
    cond_rep = jnp.repeat(cond_batch, repeats=S, axis=0)
    t = jnp.full((x0_rep.shape[0],), t_scalar, dtype=jnp.int32)
    noise = jax.random.normal(rng, x0_rep.shape, dtype=x0_rep.dtype)
    xt = q_sample(schedule, x0_rep, t, noise)
    pred = adapter.eps_apply(model, params, xt, t, cond_rep)
    per_replica = jnp.mean((pred - noise) ** 2, axis=tuple(range(1, pred.ndim)))
    return per_replica.reshape((B, S)).mean(axis=1)


def make_score_snapshot_batch_fn(
    adapter,
    model,
    schedule,
    *,
    train_mc_samples: int,
):
    def score_snapshot(params, query_grad, x0_batch, cond_batch, rng, t_scalar):
        def losses_fn(p):
            return train_losses_at_dynamic_t_mc_vectorized(
                adapter=adapter,
                model=model,
                params=p,
                schedule=schedule,
                x0_batch=x0_batch,
                cond_batch=cond_batch,
                t_scalar=t_scalar,
                num_mc_samples=train_mc_samples,
                rng=rng,
            )

        _, directional_scores = jax.jvp(losses_fn, (params,), (query_grad,))
        return directional_scores

    return jax.jit(score_snapshot)


def run_attribution(cfg: TrajAttributionConfig):
    subset_suffix = apply_score_subset_suffix_to_out_dir(cfg)
    os.makedirs(cfg.out_dir, exist_ok=True)
    t_start = time.time()

    precomputed_traj = None
    precomputed_sample_meta = None
    manifest_ckpt = None
    resolved_manifest_ckpt = None

    if cfg.attribution_sample_dir is not None and cfg.use_saved_trajectory:
        print(f"[setup] loading precomputed attribution trajectory: {cfg.attribution_sample_dir}")
        precomputed_traj = load_attribution_trajectory(cfg)
        _, t_preview, _, precomputed_sample_meta = precomputed_traj
        inferred_query = infer_query_from_attribution_meta(precomputed_sample_meta)
        if cfg.query is None:
            if inferred_query is None:
                raise ValueError(
                    "query is None and no prompt was found in seed_info.json or manifest.json."
                )
            cfg.query = inferred_query
            print(f"[setup] inferred query from attribution sample prompt: {cfg.query}")
        manifest = precomputed_sample_meta.get("manifest") or {}
        manifest_ckpt = manifest.get("checkpoint")
        resolved_manifest_ckpt = resolve_checkpoint_path_from_manifest(manifest_ckpt)
        print(
            f"[setup] loaded saved trajectory snapshots={len(t_preview)} | "
            f"shape={precomputed_sample_meta.get('loaded_xt_shape')}"
        )

    print("[setup] searching for checkpoints...")
    ckpts = list_checkpoints_sorted(cfg.checkpoint_dir)
    if cfg.checkpoint_limit is not None and cfg.checkpoint_limit > 0:
        idx = np.linspace(0, len(ckpts) - 1, min(cfg.checkpoint_limit, len(ckpts)), dtype=np.int32)
        ckpts = [ckpts[i] for i in idx]

    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {cfg.checkpoint_dir}")

    config_ckpt = resolved_manifest_ckpt if resolved_manifest_ckpt is not None else ckpts[0]
    apply_checkpoint_config(cfg, config_ckpt)

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
    print(f"score_index_ranges   : {cfg.score_index_ranges}")
    print(f"score_batch_size     : {cfg.score_batch_size}")
    print(f"topk                 : {cfg.topk}")
    print(f"progress_every       : {cfg.progress_every}")
    print(f"out_dir              : {cfg.out_dir}")
    print(f"subset_suffix        : {subset_suffix}")
    print("=" * 90)

    print("[setup] importing adapter and selecting device...")
    adapter = get_adapter(cfg)
    device = adapter.choose_device(cfg.prefer_device)
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
    schedule = schedule_to_device(make_diffusion_schedule(cfg.timesteps, cfg.beta_start, cfg.beta_end), device)
    print("[setup] diffusion schedule ready")

    print("[setup] building query conditioning...")
    query_cond = array_to_device(adapter.make_query_cond(ds, cfg.query, cfg), device)
    print(f"[setup] query conditioning shape={tuple(query_cond.shape)}")
    print(
        f"[device-check] query_cond={array_device_str(query_cond)} | "
        f"schedule_betas={array_device_str(schedule.betas)}"
    )

    print(f"[setup] found {len(ckpts)} checkpoint(s)")
    for i, ck in enumerate(ckpts[:10]):
        print(f"  - ckpt[{i}] = {os.path.basename(ck)}")
    if len(ckpts) > 10:
        print(f"  ... and {len(ckpts) - 10} more")

    N = len(ds)
    print("[setup] selecting train points to score...")
    picked = build_candidate_items(cfg, N)
    if cfg.score_index_ranges is not None:
        print(f"[setup] selected {len(picked)} / {N} train points from score_index_ranges")
    elif cfg.random_subset:
        print(f"[setup] randomly selected {len(picked)} / {N} train points")
    else:
        print(f"[setup] selected first {len(picked)} / {N} train points")

    if len(picked) > 0:
        preview = picked[: min(10, len(picked))]
        print(f"[setup] first picked indices: {preview}")

    scores = np.zeros((len(picked),), dtype=np.float64)

    snapshot_positions_used = None
    timestep_values_used = None

    batch_size = max(1, int(cfg.score_batch_size))
    if batch_size > 8:
        print(
            "[warning] trajectory scoring is memory-heavy; "
            f"score_batch_size={batch_size} may OOM. "
            "If the first progress bar stays at 0 with BFC allocator warnings, use 2 or 4."
        )
    total_batches_per_ckpt = int(math.ceil(len(picked) / batch_size))
    total_points_all_ckpts = len(ckpts) * len(picked)
    processed_points_all_ckpts = 0
    used_ckpts = []
    skipped_ckpts = []

    for ckpt_i, ckpt_path in enumerate(ckpts):
        ckpt_start = time.time()
        ckpt_name = os.path.basename(ckpt_path)

        print("\n" + "-" * 90)
        print(f"[checkpoint {ckpt_i + 1}/{len(ckpts)}] starting {ckpt_name}")
        print(f"[checkpoint {ckpt_i + 1}/{len(ckpts)}] restoring state...")

        try:
            state, payload = adapter.restore_state(ckpt_path, state_template)
        except Exception as exc:
            msg = f"{ckpt_path}: {exc}"
            if cfg.skip_unreadable_checkpoints:
                skipped_ckpts.append(msg)
                print(f"[warning] skipping unreadable checkpoint: {msg}")
                continue
            raise
        params = tree_to_device(state.ema_params, device)
        used_ckpts.append(ckpt_path)

        print(f"[checkpoint {ckpt_i + 1}/{len(ckpts)}] state restored")
        print(f"[device-check] params={first_leaf_device_str(params)}")
        print(f"[checkpoint {ckpt_i + 1}/{len(ckpts)}] building eps function")

        eps_fn = lambda p, x, t, c: adapter.eps_apply(model, p, x, t, c)

        if precomputed_traj is not None:
            xt_refs_raw, t_seq, pos_seq, _ = precomputed_traj
            xt_refs = [array_to_device(x, device) for x in xt_refs_raw]
            print(
                f"[checkpoint {ckpt_i + 1}/{len(ckpts)}] using saved reference trajectory | "
                f"num_snapshots={len(t_seq)}"
            )
            if xt_refs:
                print(f"[device-check] xt_ref[0]={array_device_str(xt_refs[0])}")
        else:
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
            xt_refs = [array_to_device(x, device) for x in xt_refs]
            if xt_refs:
                print(f"[device-check] xt_ref[0]={array_device_str(xt_refs[0])}")
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

        print(
            f"[checkpoint {ckpt_i + 1}/{len(ckpts)}] streaming query gradients + scoring "
            f"{len(picked)} training points..."
        )
        score_loop_start = time.time()
        print(f"[checkpoint {ckpt_i + 1}/{len(ckpts)}] preparing jitted query-gradient function...")
        query_grad_fn = make_query_grad_fn(adapter, model, int(cfg.m_proj))
        print(f"[checkpoint {ckpt_i + 1}/{len(ckpts)}] preparing jitted snapshot scorer...")
        score_snapshot_fn = make_score_snapshot_batch_fn(
            adapter=adapter,
            model=model,
            schedule=schedule,
            train_mc_samples=cfg.train_mc_samples,
        )

        running_sum = 0.0
        running_min = None
        running_max = None

        batch_starts = list(range(0, len(picked), batch_size))
        score_iter = iter_with_tqdm(
            range(len(t_seq) * len(batch_starts)),
            total=len(t_seq) * len(batch_starts),
            desc=f"Checkpoint {ckpt_i + 1}/{len(ckpts)}",
            enabled=cfg.use_tqdm,
        )

        done_units = 0
        snap_weight = 1.0 / float(max(1, len(t_seq)))
        report_every_batches = max(1, int(math.ceil(cfg.progress_every / batch_size)))

        for snap_id, t_int in enumerate(t_seq):
            t_scalar = array_to_device(jnp.asarray(int(t_int), dtype=jnp.int32), device)
            q_rng = array_to_device(
                jax.random.PRNGKey(cfg.seed + 1000 + 100000 * ckpt_i + snap_id),
                device,
            )
            query_grad = query_grad_fn(
                params,
                xt_refs[snap_id],
                t_scalar,
                query_cond,
                q_rng,
            )
            if snap_id == 0:
                query_grad = tree_to_device(query_grad, device)
                print(f"[device-check] query_grad[0]={first_leaf_device_str(query_grad)}")
            query_grad = tree_to_device(query_grad, device)

            for batch_no, start in enumerate(batch_starts, start=1):
                end = min(start + batch_size, len(picked))
                real_indices = picked[start:end]
                padded_indices = pad_indices_to_batch(real_indices, batch_size)
                x_batch, cond_batch = make_train_batch(adapter, ds, padded_indices, device)
                if snap_id == 0 and batch_no == 1:
                    print(
                        "[device-check] "
                        f"x_batch={array_device_str(x_batch)} | "
                        f"cond_batch={array_device_str(cond_batch)}"
                    )
                rng = array_to_device(
                    jax.random.PRNGKey(cfg.seed + 100000 * ckpt_i + 1000 * batch_no + snap_id),
                    device,
                )
                snap_scores = score_snapshot_fn(
                    params,
                    query_grad,
                    x_batch,
                    cond_batch,
                    rng,
                    t_scalar,
                )
                snap_scores.block_until_ready()
                batch_scores = np.asarray(jax.device_get(snap_scores))[: len(real_indices)]
                scores[start:end] += snap_weight * batch_scores.astype(np.float64)

                if snap_id == len(t_seq) - 1:
                    running_sum += float(scores[start:end].sum())
                    batch_min = float(scores[start:end].min()) if len(real_indices) else 0.0
                    batch_max = float(scores[start:end].max()) if len(real_indices) else 0.0
                    running_min = batch_min if running_min is None else min(running_min, batch_min)
                    running_max = batch_max if running_max is None else max(running_max, batch_max)
                    processed_points_all_ckpts += len(real_indices)

                done_units += 1
                if hasattr(score_iter, "update"):
                    score_iter.update(1)
                    score_iter.set_postfix(
                        snapshot=f"{snap_id + 1}/{len(t_seq)}",
                        samples=f"{end}/{len(picked)}",
                    )

                if (
                    (snap_id == 0 and batch_no == 1)
                    or (snap_id == len(t_seq) - 1 and (end == len(picked) or batch_no % report_every_batches == 0))
                ):
                    elapsed_ckpt = time.time() - score_loop_start
                    avg_unit = elapsed_ckpt / max(1, done_units)
                    remain_units = len(t_seq) * len(batch_starts) - done_units
                    remain_ckpt = avg_unit * remain_units
                    elapsed_total = time.time() - t_start
                    print(
                        f"[checkpoint {ckpt_i + 1}/{len(ckpts)}] "
                        f"snapshot {snap_id + 1}/{len(t_seq)} | "
                        f"samples {end}/{len(picked)} | "
                        f"last_idx={real_indices[-1]} | "
                        f"ckpt_elapsed={format_seconds(elapsed_ckpt)} | "
                        f"ckpt_eta={format_seconds(remain_ckpt)} | "
                        f"total_elapsed={format_seconds(elapsed_total)}"
                    )

            del query_grad

        if hasattr(score_iter, "close"):
            score_iter.close()

        print(
            f"[checkpoint {ckpt_i + 1}/{len(ckpts)}] {ckpt_name} done | "
            f"num_snapshots={len(t_seq)} | "
            f"elapsed={format_seconds(time.time() - ckpt_start)}"
        )

    if not used_ckpts:
        raise RuntimeError("No readable checkpoints were scored.")

    print("[final] computing top-k results...")
    topk = min(cfg.topk, len(picked))
    order = np.argsort(-scores)[:topk]
    top = [
        {
            "idx": int(picked[i]),
            "idx_1based": int(picked[i]) + 1,
            "score": float(scores[i]),
        }
        for i in order
    ]

    if len(top) > 0:
        print("[final] top result preview:")
        for rank, item in enumerate(top[: min(10, len(top))], start=1):
            print(f"  rank={rank:02d} | idx={item['idx']} | score={item['score']:.6f}")

    out = {
        "config": asdict(cfg),
        "num_scored": len(picked),
        "score_indices": [int(i) for i in picked],
        "score_indices_1based": [int(i) + 1 for i in picked],
        "score_subset_suffix": subset_suffix,
        "num_snapshots_used": 0 if timestep_values_used is None else len(timestep_values_used),
        "snapshot_positions_used": snapshot_positions_used,
        "snapshot_timesteps_used": timestep_values_used,
        "used_checkpoints": used_ckpts,
        "skipped_checkpoints": skipped_ckpts,
        "precomputed_sample_meta": precomputed_sample_meta,
        "manifest_checkpoint": manifest_ckpt,
        "resolved_manifest_checkpoint": resolved_manifest_ckpt,
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

    print("[save] writing score_indices.npy ...")
    np.save(os.path.join(cfg.out_dir, "score_indices.npy"), np.asarray(picked, dtype=np.int64))
    print("[save] score_indices.npy written")

    print("=" * 90)
    print(f"Saved to {cfg.out_dir}")
    print(f"Total elapsed: {format_seconds(time.time() - t_start)}")
    print("=" * 90)
    return out


# ============================================================
# Example main
# ============================================================

if __name__ == "__main__":
    EXAMPLE = "cifar10_sample"   # choose from: "x3", "cifar10_single", "cifar10_multi", "cifar10_sample", "artbench_latent_sample"

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
            score_batch_size=16,

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
            score_batch_size=16,

            # how many top results to save
            topk=10000,

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
            # Ranges are inclusive and 1-based by default.
            # Keep exactly one uncommented when splitting scoring jobs.
            score_index_ranges=((1, 10000),),
            # score_index_ranges=((10001, 20000),),
            # score_index_ranges=((20001, 30000),),
            # score_index_ranges=((30001, 40000),),
            # score_index_ranges=((40001, 50000),),
            score_index_base=1,
            score_batch_size=16,

            topk=10000,

            progress_every=25,
            out_dir="./traj_attr_cifar10_multi",
        )

    elif EXAMPLE == "cifar10_sample":
        cfg = TrajAttributionConfig(
            task_type="cifar10",
            module_name="DM__training_CIFAR10_pixel",
            checkpoint_dir="./models/cifar10_checkpoints",
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

            # Used only when no saved trajectory is provided.
            ddim_steps=1000,
            num_traj_snapshots=100,

            train_mc_samples=2,
            m_proj=2,

            max_train_points=1000,
            random_subset=True,
            # Ranges are inclusive and 1-based by default.
            # Keep exactly one uncommented when splitting scoring jobs.
            score_index_ranges=((1, 10000),),
            # score_index_ranges=((10001, 20000),),
            # score_index_ranges=((20001, 30000),),
            # score_index_ranges=((30001, 40000),),
            # score_index_ranges=((40001, 50000),),
            score_index_base=1,
            score_batch_size=16,

            topk=10000,
            progress_every=512,
            out_dir="./traj_attr_cifar10_from_sample",
        )

    elif EXAMPLE == "artbench_latent_sample":
        cfg = TrajAttributionConfig(
            task_type="artbench_latent",
            module_name="DM__training_ARTBENCH_latent",
            checkpoint_dir="./models/artbench_latent_dm_checkpoints256",
            attribution_sample_dir=(
                "./attribution_samples/artbench_latent/prompt_baroque/"
                "ckpt_seed_0_epoch_0100"
            ),
            attribution_sample_seed=0,
            attribution_sample_index=0,
            # Leave query=None to infer the prompt from seed_info.json/manifest.json.
            query=None,
            latent_npz_path="./latents/artbench256/train_latents.npz",
            cond_mode="multi_hot",

            # Used only when no saved trajectory is provided.
            ddim_steps=1000,
            num_traj_snapshots=100,

            train_mc_samples=2,
            m_proj=2,

            max_train_points=1000,
            random_subset=True,
            # Ranges are inclusive and 1-based by default.
            # Keep exactly one uncommented when splitting scoring jobs.
            score_index_ranges=((1, 10000),),
            # score_index_ranges=((10001, 20000),),
            # score_index_ranges=((20001, 30000),),
            # score_index_ranges=((30001, 40000),),
            # score_index_ranges=((40001, 50000),),
            score_index_base=1,
            score_batch_size=16,

            topk=10000,
            progress_every=512,
            out_dir="./traj_attr_artbench_latent_from_sample",
        )

    else:
        raise ValueError(
            "EXAMPLE must be one of: 'x3', 'cifar10_single', 'cifar10_multi', "
            "'cifar10_sample', 'artbench_latent_sample'"
        )

    run_attribution(cfg)
