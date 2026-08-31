import os
import sys
import time
import math
import json
import pickle
import re
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

def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def save_npz_compressed_atomic(path: str, **arrays) -> None:
    ensure_dir(os.path.dirname(path))
    tmp_path = f"{path}.tmp.npz"
    np.savez_compressed(tmp_path, **arrays)
    os.replace(tmp_path, path)


def load_stream_query_bank(paths_text: str, *, expected_proj_dim: int) -> Dict[str, Any]:
    paths = [p for p in str(paths_text).split(os.pathsep) if p]
    if not paths:
        raise ValueError("TRAJ_TRACIN_STREAM_QUERY_ARTIFACTS is empty.")

    query_features = []
    query_labels = []
    reference_meta = None
    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing query artifact: {path}")
        with np.load(path, allow_pickle=False) as data:
            features = np.asarray(data["query_features"], dtype=np.float32)
            if features.ndim != 2:
                raise ValueError(f"{path} query_features must be rank 2, got {features.shape}")
            if features.shape[1] < expected_proj_dim:
                raise ValueError(
                    f"{path} cached dim {features.shape[1]} is smaller than requested {expected_proj_dim}"
                )
            meta = {
                "ckpt_indices": np.asarray(data["ckpt_indices"], dtype=np.int32),
                "timesteps": np.asarray(data["timesteps"], dtype=np.int32),
                "snapshot_positions": np.asarray(data["snapshot_positions"], dtype=np.int32),
                "term_weights": np.asarray(data["term_weights"], dtype=np.float32),
            }
            if features.shape[0] != meta["ckpt_indices"].shape[0]:
                raise ValueError(f"{path} feature/metadata term count mismatch")
            if reference_meta is None:
                reference_meta = meta
            else:
                for key in ("ckpt_indices", "timesteps", "snapshot_positions"):
                    if not np.array_equal(reference_meta[key], meta[key]):
                        raise ValueError(f"{path} has different {key}; query artifacts are not aligned")
                if not np.allclose(reference_meta["term_weights"], meta["term_weights"], rtol=1e-5, atol=1e-12):
                    raise ValueError(f"{path} has different term_weights")
            query_features.append(features[:, :expected_proj_dim])
            query_labels.append(path)

    assert reference_meta is not None
    return {
        "paths": paths,
        "labels": query_labels,
        "query_features": np.stack(query_features, axis=0).astype(np.float32),
        **reference_meta,
    }


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


def tree_l2_norm(a):
    return jnp.sqrt(jnp.maximum(tree_vdot(a, a), jnp.array(0.0, dtype=jnp.float32)))


def tree_l2_normalize(a, eps: float):
    denom = tree_l2_norm(a) + jnp.asarray(eps, dtype=jnp.float32)
    return jax.tree_util.tree_map(lambda x: x / denom.astype(x.dtype), a)


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
        leave = os.environ.get("ATTRIBUTION_TQDM_LEAVE", "1") not in ("0", "false", "False")
        mininterval = float(os.environ.get("ATTRIBUTION_TQDM_MININTERVAL", "1"))
        return tqdm(
            iterable,
            total=total,
            desc=desc,
            dynamic_ncols=True,
            leave=leave,
            file=sys.stdout,
            mininterval=mininterval,
            maxinterval=max(30.0, mininterval),
        )
    return iterable


def tree_to_device(tree, device):
    return jax.tree_util.tree_map(lambda x: jax.device_put(x, device), tree)


def select_state_params(state, parameter_source: str):
    source = str(parameter_source or "ema").strip().lower()
    if source in ("ema", "ema_params"):
        return state.ema_params
    if source in ("raw", "params", "model", "train"):
        return state.params
    raise ValueError(
        f"Unknown parameter_source={parameter_source!r}; expected 'ema' or 'raw'."
    )


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


def select_precomputed_trajectory_snapshots(
    xt_refs,
    t_seq,
    pos_seq,
    num_keep: int,
    snapshot_positions: Optional[Sequence[int]] = None,
):
    total = len(t_seq)
    if total <= int(num_keep) and snapshot_positions is None:
        return list(xt_refs), np.asarray(t_seq, dtype=np.int32), np.asarray(pos_seq, dtype=np.int32)
    keep = select_snapshot_positions(total, int(num_keep), snapshot_positions)
    return (
        [xt_refs[int(i)] for i in keep],
        np.asarray([int(t_seq[int(i)]) for i in keep], dtype=np.int32),
        np.asarray([int(pos_seq[int(i)]) for i in keep], dtype=np.int32),
    )


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
    "epochs",
    "batch_size",
    "learning_rate",
    "lr_schedule",
    "lr_warmup_ratio",
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
        target_name = {
            "lr_schedule": "tracin_lr_schedule",
            "lr_warmup_ratio": "tracin_warmup_ratio",
        }.get(name, name)
        old = getattr(cfg, name, None)
        new = ckpt_cfg[name]
        if old != new:
            setattr(cfg, target_name, new)
            changed.append(f"{name}->{target_name}" if target_name != name else name)

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


def query_normalized_out_dir(out_dir: str) -> str:
    parent = os.path.dirname(os.path.normpath(out_dir))
    name = os.path.basename(os.path.normpath(out_dir))
    if name.startswith("traj_tracin_unprompted"):
        name = name.replace("traj_tracin_unprompted", "traj_tracin_normalized_unprompted", 1)
    elif name.startswith("traj_tracin"):
        name = name.replace("traj_tracin", "traj_tracin_normalized", 1)
    else:
        name = f"{name}_query_normalized"
    return os.path.join(parent, name)


# ============================================================
# Attribution core
# ============================================================

@dataclass
class TrajAttributionConfig:
    task_type: str  # 'x3', 'cifar10', or 'artbench_latent'
    module_name: str  # e.g. 'x3_training_jax' or 'cifar10_training_jax'
    checkpoint_dir: str
    reference_ckpt: Optional[str] = None
    checkpoint_limit: int = -1

    # query
    query: Any = None
    seed: int = 0
    query_objective: str = "trajectory_noise_squared_deviation"
    parameter_source: str = "ema"  # "ema" for historical behavior, "raw" for TrainState.params

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
    snapshot_chunk_size: int = 3
    traj_snapshot_positions: Optional[Tuple[int, ...]] = None
    # if provided, this overrides num_traj_snapshots selection on DDIM positions
    # positions are in [0, ddim_steps-1], not raw diffusion t

    # scoring
    train_mc_samples: int = 10
    m_proj: int = 2   # retained only for reading historical random-projection runs
    max_train_points: int = 1024
    random_subset: bool = True
    score_index_ranges: Optional[Tuple[Tuple[int, int], ...]] = None
    score_index_base: int = 1
    score_batch_size: int = 2
    topk: int = 100
    progress_every: int = 50
    skip_unreadable_checkpoints: bool = True
    tracin_use_learning_rate_weights: bool = True
    tracin_learning_rate: Optional[float] = None
    tracin_lr_schedule: str = "cosine_warmup"  # "constant", "cosine_warmup", or "none"
    tracin_warmup_ratio: float = 0.1
    save_query_normalized_scores: bool = False
    query_normalize_eps: float = 1e-8

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
    learning_rate: float = 1e-4
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


def checkpoint_epoch_from_path(path: str) -> Optional[int]:
    match = re.search(r"_epoch_(\d+)", os.path.basename(str(path)))
    return int(match.group(1)) if match else None


def cosine_warmup_learning_rate(base_lr: float, step: int, total_steps: int, warmup_steps: int) -> float:
    if total_steps <= 0:
        return float(base_lr)
    step = max(0, min(int(step), int(total_steps)))
    warmup_steps = max(0, min(int(warmup_steps), int(total_steps)))
    if warmup_steps > 0 and step < warmup_steps:
        return float(base_lr) * float(step) / float(max(1, warmup_steps))
    if total_steps <= warmup_steps:
        return float(base_lr)
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return float(base_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def tracin_checkpoint_lr_weight(
    cfg: TrajAttributionConfig,
    ckpt_path: str,
    ckpt_index: int,
    num_ckpts: int,
    ds_size: int,
) -> float:
    if not bool(cfg.tracin_use_learning_rate_weights):
        return 1.0
    schedule = str(cfg.tracin_lr_schedule or "constant").lower()
    if schedule in ("none", "off", "false", "0"):
        return 1.0
    base_lr = float(cfg.tracin_learning_rate if cfg.tracin_learning_rate is not None else cfg.learning_rate)
    if schedule in ("constant", "fixed"):
        return base_lr
    if schedule in ("cosine", "cosine_warmup", "warmup_cosine"):
        steps_per_epoch = int(math.ceil(float(max(1, ds_size)) / float(max(1, int(cfg.batch_size)))))
        total_steps = int(max(1, steps_per_epoch * max(1, int(cfg.epochs))))
        warmup_steps = int(math.ceil(total_steps * float(cfg.tracin_warmup_ratio)))
        epoch = checkpoint_epoch_from_path(ckpt_path)
        if epoch is None:
            # Historical checkpoint names may not include an epoch. In that
            # case, spread selected checkpoints uniformly over training.
            step = int(round(total_steps * float(ckpt_index + 1) / float(max(1, num_ckpts))))
        else:
            step = int(epoch) * steps_per_epoch
        return cosine_warmup_learning_rate(base_lr, step, total_steps, warmup_steps)
    raise ValueError(
        f"Unsupported tracin_lr_schedule={cfg.tracin_lr_schedule!r}; "
        "expected 'constant', 'cosine_warmup', or 'none'."
    )


def normalize_query_objective_name(name: str) -> str:
    aliases = {
        "trajectory_noise_squared_deviation": "trajectory_noise_squared_deviation",
        "noise_squared_deviation": "trajectory_noise_squared_deviation",
        "sum_l2_sq": "trajectory_noise_squared_deviation",
        "trajectory_next_checkpoint_noise_mse": "trajectory_next_checkpoint_noise_mse",
        "next_checkpoint_noise_mse": "trajectory_next_checkpoint_noise_mse",
        "next_ckpt_noise_mse": "trajectory_next_checkpoint_noise_mse",
        "next_checkpoint_predicted_noise_mse": "trajectory_next_checkpoint_noise_mse",
        "trajectory_next_checkpoint_ref_projection": "trajectory_next_checkpoint_ref_projection",
        "next_checkpoint_ref_projection": "trajectory_next_checkpoint_ref_projection",
        "next_ckpt_ref_projection": "trajectory_next_checkpoint_ref_projection",
        "next_checkpoint_reference_projection": "trajectory_next_checkpoint_ref_projection",
        "trajectory_future_residual_mixture": "trajectory_future_residual_mixture",
        "future_residual_mixture": "trajectory_future_residual_mixture",
        "future_residual_mix": "trajectory_future_residual_mixture",
        "future_checkpoint_residual_mixture": "trajectory_future_residual_mixture",
        "trajectory_noise_squared_deviation_normalized": "eps_deviation_l2_sq_mean",
        "normalized_trajectory_noise_squared_deviation": "eps_deviation_l2_sq_mean",
        "eps_deviation_l1_mean": "eps_deviation_l1_mean",
        "l1_mean": "eps_deviation_l1_mean",
        "mean_abs": "eps_deviation_l1_mean",
        "eps_deviation_l2_sq_mean": "eps_deviation_l2_sq_mean",
        "l2_sq_mean": "eps_deviation_l2_sq_mean",
        "mean_squared": "eps_deviation_l2_sq_mean",
    }
    try:
        return aliases[str(name)]
    except KeyError as exc:
        raise ValueError(
            "query_objective must be one of "
            "trajectory_noise_squared_deviation, trajectory_next_checkpoint_noise_mse, "
            "trajectory_next_checkpoint_ref_projection, trajectory_future_residual_mixture, "
            "eps_deviation_l1_mean, eps_deviation_l2_sq_mean, "
            "trajectory_noise_squared_deviation_normalized"
        ) from exc


def query_objective_uses_next_checkpoint(name: str) -> bool:
    return normalize_query_objective_name(name) in {
        "trajectory_next_checkpoint_noise_mse",
        "trajectory_next_checkpoint_ref_projection",
        "trajectory_future_residual_mixture",
    }


def query_objective_formula(name: str) -> str:
    name = normalize_query_objective_name(name)
    if name == "trajectory_noise_squared_deviation":
        return "sum_k w_k ||eps_theta(x_ref_k,k)-eps_theta_ref(x_ref_k,k)||_2^2"
    if name == "trajectory_next_checkpoint_noise_mse":
        return "sum_k w_k mean((eps_theta_c(x_c_k,k)-stopgrad(eps_theta_c_plus_1(x_c_k,k)))^2)"
    if name == "trajectory_next_checkpoint_ref_projection":
        return "sum_k w_k mean((stopgrad(eps_theta_c_plus_1)-eps_theta_c)*(stopgrad(eps_theta_ref)-eps_theta_c))"
    if name == "trajectory_future_residual_mixture":
        return (
            "sum_k w_k mean(stopgrad(sum_j alpha_j normalize(eps_theta_c-eps_theta_j)) * "
            "eps_theta_c), with alpha_next=1 and other alpha_j from residual disagreement"
        )
    if name == "eps_deviation_l1_mean":
        return "sum_k w_k mean(|eps_theta(x_ref_k,k)-eps_theta_ref(x_ref_k,k)|)"
    if name == "eps_deviation_l2_sq_mean":
        return "sum_k w_k mean((eps_theta(x_ref_k,k)-eps_theta_ref(x_ref_k,k))^2)"
    raise AssertionError(name)


def query_scalar(adapter, model, params, query_target_params, reference_params, xt_ref, t, cond, objective: str):
    """Per-snapshot target scalar f(theta) comparing theta to a stop-grad target model."""
    objective = normalize_query_objective_name(objective)
    eps = adapter.eps_apply(model, params, xt_ref, t, cond)
    eps_query_target = jax.lax.stop_gradient(
        adapter.eps_apply(model, query_target_params, xt_ref, t, cond)
    )
    eps_ref = jax.lax.stop_gradient(adapter.eps_apply(model, reference_params, xt_ref, t, cond))
    diff = eps - eps_query_target
    if objective == "trajectory_noise_squared_deviation":
        return jnp.sum(diff ** 2)
    if objective == "trajectory_next_checkpoint_noise_mse":
        return jnp.mean(diff ** 2)
    if objective == "trajectory_next_checkpoint_ref_projection":
        return jnp.mean((eps_query_target - eps) * (eps_ref - eps))
    if objective == "eps_deviation_l1_mean":
        return jnp.mean(jnp.abs(diff))
    if objective == "eps_deviation_l2_sq_mean":
        return jnp.mean(diff ** 2)
    raise AssertionError(objective)


def make_query_grad_fn(adapter, model, objective: str):
    objective = normalize_query_objective_name(objective)

    def scalar_fn(params, query_target_params, reference_params, xt_ref, t_scalar, cond):
        t = jnp.full((xt_ref.shape[0],), t_scalar, dtype=jnp.int32)
        return query_scalar(
            adapter=adapter,
            model=model,
            params=params,
            query_target_params=query_target_params,
            reference_params=reference_params,
            xt_ref=xt_ref,
            t=t,
            cond=cond,
            objective=objective,
        )

    return jax.jit(jax.grad(scalar_fn))


def make_query_grad_chunk_fn(adapter, model, objective: str):
    grad_fn = make_query_grad_fn(adapter, model, objective)

    def grad_chunk(params, query_target_params, reference_params, xt_refs_chunk, t_scalars, cond):
        return jax.vmap(grad_fn, in_axes=(None, None, None, 0, 0, None))(
            params,
            query_target_params,
            reference_params,
            xt_refs_chunk,
            t_scalars,
            cond,
        )

    return jax.jit(grad_chunk)


def future_residual_mixture_vector(eps: jnp.ndarray, future_eps: jnp.ndarray) -> jnp.ndarray:
    """Build a normalized future-residual mixture for one snapshot.

    future_eps[0] is treated as the next-checkpoint prediction and receives
    raw alpha 1. Other rows receive alpha from disagreement with the next
    residual. All residuals are RMS-normalized before mixing.
    """
    eps = jnp.asarray(eps)
    future_eps = jnp.asarray(future_eps)
    eps_float = eps.astype(jnp.float32)
    future_float = future_eps.astype(jnp.float32)
    residuals = eps_float[None, ...] - future_float
    reduce_axes = tuple(range(1, residuals.ndim))
    rms = jnp.sqrt(jnp.mean(jnp.square(residuals), axis=reduce_axes, keepdims=True))
    eps_denom = jnp.asarray(float(os.environ.get("TRAJ_TRACIN_FUTURE_MIX_EPS", "1e-8")), dtype=jnp.float32)
    normalized = residuals / jnp.maximum(rms, eps_denom)

    gamma = jnp.asarray(float(os.environ.get("TRAJ_TRACIN_FUTURE_MIX_GAMMA", "1.0")), dtype=jnp.float32)
    next_vec = normalized[0]
    flat_next = jnp.ravel(next_vec)
    flat_all = normalized.reshape((normalized.shape[0], -1))
    next_norm = jnp.linalg.norm(flat_next)
    all_norm = jnp.linalg.norm(flat_all, axis=1)
    cos = (flat_all @ flat_next) / jnp.maximum(all_norm * next_norm, jnp.asarray(1e-12, dtype=jnp.float32))
    novelty = jnp.clip((1.0 - cos) / 2.0, 0.0, 1.0)
    raw_alpha = gamma * novelty
    raw_alpha = raw_alpha.at[0].set(1.0)
    alpha = raw_alpha / jnp.maximum(jnp.sum(raw_alpha), jnp.asarray(1e-12, dtype=jnp.float32))
    mix = jnp.sum(alpha.reshape((-1,) + (1,) * (normalized.ndim - 1)) * normalized, axis=0)
    return mix.astype(eps.dtype)


def make_future_residual_mixture_grad_chunk_fn(adapter, model):
    def scalar_fn(params, xt_ref, t_scalar, cond, future_eps_one):
        t = jnp.full((xt_ref.shape[0],), t_scalar, dtype=jnp.int32)
        eps = adapter.eps_apply(model, params, xt_ref, t, cond)
        r_mix = jax.lax.stop_gradient(future_residual_mixture_vector(eps, future_eps_one))
        return jnp.mean(eps * r_mix)

    grad_fn = jax.grad(scalar_fn)

    def grad_chunk(params, xt_refs_chunk, t_scalars, cond, future_eps_chunk):
        return jax.vmap(grad_fn, in_axes=(None, 0, 0, None, 0))(
            params,
            xt_refs_chunk,
            t_scalars,
            cond,
            future_eps_chunk,
        )

    return jax.jit(grad_chunk)


def compute_query_grads(adapter, model, params, reference_params, xt_refs, t_seq, cond, cfg, base_rng, device):
    out = []
    total = len(t_seq)
    print(f"[query-grad] computing gradients for {total} trajectory snapshots")
    qg_start = time.time()
    grad_fn = make_query_grad_fn(adapter, model, cfg.query_objective)

    for snap_i, (xt_ref, t_int) in enumerate(zip(xt_refs, t_seq)):
        t_scalar = array_to_device(jnp.asarray(int(t_int), dtype=jnp.int32), device)
        g = grad_fn(params, reference_params, reference_params, xt_ref, t_scalar, cond)
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


def train_losses_at_t_sequence_mc_vectorized(
    adapter,
    model,
    params,
    schedule,
    x0_batch,
    cond_batch,
    *,
    t_values,
    num_mc_samples: int,
    rng,
):
    S = int(num_mc_samples)
    B = x0_batch.shape[0]
    K = int(t_values.shape[0])
    total = K * S
    x0_rep = jnp.repeat(x0_batch, repeats=total, axis=0)
    cond_rep = jnp.repeat(cond_batch, repeats=total, axis=0)
    t_rep = jnp.tile(jnp.repeat(t_values.astype(jnp.int32), repeats=S), reps=(B,))
    noise = jax.random.normal(rng, x0_rep.shape, dtype=x0_rep.dtype)
    xt = q_sample(schedule, x0_rep, t_rep, noise)
    pred = adapter.eps_apply(model, params, xt, t_rep, cond_rep)
    per_replica = jnp.mean((pred - noise) ** 2, axis=tuple(range(1, pred.ndim)))
    return per_replica.reshape((B, total)).mean(axis=1)


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


def make_score_snapshot_chunk_batch_fn(
    adapter,
    model,
    schedule,
    *,
    train_mc_samples: int,
    return_query_normalized: bool = False,
    query_normalize_eps: float = 1e-8,
):
    def score_one_snapshot(params, query_grad, x0_batch, cond_batch, rng, t_scalar):
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

        if return_query_normalized:
            _losses, pushfwd = jax.linearize(losses_fn, params)
            raw_scores = pushfwd(query_grad)
            normalized_query_grad = tree_l2_normalize(query_grad, query_normalize_eps)
            normalized_scores = pushfwd(normalized_query_grad)
            return raw_scores, normalized_scores

        _, directional_scores = jax.jvp(losses_fn, (params,), (query_grad,))
        return directional_scores

    def score_chunk(params, query_grads, x0_batch, cond_batch, rngs, t_scalars):
        return jax.vmap(score_one_snapshot, in_axes=(None, 0, None, None, 0, 0))(
            params,
            query_grads,
            x0_batch,
            cond_batch,
            rngs,
            t_scalars,
        )

    return jax.jit(score_chunk)


def make_score_t_sequence_batch_fn(
    adapter,
    model,
    schedule,
    *,
    train_mc_samples: int,
    return_query_normalized: bool = False,
    query_normalize_eps: float = 1e-8,
):
    def score_sequence(params, query_grad, x0_batch, cond_batch, rng, t_values):
        def losses_fn(p):
            return train_losses_at_t_sequence_mc_vectorized(
                adapter=adapter,
                model=model,
                params=p,
                schedule=schedule,
                x0_batch=x0_batch,
                cond_batch=cond_batch,
                t_values=t_values,
                num_mc_samples=train_mc_samples,
                rng=rng,
            )

        if return_query_normalized:
            _losses, pushfwd = jax.linearize(losses_fn, params)
            raw_scores = pushfwd(query_grad)
            normalized_query_grad = tree_l2_normalize(query_grad, query_normalize_eps)
            normalized_scores = pushfwd(normalized_query_grad)
            return raw_scores, normalized_scores

        _, directional_scores = jax.jvp(losses_fn, (params,), (query_grad,))
        return directional_scores

    return jax.jit(score_sequence)


def run_attribution(cfg: TrajAttributionConfig):
    stage_mode = os.environ.get("TRAJ_TRACIN_STAGE_MODE", "").strip().lower()
    stage_artifact_path = os.environ.get("TRAJ_TRACIN_STAGE_ARTIFACT_PATH")
    if stage_mode and stage_mode not in ("train", "query", "score_stream"):
        raise ValueError("TRAJ_TRACIN_STAGE_MODE must be unset, 'train', 'query', or 'score_stream'.")
    if stage_mode and not stage_artifact_path:
        raise ValueError("TRAJ_TRACIN_STAGE_ARTIFACT_PATH is required when TRAJ_TRACIN_STAGE_MODE is set.")
    cfg.query_objective = normalize_query_objective_name(cfg.query_objective)
    uses_next_checkpoint_target = query_objective_uses_next_checkpoint(cfg.query_objective)
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
    print(f"reference_ckpt       : {cfg.reference_ckpt}")
    print(f"query_objective      : {cfg.query_objective}")
    print(f"parameter_source     : {cfg.parameter_source}")
    print(
        "query_target         : "
        + ("next_checkpoint_predicted_noise" if uses_next_checkpoint_target else "reference_checkpoint_predicted_noise")
    )
    print(f"query                : {cfg.query}")
    print(f"seed                 : {cfg.seed}")
    print(f"timesteps            : {cfg.timesteps}")
    print(f"ddim_steps           : {cfg.ddim_steps}")
    print(f"num_traj_snapshots   : {cfg.num_traj_snapshots}")
    print(f"snapshot_chunk_size  : {cfg.snapshot_chunk_size}")
    print(f"train_mc_samples     : {cfg.train_mc_samples}")
    print(f"max_train_points     : {cfg.max_train_points}")
    print(f"random_subset        : {cfg.random_subset}")
    print(f"score_index_ranges   : {cfg.score_index_ranges}")
    print(f"score_batch_size     : {cfg.score_batch_size}")
    print(f"topk                 : {cfg.topk}")
    print(f"progress_every       : {cfg.progress_every}")
    print(f"out_dir              : {cfg.out_dir}")
    print(f"subset_suffix        : {subset_suffix}")
    print("=" * 90)
    if uses_next_checkpoint_target:
        print(
            "[setup] next-checkpoint query target enabled; "
            "final checkpoint has no c+1 target and will be skipped."
        )

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

    reference_ckpt = (
        cfg.reference_ckpt
        or resolved_manifest_ckpt
        or ckpts[-1]
    )
    if not os.path.isfile(reference_ckpt):
        raise FileNotFoundError(f"Reference checkpoint not found: {reference_ckpt}")
    print(f"[setup] restoring f_noise reference checkpoint: {reference_ckpt}")
    reference_state, _ = adapter.restore_state(reference_ckpt, state_template)
    reference_params = tree_to_device(select_state_params(reference_state, cfg.parameter_source), device)
    cfg.reference_ckpt = reference_ckpt
    print(f"[device-check] reference_params={first_leaf_device_str(reference_params)}")

    def query_target_params_for_checkpoint(ckpt_i: int):
        if not uses_next_checkpoint_target:
            return reference_params
        next_ckpt_path = ckpts[ckpt_i + 1]
        next_state, _ = adapter.restore_state(next_ckpt_path, state_template)
        next_params = tree_to_device(select_state_params(next_state, cfg.parameter_source), device)
        print(
            f"[checkpoint {ckpt_i + 1}/{len(ckpts)}] "
            f"query target is next checkpoint: {os.path.basename(next_ckpt_path)}"
        )
        return next_params

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

    future_noise_cache = None
    future_noise_cache_path = None

    def future_noise_cache_default_path() -> str:
        if stage_artifact_path:
            return f"{stage_artifact_path}.future_noise_cache.npz"
        return os.path.join(cfg.out_dir, "future_noise_cache.npz")

    def load_or_build_future_noise_cache(xt_refs, t_seq, pos_seq):
        nonlocal future_noise_cache, future_noise_cache_path
        if future_noise_cache is not None:
            return future_noise_cache
        cache_path = os.environ.get("TRAJ_TRACIN_FUTURE_NOISE_CACHE_PATH", "").strip()
        if not cache_path:
            cache_dir = os.environ.get("TRAJ_TRACIN_FUTURE_NOISE_CACHE_DIR", "").strip()
            cache_path = (
                os.path.join(cache_dir, os.path.basename(future_noise_cache_default_path()))
                if cache_dir
                else future_noise_cache_default_path()
            )
        future_noise_cache_path = cache_path
        ckpt_paths_np = np.asarray([str(p) for p in ckpts])
        t_seq_np = np.asarray([int(t) for t in t_seq], dtype=np.int32)
        pos_seq_np = np.asarray([int(p) for p in pos_seq], dtype=np.int32)

        if os.path.isfile(cache_path):
            try:
                with np.load(cache_path, allow_pickle=False) as payload:
                    cached_ckpts = np.asarray(payload["ckpt_paths"]).astype(str)
                    cached_t = np.asarray(payload["timesteps"], dtype=np.int32)
                    cached_pos = np.asarray(payload["snapshot_positions"], dtype=np.int32)
                    if (
                        np.array_equal(cached_ckpts, ckpt_paths_np.astype(str))
                        and np.array_equal(cached_t, t_seq_np)
                        and np.array_equal(cached_pos, pos_seq_np)
                    ):
                        future_noise_cache = np.asarray(payload["eps_predictions"])
                        print(
                            "[future-cache] loaded predicted-noise cache "
                            f"{cache_path} | shape={future_noise_cache.shape} dtype={future_noise_cache.dtype}",
                            flush=True,
                        )
                        return future_noise_cache
                    print("[future-cache] existing cache metadata mismatch; rebuilding", flush=True)
            except Exception as exc:
                print(f"[future-cache] could not read existing cache {cache_path}: {exc}; rebuilding", flush=True)

        print(
            "[future-cache] building predicted-noise cache for future residual mixture | "
            f"checkpoints={len(ckpts)} snapshots={len(t_seq)} -> {cache_path}",
            flush=True,
        )

        def eps_chunk(params_for_eps, xt_chunk, t_chunk):
            return jax.vmap(
                lambda x_one, t_one: adapter.eps_apply(
                    model,
                    params_for_eps,
                    x_one,
                    jnp.full((x_one.shape[0],), t_one, dtype=jnp.int32),
                    query_cond,
                )
            )(xt_chunk, t_chunk)

        eps_chunk_jit = jax.jit(eps_chunk)
        cache_dtype_name = os.environ.get("TRAJ_TRACIN_FUTURE_NOISE_CACHE_DTYPE", "float16").strip().lower()
        cache_dtype = np.float32 if cache_dtype_name in ("float32", "fp32") else np.float16
        chunk_size = max(1, int(cfg.snapshot_chunk_size))
        eps_by_ckpt = []
        cache_start = time.time()
        for cache_ckpt_i, cache_ckpt_path in enumerate(ckpts):
            ckpt_start = time.time()
            print(
                f"[future-cache] checkpoint {cache_ckpt_i + 1}/{len(ckpts)} "
                f"{os.path.basename(cache_ckpt_path)}",
                flush=True,
            )
            state, _payload = adapter.restore_state(cache_ckpt_path, state_template)
            params_for_eps = tree_to_device(select_state_params(state, cfg.parameter_source), device)
            chunks = []
            for chunk_start in range(0, len(t_seq), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(t_seq))
                xt_chunk = array_to_device(jnp.stack([xt_refs[i] for i in range(chunk_start, chunk_end)], axis=0), device)
                t_chunk = array_to_device(
                    jnp.asarray([int(t_seq[i]) for i in range(chunk_start, chunk_end)], dtype=jnp.int32),
                    device,
                )
                eps = eps_chunk_jit(params_for_eps, xt_chunk, t_chunk)
                eps.block_until_ready()
                chunks.append(np.asarray(jax.device_get(eps), dtype=cache_dtype))
            eps_by_ckpt.append(np.concatenate(chunks, axis=0))
            print(
                f"[future-cache] checkpoint {cache_ckpt_i + 1}/{len(ckpts)} done | "
                f"elapsed={format_seconds(time.time() - ckpt_start)} | "
                f"total={format_seconds(time.time() - cache_start)}",
                flush=True,
            )
        future_noise_cache = np.stack(eps_by_ckpt, axis=0)
        save_npz_compressed_atomic(
            cache_path,
            eps_predictions=future_noise_cache,
            ckpt_paths=ckpt_paths_np,
            timesteps=t_seq_np,
            snapshot_positions=pos_seq_np,
            query=np.asarray(str(cfg.query)),
            attribution_sample_dir=np.asarray(str(cfg.attribution_sample_dir)),
            attribution_sample_seed=np.asarray(-1 if cfg.attribution_sample_seed is None else int(cfg.attribution_sample_seed)),
        )
        print(
            "[future-cache] saved predicted-noise cache "
            f"{cache_path} | shape={future_noise_cache.shape} dtype={future_noise_cache.dtype}",
            flush=True,
        )
        return future_noise_cache

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

    if stage_mode == "score_stream":
        from dtrak.algorithm import build_countsketch_projector_jax

        cache_dim = int(os.environ.get("TRAJ_TRACIN_STREAM_CACHE_DIM", os.environ.get("TRAJ_TRACIN_PROJ_DIM", "4096")))
        proj_dims_text = os.environ.get("TRAJ_TRACIN_STREAM_PROJ_DIMS", os.environ.get("TRAJ_TRACIN_STREAM_PROJ_DIM", str(cache_dim)))
        proj_dims = tuple(int(part.strip()) for part in proj_dims_text.replace(",", " ").split() if part.strip())
        if not proj_dims:
            raise ValueError("TRAJ_TRACIN_STREAM_PROJ_DIMS is empty.")
        if any(dim <= 0 for dim in proj_dims):
            raise ValueError(f"Projection dims must be positive, got {proj_dims}")
        if any(dim > cache_dim for dim in proj_dims):
            raise ValueError(f"Projection dims {proj_dims} exceed cache dim {cache_dim}")
        max_score_dim = max(proj_dims)
        query_bank = load_stream_query_bank(
            os.environ.get("TRAJ_TRACIN_STREAM_QUERY_ARTIFACTS", ""),
            expected_proj_dim=max_score_dim,
        )
        query_features = np.asarray(query_bank["query_features"], dtype=np.float32)
        term_ckpt_indices = np.asarray(query_bank["ckpt_indices"], dtype=np.int32)
        term_timesteps = np.asarray(query_bank["timesteps"], dtype=np.int32)
        term_weights = np.asarray(query_bank["term_weights"], dtype=np.float64)
        num_queries = int(query_features.shape[0])
        num_dims = len(proj_dims)

        print("=" * 90)
        print("[stage:score_stream] query-cached streaming Traj-TracIn scorer")
        print(f"[stage:score_stream] query_artifacts={num_queries}")
        print(f"[stage:score_stream] terms={query_features.shape[1]} | cache_dim={cache_dim} | proj_dims={proj_dims}")
        print(f"[stage:score_stream] scored_points={len(picked)}")
        print(f"[stage:score_stream] artifact_path={stage_artifact_path}")
        print("=" * 90)

        scores_raw = np.zeros((num_dims, num_queries, len(picked)), dtype=np.float64)
        scores_query_l2 = np.zeros_like(scores_raw)
        scores_train_l2 = np.zeros_like(scores_raw)
        scores_query_train_l2 = np.zeros_like(scores_raw)
        term_score_variants_text = os.environ.get("TRAJ_TRACIN_STREAM_SAVE_TERM_SCORE_VARIANTS", "").strip()
        term_score_variant_aliases = {
            "raw": "scores_by_term_raw",
            "query_l2": "scores_by_term_query_l2_normalized",
            "query_l2_normalized": "scores_by_term_query_l2_normalized",
            "train_l2": "scores_by_term_train_l2_normalized",
            "train_l2_normalized": "scores_by_term_train_l2_normalized",
            "query_train_l2": "scores_by_term_query_train_l2_normalized",
            "query_train_l2_normalized": "scores_by_term_query_train_l2_normalized",
        }
        term_score_keys = []
        if term_score_variants_text:
            for part in term_score_variants_text.replace(",", " ").split():
                key = term_score_variant_aliases.get(part.strip())
                if key is None:
                    raise ValueError(
                        "Unknown TRAJ_TRACIN_STREAM_SAVE_TERM_SCORE_VARIANTS entry "
                        f"{part!r}; expected one of {sorted(term_score_variant_aliases)}"
                    )
                if key not in term_score_keys:
                    term_score_keys.append(key)
        term_scores_by_key = {
            key: np.zeros((query_features.shape[1], num_dims, num_queries, len(picked)), dtype=np.float32)
            for key in term_score_keys
        }
        if term_scores_by_key:
            print(
                "[stage:score_stream] saving per-term score contributions for variants="
                f"{tuple(key.replace('scores_by_term_', '') for key in term_score_keys)}"
            )

        def train_phi_one(p, x0_one, cond_one, rng_one, t_scalar):
            x0_one = x0_one[None, ...]
            if cond_one.ndim == 0:
                cond_one = cond_one[None]
            else:
                cond_one = cond_one[None, ...]

            def loss_fn(pp):
                return train_losses_at_dynamic_t_mc_vectorized(
                    adapter=adapter,
                    model=model,
                    params=pp,
                    schedule=schedule,
                    x0_batch=x0_one,
                    cond_batch=cond_one,
                    t_scalar=t_scalar,
                    num_mc_samples=cfg.train_mc_samples,
                    rng=rng_one,
                )[0]

            _loss, grads = jax.value_and_grad(loss_fn)(p)
            return projector(grads)

        batch_size_stream = max(1, int(cfg.score_batch_size))
        batch_starts = list(range(0, len(picked), batch_size_stream))
        stream_start = time.time()

        for ckpt_i, ckpt_path in enumerate(ckpts):
            term_ids = np.where(term_ckpt_indices == int(ckpt_i))[0]
            if term_ids.size == 0:
                continue
            ckpt_start = time.time()
            print(
                f"[stage:score_stream] checkpoint {ckpt_i + 1}/{len(ckpts)} | "
                f"terms={len(term_ids)} | {os.path.basename(ckpt_path)}"
            )
            try:
                state, _payload = adapter.restore_state(ckpt_path, state_template)
            except Exception as exc:
                if cfg.skip_unreadable_checkpoints:
                    print(f"[warning] skipping unreadable checkpoint: {ckpt_path}: {exc}")
                    continue
                raise
            params = tree_to_device(select_state_params(state, cfg.parameter_source), device)
            projector = build_countsketch_projector_jax(
                params,
                cache_dim,
                seed_parts=(cfg.seed, "traj_tracin_projection", ckpt_i),
                device=device,
            )
            train_phi_batch = jax.jit(jax.vmap(train_phi_one, in_axes=(None, 0, 0, 0, None)))

            for local_term_no, term_id in enumerate(term_ids, start=1):
                t_scalar = array_to_device(jnp.asarray(int(term_timesteps[term_id]), dtype=jnp.int32), device)
                q_raw_by_dim = []
                q_norm_by_dim = []
                for dim in proj_dims:
                    q_np = query_features[:, term_id, :dim]
                    q_norm_np = q_np / np.maximum(
                        np.linalg.norm(q_np, axis=-1, keepdims=True),
                        float(cfg.query_normalize_eps),
                    )
                    q_raw_by_dim.append(array_to_device(jnp.asarray(q_np, dtype=jnp.float32), device))
                    q_norm_by_dim.append(array_to_device(jnp.asarray(q_norm_np, dtype=jnp.float32), device))
                term_weight = float(term_weights[term_id])

                for start in batch_starts:
                    end = min(start + batch_size_stream, len(picked))
                    real_indices = picked[start:end]
                    padded_indices = pad_indices_to_batch(real_indices, batch_size_stream)
                    x_batch, cond_batch = make_train_batch(adapter, ds, padded_indices, device)
                    rngs = array_to_device(
                        jnp.stack(
                            [
                                jax.random.PRNGKey(
                                    cfg.seed + 700_000 * (ckpt_i + 1) + 10_000 * int(term_id) + start + j
                                )
                                for j in range(batch_size_stream)
                            ],
                            axis=0,
                        ),
                        device,
                    )
                    phi = train_phi_batch(params, x_batch, cond_batch, rngs, t_scalar)
                    phi = phi[: len(real_indices), :max_score_dim]
                    for dim_i, dim in enumerate(proj_dims):
                        phi_dim = phi[:, :dim]
                        phi_norm = phi_dim / jnp.maximum(
                            jnp.linalg.norm(phi_dim, axis=-1, keepdims=True),
                            jnp.asarray(float(cfg.query_normalize_eps), dtype=jnp.float32),
                        )

                        raw = phi_dim @ q_raw_by_dim[dim_i].T
                        query_l2 = phi_dim @ q_norm_by_dim[dim_i].T
                        train_l2 = phi_norm @ q_raw_by_dim[dim_i].T
                        both_l2 = phi_norm @ q_norm_by_dim[dim_i].T
                        raw.block_until_ready()
                        raw_np = np.asarray(jax.device_get(raw), dtype=np.float64).T
                        query_l2_np = np.asarray(jax.device_get(query_l2), dtype=np.float64).T
                        train_l2_np = np.asarray(jax.device_get(train_l2), dtype=np.float64).T
                        both_l2_np = np.asarray(jax.device_get(both_l2), dtype=np.float64).T
                        raw_contrib = term_weight * raw_np
                        query_l2_contrib = term_weight * query_l2_np
                        train_l2_contrib = term_weight * train_l2_np
                        both_l2_contrib = term_weight * both_l2_np
                        scores_raw[dim_i, :, start:end] += raw_contrib
                        scores_query_l2[dim_i, :, start:end] += query_l2_contrib
                        scores_train_l2[dim_i, :, start:end] += train_l2_contrib
                        scores_query_train_l2[dim_i, :, start:end] += both_l2_contrib
                        if "scores_by_term_raw" in term_scores_by_key:
                            term_scores_by_key["scores_by_term_raw"][term_id, dim_i, :, start:end] = raw_contrib.astype(np.float32)
                        if "scores_by_term_query_l2_normalized" in term_scores_by_key:
                            term_scores_by_key["scores_by_term_query_l2_normalized"][term_id, dim_i, :, start:end] = query_l2_contrib.astype(np.float32)
                        if "scores_by_term_train_l2_normalized" in term_scores_by_key:
                            term_scores_by_key["scores_by_term_train_l2_normalized"][term_id, dim_i, :, start:end] = train_l2_contrib.astype(np.float32)
                        if "scores_by_term_query_train_l2_normalized" in term_scores_by_key:
                            term_scores_by_key["scores_by_term_query_train_l2_normalized"][term_id, dim_i, :, start:end] = both_l2_contrib.astype(np.float32)

                if (
                    local_term_no == 1
                    or local_term_no == len(term_ids)
                    or local_term_no % max(1, int(cfg.progress_every)) == 0
                ):
                    elapsed = time.time() - ckpt_start
                    print(
                        f"[stage:score_stream] ckpt {ckpt_i + 1}/{len(ckpts)} "
                        f"term {local_term_no}/{len(term_ids)} | "
                        f"elapsed={format_seconds(elapsed)}"
                    )

            print(
                f"[stage:score_stream] checkpoint {ckpt_i + 1}/{len(ckpts)} done | "
                f"elapsed={format_seconds(time.time() - ckpt_start)}"
            )

        stream_payload = dict(
            scores_raw=scores_raw,
            scores_query_l2_normalized=scores_query_l2,
            scores_train_l2_normalized=scores_train_l2,
            scores_query_train_l2_normalized=scores_query_train_l2,
            score_indices=np.asarray(picked, dtype=np.int64),
            query_artifacts=np.asarray(query_bank["paths"]),
            proj_dims=np.asarray(proj_dims, dtype=np.int32),
            cache_dim=np.asarray(cache_dim, dtype=np.int32),
            term_ckpt_indices=term_ckpt_indices,
            term_timesteps=term_timesteps,
            term_weights=np.asarray(term_weights, dtype=np.float32),
            elapsed_sec=np.asarray(time.time() - stream_start, dtype=np.float64),
        )
        if term_scores_by_key:
            stream_payload["term_score_variants"] = np.asarray(
                [key.replace("scores_by_term_", "") for key in term_score_keys]
            )
            stream_payload.update(term_scores_by_key)
        save_npz_compressed_atomic(stage_artifact_path, **stream_payload)
        print(f"[saved] TrajTracIn stream score artifact: {stage_artifact_path}")
        return

    if stage_mode:
        from dtrak.algorithm import build_countsketch_projector_jax

        proj_dim = int(getattr(cfg, "proj_dim", int(os.environ.get("TRAJ_TRACIN_PROJ_DIM", "4096"))))
        stage_features = []
        stage_ckpt_indices = []
        stage_timesteps = []
        stage_snapshot_positions = []
        stage_ckpt_paths = []
        stage_term_weights = []
        used_ckpts_for_stage = []
        stage_part_dir = f"{stage_artifact_path}.parts" if stage_mode == "train" else None
        stage_total_terms = (len(ckpts) - 1 if uses_next_checkpoint_target else len(ckpts)) * int(
            cfg.num_traj_snapshots
        )
        stage_terms_done = 0
        stage_start_time = time.time()
        ckpt_shard_count = max(1, int(os.environ.get("TRAJ_TRACIN_CKPT_SHARD_COUNT", "1")))
        ckpt_shard_index = int(os.environ.get("TRAJ_TRACIN_CKPT_SHARD_INDEX", "0"))
        aggregate_train_timestamps = os.environ.get(
            "TRAJ_TRACIN_TRAIN_AGGREGATE_TIMESTAMPS", "0"
        ) in ("1", "true", "True", "yes")
        if ckpt_shard_index < 0 or ckpt_shard_index >= ckpt_shard_count:
            raise ValueError(
                "TRAJ_TRACIN_CKPT_SHARD_INDEX must be in "
                f"[0, {ckpt_shard_count}), got {ckpt_shard_index}"
            )
        skip_stage_merge = os.environ.get("TRAJ_TRACIN_SKIP_STAGE_MERGE", "0") in (
            "1",
            "true",
            "True",
            "yes",
        )

        for ckpt_i, ckpt_path in enumerate(ckpts):
            if uses_next_checkpoint_target and ckpt_i + 1 >= len(ckpts):
                print(
                    f"[stage:{stage_mode}] skipping final checkpoint "
                    f"{ckpt_i + 1}/{len(ckpts)}: no next-checkpoint query target",
                    flush=True,
                )
                continue
            if stage_mode == "train" and ckpt_shard_count > 1 and ckpt_i % ckpt_shard_count != ckpt_shard_index:
                print(
                    f"[stage:{stage_mode}] checkpoint shard skip "
                    f"{ckpt_i + 1}/{len(ckpts)} for shard "
                    f"{ckpt_shard_index}/{ckpt_shard_count}",
                    flush=True,
                )
                continue
            stage_part_path = (
                os.path.join(stage_part_dir, f"ckpt_{ckpt_i:04d}.npz")
                if stage_part_dir is not None
                else None
            )
            if stage_part_path is not None and os.path.isfile(stage_part_path):
                print(
                    f"[stage:{stage_mode}] skip existing checkpoint part "
                    f"{ckpt_i + 1}/{len(ckpts)}: {stage_part_path}",
                    flush=True,
                )
                continue
            ckpt_lr_weight = tracin_checkpoint_lr_weight(cfg, ckpt_path, ckpt_i, len(ckpts), len(ds))
            print(
                f"[stage:{stage_mode}] TrajTracIn checkpoint {ckpt_i + 1}/{len(ckpts)} | "
                f"{os.path.basename(ckpt_path)} | lr_weight={ckpt_lr_weight:.8g} "
                f"schedule={cfg.tracin_lr_schedule}",
                flush=True,
            )
            stage_ckpt_start = time.time()
            try:
                state, _payload = adapter.restore_state(ckpt_path, state_template)
            except Exception as exc:
                if cfg.skip_unreadable_checkpoints:
                    print(f"[warning] skipping unreadable checkpoint: {ckpt_path}: {exc}", flush=True)
                    continue
                raise
            params = tree_to_device(select_state_params(state, cfg.parameter_source), device)
            print(f"[stage:{stage_mode}] checkpoint {ckpt_i + 1}/{len(ckpts)} restored", flush=True)
            projector = build_countsketch_projector_jax(
                params,
                proj_dim,
                seed_parts=(cfg.seed, "traj_tracin_projection", ckpt_i),
                device=device,
            )
            eps_fn = lambda p, x, t, c: adapter.eps_apply(model, p, x, t, c)
            if stage_mode == "train":
                T = int(schedule.betas.shape[0])
                ddim_ts = np.linspace(T - 1, 0, int(cfg.ddim_steps), dtype=np.int32)
                pos_seq = np.asarray(
                    select_snapshot_positions(
                        int(cfg.ddim_steps),
                        int(cfg.num_traj_snapshots),
                        cfg.traj_snapshot_positions,
                    ),
                    dtype=np.int32,
                )
                t_seq = np.asarray([int(ddim_ts[int(pos)]) for pos in pos_seq], dtype=np.int32)
                xt_refs = []
                print(
                    f"[stage:{stage_mode}] using timestamp schedule only | "
                    f"snapshots={len(t_seq)}; reference trajectory states are not needed for train loss gradients",
                    flush=True,
                )
            elif precomputed_traj is not None:
                xt_refs_raw, t_seq, pos_seq, _ = precomputed_traj
                if (
                    stage_mode == "query"
                    and os.environ.get("TRAJ_QUERY_USE_CONFIG_SNAPSHOTS", "0")
                    in ("1", "true", "True", "yes")
                ):
                    xt_refs_raw, t_seq, pos_seq = select_precomputed_trajectory_snapshots(
                        xt_refs_raw,
                        t_seq,
                        pos_seq,
                        num_keep=cfg.num_traj_snapshots,
                        snapshot_positions=cfg.traj_snapshot_positions,
                    )
                xt_refs = [array_to_device(x, device) for x in xt_refs_raw]
            else:
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
            print(
                f"[stage:{stage_mode}] checkpoint {ckpt_i + 1}/{len(ckpts)} trajectory ready | "
                f"snapshots={len(t_seq)}",
                flush=True,
            )

            if stage_mode == "query":
                use_future_residual_mixture = cfg.query_objective == "trajectory_future_residual_mixture"
                if use_future_residual_mixture:
                    query_grad_chunk_fn = make_future_residual_mixture_grad_chunk_fn(adapter, model)
                    eps_cache = load_or_build_future_noise_cache(xt_refs, t_seq, pos_seq)
                    future_count = len(ckpts) - ckpt_i - 1
                    if future_count <= 0:
                        raise RuntimeError("future residual mixture reached a checkpoint with no future targets")
                    print(
                        f"[stage:query] future residual mixture targets={future_count} "
                        f"(next + {max(0, future_count - 1)} later/ref checkpoints) | "
                        f"cache={future_noise_cache_path}",
                        flush=True,
                    )
                else:
                    query_grad_chunk_fn = make_query_grad_chunk_fn(adapter, model, cfg.query_objective)
                    query_target_params = query_target_params_for_checkpoint(ckpt_i)
                snapshot_chunk_size = max(1, int(cfg.snapshot_chunk_size))
                for chunk_start in range(0, len(t_seq), snapshot_chunk_size):
                    chunk_end = min(chunk_start + snapshot_chunk_size, len(t_seq))
                    chunk_ids = list(range(chunk_start, chunk_end))
                    xt_chunk = array_to_device(jnp.stack([xt_refs[i] for i in chunk_ids], axis=0), device)
                    t_chunk = array_to_device(jnp.asarray([int(t_seq[i]) for i in chunk_ids], dtype=jnp.int32), device)
                    if use_future_residual_mixture:
                        future_eps_chunk_np = np.stack(
                            [
                                eps_cache[ckpt_i + 1 :, snap_id].astype(np.float32, copy=False)
                                for snap_id in chunk_ids
                            ],
                            axis=0,
                        )
                        future_eps_chunk = array_to_device(jnp.asarray(future_eps_chunk_np), device)
                        query_grads = query_grad_chunk_fn(
                            params,
                            xt_chunk,
                            t_chunk,
                            query_cond,
                            future_eps_chunk,
                        )
                    else:
                        query_grads = query_grad_chunk_fn(
                            params,
                            query_target_params,
                            reference_params,
                            xt_chunk,
                            t_chunk,
                            query_cond,
                        )
                    for local_i, snap_id in enumerate(chunk_ids):
                        one_grad = jax.tree_util.tree_map(lambda x: x[local_i], query_grads)
                        stage_features.append(np.asarray(projector(one_grad), dtype=np.float32))
                        stage_ckpt_indices.append(int(ckpt_i))
                        stage_timesteps.append(int(t_seq[snap_id]))
                        stage_snapshot_positions.append(int(pos_seq[snap_id]))
                        stage_ckpt_paths.append(str(ckpt_path))
                        stage_term_weights.append(ckpt_lr_weight / float(max(1, len(t_seq))))
                    stage_terms_done += len(chunk_ids)
                    query_total_terms = (len(ckpts) - 1 if uses_next_checkpoint_target else len(ckpts)) * len(t_seq)
                    print(
                        f"[stage:query] checkpoint {ckpt_i + 1}/{len(ckpts)} "
                        f"snapshot {chunk_start + 1}-{chunk_end}/{len(t_seq)} | "
                        f"terms={stage_terms_done}/{query_total_terms} | "
                        f"elapsed={format_seconds(time.time() - stage_start_time)}",
                        flush=True,
                    )
            else:
                train_phi_terms = []
                train_ckpt_indices = []
                train_timesteps = []
                train_snapshot_positions = []
                train_ckpt_paths = []
                train_term_weights = []

                if aggregate_train_timestamps:
                    aggregate_timestamp_count = max(
                        1,
                        int(os.environ.get("TRAJ_TRACIN_TRAIN_AGGREGATE_NUM_TIMESTEPS", "10")),
                    )
                    if aggregate_timestamp_count < len(t_seq):
                        aggregate_positions = np.linspace(
                            0,
                            len(t_seq) - 1,
                            aggregate_timestamp_count,
                            dtype=np.int32,
                        )
                        train_t_seq = np.asarray(
                            [int(t_seq[int(pos)]) for pos in aggregate_positions],
                            dtype=np.int32,
                        )
                    else:
                        train_t_seq = np.asarray([int(t) for t in t_seq], dtype=np.int32)
                    timestamp_chunk_size = max(
                        1,
                        int(
                            os.environ.get(
                                "TRAJ_TRACIN_TRAIN_TIMESTAMP_CHUNK_SIZE",
                                str(len(train_t_seq)),
                            )
                        ),
                    )
                    t_chunks = [
                        array_to_device(
                            jnp.asarray(
                                [int(t) for t in train_t_seq[start : start + timestamp_chunk_size]],
                                dtype=jnp.int32,
                            ),
                            device,
                        )
                        for start in range(0, len(train_t_seq), timestamp_chunk_size)
                    ]

                    def train_phi_one_aggregate_chunk(p, x0_one, cond_one, rng_one, t_values_chunk):
                        x0_one = x0_one[None, ...]
                        if cond_one.ndim == 0:
                            cond_one = cond_one[None]
                        else:
                            cond_one = cond_one[None, ...]

                        def loss_fn(pp):
                            return train_losses_at_t_sequence_mc_vectorized(
                                adapter=adapter,
                                model=model,
                                params=pp,
                                schedule=schedule,
                                x0_batch=x0_one,
                                cond_batch=cond_one,
                                t_values=t_values_chunk,
                                num_mc_samples=cfg.train_mc_samples,
                                rng=rng_one,
                            )[0]

                        _loss, grads = jax.value_and_grad(loss_fn)(p)
                        return projector(grads)

                    train_phi_batch_aggregate_chunk = jax.jit(
                        jax.vmap(train_phi_one_aggregate_chunk, in_axes=(None, 0, 0, 0, None))
                    )
                    bs_stage = max(1, int(cfg.score_batch_size))
                    total_batches = (len(picked) + bs_stage - 1) // bs_stage
                    progress_every = max(
                        1,
                        int(os.environ.get("TRAJ_TRACIN_TRAIN_BATCH_LOG_EVERY", "10")),
                    )
                    term_features = np.empty((len(picked), proj_dim), dtype=np.float32)
                    for batch_id, start in enumerate(range(0, len(picked), bs_stage), start=1):
                        end = min(len(picked), start + bs_stage)
                        real_indices = picked[start:end]
                        padded_indices = pad_indices_to_batch(real_indices, bs_stage)
                        x_batch, cond_batch = make_train_batch(adapter, ds, padded_indices, device)
                        phi_accum = np.zeros((bs_stage, proj_dim), dtype=np.float64)
                        for chunk_id, t_values_chunk in enumerate(t_chunks):
                            chunk_len = int(t_values_chunk.shape[0])
                            rngs = array_to_device(
                                jnp.stack(
                                    [
                                        jax.random.PRNGKey(
                                            cfg.seed
                                            + 700_000 * (ckpt_i + 1)
                                            + 10_000 * chunk_id
                                            + start
                                            + j
                                        )
                                        for j in range(bs_stage)
                                    ],
                                    axis=0,
                                ),
                                device,
                            )
                            phi_chunk = train_phi_batch_aggregate_chunk(
                                params, x_batch, cond_batch, rngs, t_values_chunk
                            )
                            phi_chunk.block_until_ready()
                            phi_accum += (
                                np.asarray(phi_chunk, dtype=np.float64)
                                * (float(chunk_len) / float(max(1, len(train_t_seq))))
                            )
                        term_features[start:end] = phi_accum[: end - start].astype(np.float32)
                        if batch_id == 1 or batch_id == total_batches or batch_id % progress_every == 0:
                            print(
                                f"[stage:train] checkpoint-level MC batch {batch_id}/{total_batches} | "
                                f"ckpt={ckpt_i + 1}/{len(ckpts)} | datapoints={end}/{len(picked)} | "
                                f"aggregate_timestamps={len(train_t_seq)} | "
                                f"timestamp_chunk_size={timestamp_chunk_size} | "
                                f"elapsed={format_seconds(time.time() - stage_ckpt_start)} | "
                                f"total_elapsed={format_seconds(time.time() - stage_start_time)}",
                                flush=True,
                            )
                    train_phi_terms.append(term_features)
                    train_ckpt_indices.append(int(ckpt_i))
                    train_timesteps.append(-1)
                    train_snapshot_positions.append(-1)
                    train_ckpt_paths.append(str(ckpt_path))
                    train_term_weights.append(1.0)
                    stage_terms_done += 1
                    print(
                        f"[stage:train] checkpoint-level MC loss ckpt={ckpt_i + 1}/{len(ckpts)} | "
                        f"terms={stage_terms_done}/{len(ckpts) - 1 if uses_next_checkpoint_target else len(ckpts)} | "
                        f"aggregate_timestamps={len(train_t_seq)} | "
                        f"mc_per_timestamp={cfg.train_mc_samples} | "
                        f"elapsed={format_seconds(time.time() - stage_start_time)}",
                        flush=True,
                    )
                    if stage_part_path is None:
                        stage_features.extend(train_phi_terms)
                        stage_ckpt_indices.extend(train_ckpt_indices)
                        stage_timesteps.extend(train_timesteps)
                        stage_snapshot_positions.extend(train_snapshot_positions)
                        stage_ckpt_paths.extend(train_ckpt_paths)
                        stage_term_weights.extend(train_term_weights)
                    else:
                        save_npz_compressed_atomic(
                            stage_part_path,
                            train_features=np.stack(train_phi_terms, axis=0).astype(np.float32),
                            score_indices=np.asarray(picked, dtype=np.int64),
                            ckpt_indices=np.asarray(train_ckpt_indices, dtype=np.int32),
                            timesteps=np.asarray(train_timesteps, dtype=np.int32),
                            snapshot_positions=np.asarray(train_snapshot_positions, dtype=np.int32),
                            term_weights=np.asarray(train_term_weights, dtype=np.float32),
                            ckpt_paths=np.asarray(train_ckpt_paths),
                            proj_dim=np.asarray(proj_dim, dtype=np.int32),
                            train_timestamp_aggregation=np.asarray("checkpoint_mc_loss"),
                            train_timestamp_count=np.asarray(len(train_t_seq), dtype=np.int32),
                            train_timesteps_used=np.asarray(train_t_seq, dtype=np.int32),
                            train_timestamp_chunk_size=np.asarray(timestamp_chunk_size, dtype=np.int32),
                        )
                        print(
                            f"[stage:train] saved checkpoint-level MC part "
                            f"{ckpt_i + 1}/{len(ckpts)}: {stage_part_path}",
                            flush=True,
                        )
                    used_ckpts_for_stage.append(ckpt_path)
                    print(
                        f"[stage:{stage_mode}] checkpoint {ckpt_i + 1}/{len(ckpts)} done | "
                        f"elapsed={format_seconds(time.time() - stage_ckpt_start)} | "
                        f"total_elapsed={format_seconds(time.time() - stage_start_time)}",
                        flush=True,
                    )
                    continue

                def train_phi_one(p, x0_one, cond_one, rng_one, t_scalar):
                    x0_one = x0_one[None, ...]
                    if cond_one.ndim == 0:
                        cond_one = cond_one[None]
                    else:
                        cond_one = cond_one[None, ...]

                    def loss_fn(pp):
                        return train_losses_at_dynamic_t_mc_vectorized(
                            adapter=adapter,
                            model=model,
                            params=pp,
                            schedule=schedule,
                            x0_batch=x0_one,
                            cond_batch=cond_one,
                            t_scalar=t_scalar,
                            num_mc_samples=cfg.train_mc_samples,
                            rng=rng_one,
                        )[0]

                    _loss, grads = jax.value_and_grad(loss_fn)(p)
                    return projector(grads)

                train_phi_batch = jax.jit(jax.vmap(train_phi_one, in_axes=(None, 0, 0, 0, None)))
                bs_stage = max(1, int(cfg.score_batch_size))
                for snap_id, t_value in enumerate(t_seq):
                    term_features = np.empty((len(picked), proj_dim), dtype=np.float32)
                    t_scalar = array_to_device(jnp.asarray(int(t_value), dtype=jnp.int32), device)
                    for start in range(0, len(picked), bs_stage):
                        end = min(len(picked), start + bs_stage)
                        real_indices = picked[start:end]
                        padded_indices = pad_indices_to_batch(real_indices, bs_stage)
                        x_batch, cond_batch = make_train_batch(adapter, ds, padded_indices, device)
                        rngs = array_to_device(
                            jnp.stack(
                                [
                                    jax.random.PRNGKey(cfg.seed + 700_000 * (ckpt_i + 1) + 10_000 * snap_id + start + j)
                                    for j in range(bs_stage)
                                ],
                                axis=0,
                            ),
                            device,
                        )
                        phi_batch = train_phi_batch(params, x_batch, cond_batch, rngs, t_scalar)
                        phi_batch.block_until_ready()
                        term_features[start:end] = np.asarray(phi_batch[: end - start], dtype=np.float32)
                    train_phi_terms.append(term_features)
                    train_ckpt_indices.append(int(ckpt_i))
                    train_timesteps.append(int(t_value))
                    train_snapshot_positions.append(int(pos_seq[snap_id]))
                    train_ckpt_paths.append(str(ckpt_path))
                    train_term_weights.append(ckpt_lr_weight / float(max(1, len(t_seq))))
                    stage_terms_done += 1
                    print(
                        f"[stage:train] TrajTracIn ckpt={ckpt_i + 1} snapshot={snap_id + 1}/{len(t_seq)} | "
                        f"terms={stage_terms_done}/{stage_total_terms} | "
                        f"elapsed={format_seconds(time.time() - stage_start_time)}",
                        flush=True,
                    )
                if stage_part_path is None:
                    stage_features.extend(train_phi_terms)
                    stage_ckpt_indices.extend(train_ckpt_indices)
                    stage_timesteps.extend(train_timesteps)
                    stage_snapshot_positions.extend(train_snapshot_positions)
                    stage_ckpt_paths.extend(train_ckpt_paths)
                    stage_term_weights.extend(train_term_weights)
                else:
                    save_npz_compressed_atomic(
                        stage_part_path,
                        train_features=np.stack(train_phi_terms, axis=0).astype(np.float32),
                        score_indices=np.asarray(picked, dtype=np.int64),
                        ckpt_indices=np.asarray(train_ckpt_indices, dtype=np.int32),
                        timesteps=np.asarray(train_timesteps, dtype=np.int32),
                        snapshot_positions=np.asarray(train_snapshot_positions, dtype=np.int32),
                        term_weights=np.asarray(train_term_weights, dtype=np.float32),
                        ckpt_paths=np.asarray(train_ckpt_paths),
                        proj_dim=np.asarray(proj_dim, dtype=np.int32),
                    )
                    print(
                        f"[stage:train] saved checkpoint part "
                        f"{ckpt_i + 1}/{len(ckpts)}: {stage_part_path}",
                        flush=True,
                    )
            used_ckpts_for_stage.append(ckpt_path)
            print(
                f"[stage:{stage_mode}] checkpoint {ckpt_i + 1}/{len(ckpts)} done | "
                f"elapsed={format_seconds(time.time() - stage_ckpt_start)} | "
                f"total_elapsed={format_seconds(time.time() - stage_start_time)}",
                flush=True,
            )

        if stage_mode == "train" and stage_part_dir is not None:
            if skip_stage_merge:
                print(
                    f"[stage:train] skipping merge by request; checkpoint parts remain under {stage_part_dir}",
                    flush=True,
                )
                return
            part_paths = [
                os.path.join(stage_part_dir, f"ckpt_{ckpt_i:04d}.npz")
                for ckpt_i in range(len(ckpts))
                if os.path.isfile(os.path.join(stage_part_dir, f"ckpt_{ckpt_i:04d}.npz"))
            ]
            if not part_paths:
                raise RuntimeError(f"No TrajTracIn train checkpoint parts were produced under {stage_part_dir}.")
            expected_stage_parts = len(ckpts) - 1 if uses_next_checkpoint_target else len(ckpts)
            if len(part_paths) != expected_stage_parts:
                missing_parts = [
                    os.path.join(stage_part_dir, f"ckpt_{ckpt_i:04d}.npz")
                    for ckpt_i in range(expected_stage_parts)
                    if not os.path.isfile(os.path.join(stage_part_dir, f"ckpt_{ckpt_i:04d}.npz"))
                ]
                raise RuntimeError(
                    "TrajTracIn train checkpoint parts are incomplete: "
                    f"{len(part_paths)}/{expected_stage_parts} present. First missing: {missing_parts[:3]}"
                )
            print(f"[stage:train] merging {len(part_paths)} checkpoint parts into {stage_artifact_path}")
            train_features_parts = []
            ckpt_indices_parts = []
            timesteps_parts = []
            snapshot_positions_parts = []
            ckpt_paths_parts = []
            term_weights_parts = []
            score_indices = None
            for part_path in part_paths:
                with np.load(part_path, allow_pickle=True) as part:
                    part_score_indices = np.asarray(part["score_indices"], dtype=np.int64)
                    if score_indices is None:
                        score_indices = part_score_indices
                    elif not np.array_equal(score_indices, part_score_indices):
                        raise ValueError(f"score_indices mismatch in checkpoint part: {part_path}")
                    train_features_parts.append(np.asarray(part["train_features"], dtype=np.float32))
                    ckpt_indices_parts.append(np.asarray(part["ckpt_indices"], dtype=np.int32))
                    timesteps_parts.append(np.asarray(part["timesteps"], dtype=np.int32))
                    snapshot_positions_parts.append(np.asarray(part["snapshot_positions"], dtype=np.int32))
                    ckpt_paths_parts.append(np.asarray(part["ckpt_paths"]))
                    term_weights_parts.append(np.asarray(part["term_weights"], dtype=np.float32))
            save_npz_compressed_atomic(
                stage_artifact_path,
                train_features=np.concatenate(train_features_parts, axis=0).astype(np.float32),
                score_indices=np.asarray(score_indices, dtype=np.int64),
                ckpt_indices=np.concatenate(ckpt_indices_parts, axis=0).astype(np.int32),
                timesteps=np.concatenate(timesteps_parts, axis=0).astype(np.int32),
                snapshot_positions=np.concatenate(snapshot_positions_parts, axis=0).astype(np.int32),
                term_weights=np.concatenate(term_weights_parts, axis=0).astype(np.float32),
                ckpt_paths=np.concatenate(ckpt_paths_parts, axis=0),
                proj_dim=np.asarray(proj_dim, dtype=np.int32),
                query_objective=np.asarray(cfg.query_objective),
                query_target_checkpoint=np.asarray(
                    "next_checkpoint" if uses_next_checkpoint_target else "reference_checkpoint"
                ),
            )
            print(f"[saved] TrajTracIn train artifact: {stage_artifact_path}")
            return

        if not stage_features:
            raise RuntimeError(f"No TrajTracIn {stage_mode} stage features were produced.")
        if stage_mode == "query":
            query_payload = dict(
                query_features=np.stack(stage_features, axis=0).astype(np.float32),
                ckpt_indices=np.asarray(stage_ckpt_indices, dtype=np.int32),
                timesteps=np.asarray(stage_timesteps, dtype=np.int32),
                snapshot_positions=np.asarray(stage_snapshot_positions, dtype=np.int32),
                term_weights=np.asarray(stage_term_weights, dtype=np.float32),
                ckpt_paths=np.asarray(stage_ckpt_paths),
                proj_dim=np.asarray(proj_dim, dtype=np.int32),
                query_objective=np.asarray(cfg.query_objective),
                query_target_checkpoint=np.asarray(
                    "next_checkpoint" if uses_next_checkpoint_target else "reference_checkpoint"
                ),
            )
            if cfg.query_objective == "trajectory_future_residual_mixture":
                query_payload.update(
                    future_noise_cache_path=np.asarray("" if future_noise_cache_path is None else future_noise_cache_path),
                    future_mix_gamma=np.asarray(float(os.environ.get("TRAJ_TRACIN_FUTURE_MIX_GAMMA", "1.0")), dtype=np.float32),
                    future_mix_eps=np.asarray(float(os.environ.get("TRAJ_TRACIN_FUTURE_MIX_EPS", "1e-8")), dtype=np.float32),
                    future_mix_rule=np.asarray(
                        "alpha_next=1; alpha_future=gamma*clip((1-cos(normalized_next,normalized_future))/2,0,1); "
                        "alphas normalized to sum 1; residuals RMS-normalized"
                    ),
                )
            save_npz_compressed_atomic(stage_artifact_path, **query_payload)
        else:
            save_npz_compressed_atomic(
                stage_artifact_path,
                train_features=np.stack(stage_features, axis=0).astype(np.float32),
                score_indices=np.asarray(picked, dtype=np.int64),
                ckpt_indices=np.asarray(stage_ckpt_indices, dtype=np.int32),
                timesteps=np.asarray(stage_timesteps, dtype=np.int32),
                snapshot_positions=np.asarray(stage_snapshot_positions, dtype=np.int32),
                term_weights=np.asarray(stage_term_weights, dtype=np.float32),
                ckpt_paths=np.asarray(stage_ckpt_paths),
                proj_dim=np.asarray(proj_dim, dtype=np.int32),
                query_objective=np.asarray(cfg.query_objective),
                query_target_checkpoint=np.asarray(
                    "next_checkpoint" if uses_next_checkpoint_target else "reference_checkpoint"
                ),
            )
        print(f"[saved] TrajTracIn {stage_mode} artifact: {stage_artifact_path}")
        return

    scores = np.zeros((len(picked),), dtype=np.float64)
    query_normalized_scores = (
        np.zeros((len(picked),), dtype=np.float64)
        if cfg.save_query_normalized_scores
        else None
    )
    full_term_score_variants_text = os.environ.get("TRAJ_TRACIN_FULL_SAVE_TERM_SCORE_VARIANTS", "").strip()
    full_term_score_variant_aliases = {
        "raw": "scores_by_term_raw",
        "query_l2": "scores_by_term_query_l2_normalized",
        "query_l2_normalized": "scores_by_term_query_l2_normalized",
    }
    full_term_score_keys = []
    if full_term_score_variants_text:
        for part in full_term_score_variants_text.replace(",", " ").split():
            key = full_term_score_variant_aliases.get(part.strip())
            if key is None:
                raise ValueError(
                    "Unknown TRAJ_TRACIN_FULL_SAVE_TERM_SCORE_VARIANTS entry "
                    f"{part!r}; expected one of {sorted(full_term_score_variant_aliases)}"
                )
            if key == "scores_by_term_query_l2_normalized" and not cfg.save_query_normalized_scores:
                raise ValueError(
                    "TRAJ_TRACIN_FULL_SAVE_TERM_SCORE_VARIANTS includes query_l2_normalized, "
                    "but cfg.save_query_normalized_scores is False."
                )
            if key not in full_term_score_keys:
                full_term_score_keys.append(key)
    num_term_ckpts = len(ckpts) - 1 if uses_next_checkpoint_target else len(ckpts)
    full_term_scores_by_key = {
        key: np.zeros((num_term_ckpts * int(cfg.num_traj_snapshots), len(picked)), dtype=np.float32)
        for key in full_term_score_keys
    }
    full_term_ckpt_indices: List[int] = []
    full_term_timesteps: List[int] = []
    full_term_snapshot_positions: List[int] = []
    full_term_weights: List[float] = []
    full_term_ckpt_paths: List[str] = []
    if full_term_scores_by_key:
        print(
            "[setup] saving full-dim per-term score contributions for variants="
            f"{tuple(key.replace('scores_by_term_', '') for key in full_term_score_keys)}"
        )
    full_aggregate_train_timestamps = os.environ.get(
        "TRAJ_TRACIN_FULL_AGGREGATE_TRAIN_TIMESTAMPS", "0"
    ) in ("1", "true", "True", "yes")
    full_aggregate_timestamp_count = max(
        1,
        int(os.environ.get("TRAJ_TRACIN_FULL_AGGREGATE_NUM_TIMESTEPS", "10")),
    )
    if full_aggregate_train_timestamps and full_term_scores_by_key:
        raise ValueError(
            "TRAJ_TRACIN_FULL_AGGREGATE_TRAIN_TIMESTAMPS=1 is incompatible with "
            "TRAJ_TRACIN_FULL_SAVE_TERM_SCORE_VARIANTS, because the train gradient is "
            "collapsed across timesteps."
        )

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
    total_points_all_ckpts = num_term_ckpts * len(picked)
    processed_points_all_ckpts = 0
    used_ckpts = []
    skipped_ckpts = []

    for ckpt_i, ckpt_path in enumerate(ckpts):
        if uses_next_checkpoint_target and ckpt_i + 1 >= len(ckpts):
            print(
                f"[checkpoint {ckpt_i + 1}/{len(ckpts)}] skipping final checkpoint: "
                "no next-checkpoint query target"
            )
            continue
        ckpt_start = time.time()
        ckpt_name = os.path.basename(ckpt_path)
        ckpt_lr_weight = tracin_checkpoint_lr_weight(cfg, ckpt_path, ckpt_i, len(ckpts), len(ds))

        print("\n" + "-" * 90)
        print(f"[checkpoint {ckpt_i + 1}/{len(ckpts)}] starting {ckpt_name}")
        print(
            f"[checkpoint {ckpt_i + 1}/{len(ckpts)}] "
            f"lr_weight={ckpt_lr_weight:.8g} schedule={cfg.tracin_lr_schedule}"
        )
        if os.path.abspath(ckpt_path) == os.path.abspath(reference_ckpt):
            print(
                "[warning] this checkpoint is theta_ref, so f_noise=0 and its "
                "query gradient is exactly zero; nonzero attribution requires "
                "at least one checkpoint different from reference_ckpt."
            )
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
        params = tree_to_device(select_state_params(state, cfg.parameter_source), device)
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
        print(f"[checkpoint {ckpt_i + 1}/{len(ckpts)}] preparing jitted query-gradient chunk function...")
        query_grad_chunk_fn = make_query_grad_chunk_fn(adapter, model, cfg.query_objective)
        query_target_params = query_target_params_for_checkpoint(ckpt_i)
        if full_aggregate_train_timestamps:
            print(
                f"[checkpoint {ckpt_i + 1}/{len(ckpts)}] preparing jitted aggregate train-loss scorer "
                f"| aggregate_timesteps={full_aggregate_timestamp_count} | mc_per_timestep={cfg.train_mc_samples}"
            )
            score_t_sequence_batch_fn = make_score_t_sequence_batch_fn(
                adapter=adapter,
                model=model,
                schedule=schedule,
                train_mc_samples=cfg.train_mc_samples,
                return_query_normalized=bool(cfg.save_query_normalized_scores),
                query_normalize_eps=float(cfg.query_normalize_eps),
            )
        else:
            score_t_sequence_batch_fn = None
        print(f"[checkpoint {ckpt_i + 1}/{len(ckpts)}] preparing jitted snapshot chunk scorer...")
        score_snapshot_chunk_fn = make_score_snapshot_chunk_batch_fn(
            adapter=adapter,
            model=model,
            schedule=schedule,
            train_mc_samples=cfg.train_mc_samples,
            return_query_normalized=bool(cfg.save_query_normalized_scores),
            query_normalize_eps=float(cfg.query_normalize_eps),
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
        snap_weight = ckpt_lr_weight / float(max(1, len(t_seq)))
        report_every_batches = max(1, int(math.ceil(cfg.progress_every / batch_size)))
        snapshot_chunk_size = max(1, int(cfg.snapshot_chunk_size))

        if full_aggregate_train_timestamps:
            print(
                f"[checkpoint {ckpt_i + 1}/{len(ckpts)}] aggregating query gradients over "
                f"{len(t_seq)} trajectory snapshots..."
            )
            query_grad_total = None
            for chunk_start in range(0, len(t_seq), snapshot_chunk_size):
                chunk_end = min(chunk_start + snapshot_chunk_size, len(t_seq))
                chunk_ids = list(range(chunk_start, chunk_end))
                xt_chunk = array_to_device(jnp.stack([xt_refs[i] for i in chunk_ids], axis=0), device)
                t_chunk = array_to_device(
                    jnp.asarray([int(t_seq[i]) for i in chunk_ids], dtype=jnp.int32),
                    device,
                )
                query_grads = query_grad_chunk_fn(
                    params,
                    query_target_params,
                    reference_params,
                    xt_chunk,
                    t_chunk,
                    query_cond,
                )
                query_grads = tree_to_device(query_grads, device)
                chunk_sum = jax.tree_util.tree_map(lambda x: jnp.sum(x, axis=0), query_grads)
                query_grad_total = chunk_sum if query_grad_total is None else tree_add(query_grad_total, chunk_sum)
                print(
                    f"[checkpoint {ckpt_i + 1}/{len(ckpts)}] query-gradient aggregate "
                    f"snapshots={chunk_end}/{len(t_seq)}",
                    flush=True,
                )
            if query_grad_total is None:
                raise RuntimeError("No query gradients were produced for aggregate full-dim scoring.")
            query_grad_total = tree_scalar_mul(query_grad_total, 1.0 / float(max(1, len(t_seq))))
            query_grad_total = tree_to_device(query_grad_total, device)

            if full_aggregate_timestamp_count < len(t_seq):
                aggregate_positions = np.linspace(
                    0,
                    len(t_seq) - 1,
                    full_aggregate_timestamp_count,
                    dtype=np.int32,
                )
                train_t_seq = np.asarray(
                    [int(t_seq[int(pos)]) for pos in aggregate_positions],
                    dtype=np.int32,
                )
            else:
                train_t_seq = np.asarray([int(t) for t in t_seq], dtype=np.int32)
            train_t_values = array_to_device(jnp.asarray(train_t_seq, dtype=jnp.int32), device)
            print(
                f"[checkpoint {ckpt_i + 1}/{len(ckpts)}] train loss gradient uses "
                f"{len(train_t_seq)} timesteps: {train_t_seq[:min(10, len(train_t_seq))].tolist()}",
                flush=True,
            )

            score_iter = iter_with_tqdm(
                range(len(batch_starts)),
                total=len(batch_starts),
                desc=f"Checkpoint {ckpt_i + 1}/{len(ckpts)} aggregate",
                enabled=cfg.use_tqdm,
            )
            for batch_no, start in enumerate(batch_starts, start=1):
                end = min(start + batch_size, len(picked))
                real_indices = picked[start:end]
                padded_indices = pad_indices_to_batch(real_indices, batch_size)
                x_batch, cond_batch = make_train_batch(adapter, ds, padded_indices, device)
                if batch_no == 1:
                    print(
                        "[device-check] "
                        f"x_batch={array_device_str(x_batch)} | "
                        f"cond_batch={array_device_str(cond_batch)}"
                    )
                rng = array_to_device(
                    jax.random.PRNGKey(cfg.seed + 100000 * ckpt_i + 9000 * batch_no),
                    device,
                )
                assert score_t_sequence_batch_fn is not None
                aggregate_scores_out = score_t_sequence_batch_fn(
                    params,
                    query_grad_total,
                    x_batch,
                    cond_batch,
                    rng,
                    train_t_values,
                )
                if cfg.save_query_normalized_scores:
                    raw_scores, normalized_scores = aggregate_scores_out
                    raw_scores.block_until_ready()
                    normalized_scores.block_until_ready()
                    batch_scores = np.asarray(jax.device_get(raw_scores))[: len(real_indices)]
                    batch_query_normalized_scores = np.asarray(jax.device_get(normalized_scores))[: len(real_indices)]
                else:
                    aggregate_scores_out.block_until_ready()
                    batch_scores = np.asarray(jax.device_get(aggregate_scores_out))[: len(real_indices)]
                    batch_query_normalized_scores = None
                batch_contrib = ckpt_lr_weight * batch_scores.astype(np.float64)
                scores[start:end] += batch_contrib
                if query_normalized_scores is not None and batch_query_normalized_scores is not None:
                    query_normalized_scores[start:end] += ckpt_lr_weight * batch_query_normalized_scores.astype(np.float64)

                running_sum += float(scores[start:end].sum())
                batch_min = float(scores[start:end].min()) if len(real_indices) else 0.0
                batch_max = float(scores[start:end].max()) if len(real_indices) else 0.0
                running_min = batch_min if running_min is None else min(running_min, batch_min)
                running_max = batch_max if running_max is None else max(running_max, batch_max)
                processed_points_all_ckpts += len(real_indices)

                if hasattr(score_iter, "update"):
                    score_iter.update(1)
                    score_iter.set_postfix(
                        samples=f"{end}/{len(picked)}",
                        timesteps=f"{len(train_t_seq)}",
                        mc=f"{cfg.train_mc_samples}",
                    )
                if (not cfg.use_tqdm) and (
                    batch_no == 1 or end == len(picked) or batch_no % report_every_batches == 0
                ):
                    elapsed_ckpt = time.time() - score_loop_start
                    avg_unit = elapsed_ckpt / max(1, batch_no)
                    remain_ckpt = avg_unit * (len(batch_starts) - batch_no)
                    print(
                        f"[checkpoint {ckpt_i + 1}/{len(ckpts)}] aggregate train loss | "
                        f"samples {end}/{len(picked)} | "
                        f"last_idx={real_indices[-1]} | "
                        f"ckpt_elapsed={format_seconds(elapsed_ckpt)} | "
                        f"ckpt_eta={format_seconds(remain_ckpt)}"
                    )
            if hasattr(score_iter, "close"):
                score_iter.close()
            print(
                f"[checkpoint {ckpt_i + 1}/{len(ckpts)}] aggregate train-loss score complete | "
                f"train_timesteps={len(train_t_seq)} | mc_per_timestep={cfg.train_mc_samples}",
                flush=True,
            )
            continue

        for chunk_start in range(0, len(t_seq), snapshot_chunk_size):
            chunk_end = min(chunk_start + snapshot_chunk_size, len(t_seq))
            chunk_ids = list(range(chunk_start, chunk_end))
            term_base = len(full_term_ckpt_indices)
            if full_term_scores_by_key:
                for local_snapshot_idx in chunk_ids:
                    full_term_ckpt_indices.append(int(ckpt_i))
                    full_term_timesteps.append(int(t_seq[local_snapshot_idx]))
                    full_term_snapshot_positions.append(int(pos_seq[local_snapshot_idx]))
                    full_term_weights.append(float(snap_weight))
                    full_term_ckpt_paths.append(str(ckpt_path))
            xt_chunk = array_to_device(jnp.stack([xt_refs[i] for i in chunk_ids], axis=0), device)
            t_chunk = array_to_device(
                jnp.asarray([int(t_seq[i]) for i in chunk_ids], dtype=jnp.int32),
                device,
            )
            query_grads = query_grad_chunk_fn(
                params,
                query_target_params,
                reference_params,
                xt_chunk,
                t_chunk,
                query_cond,
            )
            if chunk_start == 0:
                query_grads = tree_to_device(query_grads, device)
                print(f"[device-check] query_grad_chunk[0]={first_leaf_device_str(query_grads)}")
            query_grads = tree_to_device(query_grads, device)

            for batch_no, start in enumerate(batch_starts, start=1):
                end = min(start + batch_size, len(picked))
                real_indices = picked[start:end]
                padded_indices = pad_indices_to_batch(real_indices, batch_size)
                x_batch, cond_batch = make_train_batch(adapter, ds, padded_indices, device)
                if chunk_start == 0 and batch_no == 1:
                    print(
                        "[device-check] "
                        f"x_batch={array_device_str(x_batch)} | "
                        f"cond_batch={array_device_str(cond_batch)}"
                    )
                rng = array_to_device(
                    jnp.stack(
                        [
                            jax.random.PRNGKey(
                                cfg.seed + 100000 * ckpt_i + 1000 * batch_no + snap_id
                            )
                            for snap_id in chunk_ids
                        ],
                        axis=0,
                    ),
                    device,
                )
                chunk_scores_out = score_snapshot_chunk_fn(
                    params,
                    query_grads,
                    x_batch,
                    cond_batch,
                    rng,
                    t_chunk,
                )
                if cfg.save_query_normalized_scores:
                    raw_chunk_scores, normalized_chunk_scores = chunk_scores_out
                    raw_chunk_scores.block_until_ready()
                    normalized_chunk_scores.block_until_ready()
                    batch_scores = np.asarray(jax.device_get(raw_chunk_scores))[:, : len(real_indices)]
                    batch_query_normalized_scores = np.asarray(
                        jax.device_get(normalized_chunk_scores)
                    )[:, : len(real_indices)]
                else:
                    chunk_scores_out.block_until_ready()
                    batch_scores = np.asarray(jax.device_get(chunk_scores_out))[:, : len(real_indices)]
                    batch_query_normalized_scores = None
                batch_contrib = snap_weight * batch_scores.astype(np.float64)
                scores[start:end] += batch_contrib.sum(axis=0)
                if "scores_by_term_raw" in full_term_scores_by_key:
                    full_term_scores_by_key["scores_by_term_raw"][
                        term_base : term_base + len(chunk_ids),
                        start:end,
                    ] = batch_contrib.astype(np.float32)
                if query_normalized_scores is not None and batch_query_normalized_scores is not None:
                    query_normalized_contrib = snap_weight * batch_query_normalized_scores.astype(np.float64)
                    query_normalized_scores[start:end] += query_normalized_contrib.sum(axis=0)
                    if "scores_by_term_query_l2_normalized" in full_term_scores_by_key:
                        full_term_scores_by_key["scores_by_term_query_l2_normalized"][
                            term_base : term_base + len(chunk_ids),
                            start:end,
                        ] = query_normalized_contrib.astype(np.float32)

                if chunk_end == len(t_seq):
                    running_sum += float(scores[start:end].sum())
                    batch_min = float(scores[start:end].min()) if len(real_indices) else 0.0
                    batch_max = float(scores[start:end].max()) if len(real_indices) else 0.0
                    running_min = batch_min if running_min is None else min(running_min, batch_min)
                    running_max = batch_max if running_max is None else max(running_max, batch_max)
                    processed_points_all_ckpts += len(real_indices)

                done_units += len(chunk_ids)
                if hasattr(score_iter, "update"):
                    score_iter.update(len(chunk_ids))
                    score_iter.set_postfix(
                        snapshot=f"{chunk_start + 1}-{chunk_end}/{len(t_seq)}",
                        samples=f"{end}/{len(picked)}",
                    )

                if (not cfg.use_tqdm) and (
                    (chunk_start == 0 and batch_no == 1)
                    or (chunk_end == len(t_seq) and (end == len(picked) or batch_no % report_every_batches == 0))
                ):
                    elapsed_ckpt = time.time() - score_loop_start
                    avg_unit = elapsed_ckpt / max(1, done_units)
                    remain_units = len(t_seq) * len(batch_starts) - done_units
                    remain_ckpt = avg_unit * remain_units
                    elapsed_total = time.time() - t_start
                    print(
                        f"[checkpoint {ckpt_i + 1}/{len(ckpts)}] "
                        f"snapshot {chunk_start + 1}-{chunk_end}/{len(t_seq)} | "
                        f"samples {end}/{len(picked)} | "
                        f"last_idx={real_indices[-1]} | "
                        f"ckpt_elapsed={format_seconds(elapsed_ckpt)} | "
                        f"ckpt_eta={format_seconds(remain_ckpt)} | "
                        f"total_elapsed={format_seconds(elapsed_total)}"
                    )

            del query_grads

        if hasattr(score_iter, "close"):
            score_iter.close()

        expected_terms = len(full_term_ckpt_indices)
        if full_term_scores_by_key and expected_terms > next(iter(full_term_scores_by_key.values())).shape[0]:
            raise RuntimeError("Full-dim term score buffer was too small for the number of scored terms.")

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
        "query_objective": {
            "name": cfg.query_objective,
            "formula": query_objective_formula(cfg.query_objective),
            "reference_ckpt": reference_ckpt,
            "target_checkpoint": (
                "next_checkpoint" if uses_next_checkpoint_target else "reference_checkpoint"
            ),
            "snapshot_reduction": "mean",
            "checkpoint_reduction": "sum",
        },
        "used_checkpoints": used_ckpts,
        "skipped_checkpoints": skipped_ckpts,
        "precomputed_sample_meta": precomputed_sample_meta,
        "manifest_checkpoint": manifest_ckpt,
        "resolved_manifest_checkpoint": resolved_manifest_ckpt,
        "topk": top,
        "elapsed_sec": time.time() - t_start,
    }

    def write_score_artifact(out_dir: str, score_values: np.ndarray, variant: str, normalization: Dict[str, Any]):
        import json

        ensure_dir(out_dir)
        order_variant = np.argsort(-score_values)[:topk]
        top_variant = [
            {
                "idx": int(picked[i]),
                "idx_1based": int(picked[i]) + 1,
                "score": float(score_values[i]),
            }
            for i in order_variant
        ]
        out_variant = dict(out)
        out_variant["topk"] = top_variant
        out_variant["score_variant"] = variant
        out_variant["gradient_normalization"] = normalization

        print(f"[save:{variant}] writing traj_attr_result.json ...")
        with open(os.path.join(out_dir, "traj_attr_result.json"), "w") as f:
            json.dump(out_variant, f, indent=2)
        print(f"[save:{variant}] traj_attr_result.json written")

        print(f"[save:{variant}] writing scores.npy ...")
        np.save(os.path.join(out_dir, "scores.npy"), score_values)
        print(f"[save:{variant}] scores.npy written")

        print(f"[save:{variant}] writing score_indices.npy ...")
        np.save(os.path.join(out_dir, "score_indices.npy"), np.asarray(picked, dtype=np.int64))
        print(f"[save:{variant}] score_indices.npy written")

    write_score_artifact(
        cfg.out_dir,
        scores,
        "raw",
        {
            "query_gradient": "none",
            "train_gradient": "none",
        },
    )

    normalized_dir = None
    if query_normalized_scores is not None:
        normalized_dir = query_normalized_out_dir(cfg.out_dir)
        write_score_artifact(
            normalized_dir,
            query_normalized_scores,
            "query_gradient_l2_normalized",
            {
                "query_gradient": "l2",
                "query_normalize_eps": float(cfg.query_normalize_eps),
                "train_gradient": "none",
                "note": (
                    "This paired score reuses the raw TrajTracIn pass and normalizes "
                    "the query-gradient tangent before the train-loss JVP. It does not "
                    "materialize or L2-normalize each train datapoint gradient."
                ),
            },
        )

    if full_term_scores_by_key:
        term_count = len(full_term_ckpt_indices)
        term_artifact_path = os.environ.get("TRAJ_TRACIN_FULL_TERM_SCORE_ARTIFACT_PATH")
        if not term_artifact_path:
            term_artifact_path = os.path.join(cfg.out_dir, "full_dim_term_scores.npz")
        term_payload = dict(
            score_indices=np.asarray(picked, dtype=np.int64),
            term_ckpt_indices=np.asarray(full_term_ckpt_indices, dtype=np.int32),
            term_timesteps=np.asarray(full_term_timesteps, dtype=np.int32),
            term_snapshot_positions=np.asarray(full_term_snapshot_positions, dtype=np.int32),
            term_weights=np.asarray(full_term_weights, dtype=np.float32),
            term_ckpt_paths=np.asarray(full_term_ckpt_paths),
            term_score_variants=np.asarray(
                [key.replace("scores_by_term_", "") for key in full_term_score_keys]
            ),
            query_objective=np.asarray(cfg.query_objective),
            query_target_checkpoint=np.asarray(
                "next_checkpoint" if uses_next_checkpoint_target else "reference_checkpoint"
            ),
        )
        for key, values in full_term_scores_by_key.items():
            term_payload[key] = values[:term_count].astype(np.float32)
        save_npz_compressed_atomic(term_artifact_path, **term_payload)
        print(f"[saved] full-dim per-term TrajTracIn score artifact: {term_artifact_path}")

    print("=" * 90)
    print(f"Saved to {cfg.out_dir}")
    if normalized_dir is not None:
        print(f"Saved query-normalized scores to {normalized_dir}")
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
            train_mc_samples=10,
            m_proj=2,   # number of random r projections for query scalarization

            # how many training points to score
            max_train_points=2000,
            random_subset=True,
            score_batch_size=16,

            # how many highest-scoring points to save
            topk=2000,

            progress_every=50,
            out_dir="./attribution_results/traj_tracein/traj_attr_x3",
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

            train_mc_samples=10,
            m_proj=2,

            # how many training points to score
            max_train_points=100,
            random_subset=True,
            score_batch_size=16,

            # how many top results to save
            topk=10000,

            progress_every=10,
            out_dir="./attribution_results/traj_tracein/traj_attr_cifar10_single",
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

            train_mc_samples=10,
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
            out_dir="./attribution_results/traj_tracein/traj_attr_cifar10_multi",
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

            train_mc_samples=10,
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
            out_dir="./attribution_results/traj_tracein/traj_attr_cifar10_from_sample",
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

            train_mc_samples=10,
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
            out_dir="./attribution_results/traj_tracein/traj_attr_artbench_latent_from_sample",
        )

    else:
        raise ValueError(
            "EXAMPLE must be one of: 'x3', 'cifar10_single', 'cifar10_multi', "
            "'cifar10_sample', 'artbench_latent_sample'"
        )

    run_attribution(cfg)
