# trajectory_tracin_x3_updated.py
"""
Updated PyTorch X3 Traj-TracIn.

Adds:
1) query target switch:
      QUERY_TARGET = "reference"
      QUERY_TARGET = "next_checkpoint"

2) checkpoint parameter switch:
      PARAM_SOURCE = "raw"
      PARAM_SOURCE = "ema"

The core old X3 Traj-TracIn semantics are preserved:
- fixed reference diffusion trajectory
- simple diffusion training loss at the same trajectory timestep
- Monte Carlo expectation over training noise
- snapshot mean
- checkpoint sum
- no projection
- no gradient normalization
- checkpoint weight = 1.0

Definitions
-----------
Reference target:
    f_{c,k}(theta_c)
      = || eps_{theta_c}(x_ref_k, t_k, q)
           - stopgrad(eps_{theta_ref}(x_ref_k, t_k, q)) ||_2^2

Next-checkpoint target:
    f_{c,k}(theta_c)
      = || eps_{theta_c}(x_ref_k, t_k, q)
           - stopgrad(eps_{theta_{c+1}}(x_ref_k, t_k, q)) ||_2^2

Score:
    score(z)
      = sum_c 1/K sum_k
          < grad_theta f_{c,k},
            grad_theta E_noise[L_simple(z, t_k; theta_c)] >

Important:
- The reference diffusion trajectory x_ref_k is fixed and is generated once
  using REFERENCE_CKPT and PARAM_SOURCE.
- In next-checkpoint mode, only the *query target prediction* changes to c+1.
  The x_ref trajectory remains fixed.
- The final checkpoint is skipped in next-checkpoint mode.
"""

import os
import glob
import json
import random
import time
import re
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import x3pixel_DM_training as base
from dataset_loader import ColorGridDataset


# ============================================================
# Repro
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(False)


# ============================================================
# Progress helpers
# ============================================================

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


def print_item_progress(
    phase: str,
    idx: int,
    total: int,
    extra: str = "",
):
    msg = f"    [{phase}] {idx + 1}/{total}"

    if extra:
        msg += f" | {extra}"

    print(msg, flush=True)


def should_print_item(
    idx: int,
    total: int,
    every: int = 100,
) -> bool:
    return (
        idx == 0
        or (idx + 1) % every == 0
        or (idx + 1) == total
    )


# ============================================================
# LoRA wrapper
# ============================================================

class LoRAConv2d(nn.Module):
    def __init__(
        self,
        base_conv: nn.Conv2d,
        r: int = 4,
        alpha: float = 1.0,
    ):
        super().__init__()

        if not isinstance(base_conv, nn.Conv2d):
            raise TypeError(
                "LoRAConv2d expects nn.Conv2d"
            )

        if r <= 0:
            raise ValueError(
                "LoRA rank r must be > 0"
            )

        self.base = base_conv
        self.r = int(r)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.r

        in_ch = base_conv.in_channels
        out_ch = base_conv.out_channels
        k = base_conv.kernel_size
        s = base_conv.stride
        p = base_conv.padding
        d = base_conv.dilation
        g = base_conv.groups

        self.lora_down = nn.Conv2d(
            in_ch,
            self.r,
            kernel_size=1,
            bias=False,
        )

        self.lora_up = nn.Conv2d(
            self.r,
            out_ch,
            kernel_size=k,
            stride=s,
            padding=p,
            dilation=d,
            groups=g,
            bias=False,
        )

        nn.init.kaiming_uniform_(
            self.lora_down.weight,
            a=5 ** 0.5,
        )

        nn.init.zeros_(
            self.lora_up.weight
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.base(x)
            + self.scale
            * self.lora_up(
                self.lora_down(x)
            )
        )


def inject_lora_into_selected_convs(
    model: nn.Module,
    r: int,
    alpha: float,
    target_names,
):
    target_names = (
        set(target_names)
        if target_names is not None
        else {"out_conv"}
    )

    for full_name, module in list(
        model.named_modules()
    ):
        if (
            full_name in target_names
            and isinstance(module, nn.Conv2d)
        ):
            parent = model
            parts = full_name.split(".")

            for p in parts[:-1]:
                parent = getattr(parent, p)

            setattr(
                parent,
                parts[-1],
                LoRAConv2d(
                    module,
                    r=r,
                    alpha=alpha,
                ),
            )


# ============================================================
# Checkpoint discovery
# ============================================================

def _natural_key(path: str):
    """
    Natural sort:
      baseline_2.pt < baseline_10.pt
    """
    name = os.path.basename(path)
    parts = re.split(r"(\d+)", name)

    return [
        int(p) if p.isdigit() else p
        for p in parts
    ]


def list_checkpoints_sorted(
    dir_path: str,
    pattern: str = "*.pt",
) -> List[str]:
    if not os.path.isdir(dir_path):
        return []

    paths = glob.glob(
        os.path.join(
            dir_path,
            pattern,
        )
    )

    paths.sort(key=_natural_key)
    return paths


def latest_checkpoint_in_dir(
    dir_path: str,
    pattern: str = "*.pt",
) -> Optional[str]:
    paths = list_checkpoints_sorted(
        dir_path,
        pattern,
    )

    return paths[-1] if paths else None


# ============================================================
# Vocab / query conditioning
# ============================================================

def vocab_to_index(
    vocab: Any,
) -> Dict[str, int]:
    if vocab is None:
        raise ValueError(
            "Checkpoint/dataset has no vocab; "
            "cannot build cond from label strings."
        )

    if isinstance(vocab, dict):
        return {
            str(k): int(v)
            for k, v in vocab.items()
        }

    if isinstance(vocab, (list, tuple)):
        return {
            str(lbl): i
            for i, lbl in enumerate(vocab)
        }

    raise TypeError(
        f"Unsupported vocab type: {type(vocab)}"
    )


def labels_to_cond(
    labels: List[str],
    vocab: Any,
    cond_dim: int,
    device: torch.device,
) -> torch.Tensor:
    m = vocab_to_index(vocab)

    cond = torch.zeros(
        cond_dim,
        device=device,
        dtype=torch.float32,
    )

    missing = [
        lab
        for lab in labels
        if lab not in m
    ]

    if missing:
        raise KeyError(
            "Query labels not found in vocab: "
            f"{missing[:10]}"
        )

    for lab in labels:
        cond[m[lab]] = 1.0

    return cond.unsqueeze(0)


# ============================================================
# Raw / EMA state selection
# ============================================================

def normalize_param_source(
    param_source: str,
) -> str:
    x = str(param_source).strip().lower()

    aliases = {
        "raw": "raw",
        "params": "raw",
        "model": "raw",
        "model_state": "raw",

        "ema": "ema",
        "ema_model": "ema",
        "ema_model_state": "ema",
    }

    if x not in aliases:
        raise ValueError(
            "PARAM_SOURCE must be 'raw' or 'ema'. "
            f"Got: {param_source!r}"
        )

    return aliases[x]


def state_dict_from_baseline_payload(
    ckpt: Dict[str, Any],
    *,
    param_source: str,
    ckpt_path: str,
) -> Dict[str, torch.Tensor]:
    """
    Select RAW or EMA model state from a baseline checkpoint.

    New X3 trainer:
      raw -> ckpt["model_state"]
      ema -> ckpt["ema_model_state"]

    For old checkpoints without EMA, requesting EMA raises a clear error.
    """
    param_source = normalize_param_source(
        param_source
    )

    if param_source == "raw":
        if "model_state" not in ckpt:
            raise KeyError(
                f"RAW requested, but checkpoint has no "
                f"'model_state': {ckpt_path}"
            )

        return ckpt["model_state"]

    if "ema_model_state" not in ckpt:
        raise KeyError(
            "EMA requested, but checkpoint has no "
            f"'ema_model_state': {ckpt_path}\n"
            "This is probably an older X3 checkpoint "
            "saved before EMA was added."
        )

    return ckpt["ema_model_state"]


# ============================================================
# Build baseline model from checkpoint
# ============================================================

def build_model_from_baseline_ckpt(
    baseline_ckpt_path: str,
    device: torch.device,
    *,
    param_source: str = "raw",
) -> Tuple[nn.Module, Dict[str, Any]]:
    ckpt = torch.load(
        baseline_ckpt_path,
        map_location=str(device),
    )

    need = [
        "T",
        "cond_dim",
    ]

    for k in need:
        if k not in ckpt:
            raise KeyError(
                f"Baseline ckpt missing {k}: "
                f"{baseline_ckpt_path}"
            )

    cond_dim = int(
        ckpt["cond_dim"]
    )

    base_ch = int(
        ckpt.get(
            "base_ch",
            64,
        )
    )

    time_dim = int(
        ckpt.get(
            "time_dim",
            128,
        )
    )

    grid_size = int(
        ckpt.get(
            "grid_size",
            3,
        )
    )

    model = base.CondEpsModel(
        in_ch=3,
        cond_dim=cond_dim,
        base_ch=base_ch,
        time_dim=time_dim,
    ).to(device)

    selected_state = state_dict_from_baseline_payload(
        ckpt,
        param_source=param_source,
        ckpt_path=baseline_ckpt_path,
    )

    model.load_state_dict(
        selected_state,
        strict=True,
    )

    model.eval()

    meta = {
        "T": int(ckpt["T"]),
        "grid_size": grid_size,
        "cond_dim": cond_dim,
        "vocab": ckpt.get(
            "vocab",
            None,
        ),
        "base_ch": base_ch,
        "time_dim": time_dim,
        "param_source": normalize_param_source(
            param_source
        ),
        "epoch": ckpt.get(
            "epoch",
            None,
        ),
        "global_step": ckpt.get(
            "global_step",
            None,
        ),
    }

    return model, meta


# ============================================================
# Build LoRA model
# ============================================================

def build_model_from_lora_ckpt(
    lora_ckpt_path: str,
    device: torch.device,
    *,
    baseline_param_source: str = "raw",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    LoRA checkpoints themselves only contain LoRA state in the old workflow.

    baseline_param_source controls whether the underlying baseline is RAW
    or EMA, *if* that baseline checkpoint contains EMA state.

    Note:
    EMA of LoRA parameters is not supported unless your LoRA trainer
    explicitly saves such an EMA state.
    """
    payload = torch.load(
        lora_ckpt_path,
        map_location="cpu",
    )

    lora_sd = payload.get(
        "lora_state",
        payload,
    )

    baseline_ckpt_path = payload.get(
        "baseline_ckpt",
        None,
    )

    if baseline_ckpt_path is None:
        raise KeyError(
            "LoRA ckpt missing baseline_ckpt: "
            f"{lora_ckpt_path}"
        )

    base_model, meta = (
        build_model_from_baseline_ckpt(
            baseline_ckpt_path,
            device=device,
            param_source=baseline_param_source,
        )
    )

    r = int(
        payload.get(
            "lora_r",
            4,
        )
    )

    alpha = float(
        payload.get(
            "lora_alpha",
            4.0,
        )
    )

    targets = payload.get(
        "lora_targets",
        ["out_conv"],
    )

    inject_lora_into_selected_convs(
        base_model,
        r=r,
        alpha=alpha,
        target_names=targets,
    )

    base_model.load_state_dict(
        lora_sd,
        strict=False,
    )

    base_model.eval()

    meta = dict(meta)
    meta.update(
        {
            "lora_r": r,
            "lora_alpha": alpha,
            "lora_targets": list(targets),
        }
    )

    return base_model, meta


# ============================================================
# Active-parameter switching
# ============================================================

def set_active_params_baseline(
    model: nn.Module,
) -> List[nn.Parameter]:
    """
    Baseline:
    differentiate all non-LoRA parameters.
    """
    active = []

    for name, p in model.named_parameters():
        if ".lora_" in name:
            p.requires_grad_(False)

        else:
            p.requires_grad_(True)
            active.append(p)

    return active


def set_active_params_lora(
    model: nn.Module,
) -> List[nn.Parameter]:
    """
    LoRA:
    differentiate only LoRA parameters.
    """
    active = []

    for name, p in model.named_parameters():
        if ".lora_" in name:
            p.requires_grad_(True)
            active.append(p)

        else:
            p.requires_grad_(False)

    return active


# ============================================================
# Query target switch
# ============================================================

def normalize_query_target(
    query_target: str,
) -> str:
    x = str(query_target).strip().lower()

    aliases = {
        "ref": "reference",
        "reference": "reference",
        "reference_checkpoint": "reference",

        "next": "next_checkpoint",
        "next_checkpoint": "next_checkpoint",
        "next_ckpt": "next_checkpoint",
        "c+1": "next_checkpoint",
    }

    if x not in aliases:
        raise ValueError(
            "QUERY_TARGET must be 'reference' "
            "or 'next_checkpoint'. "
            f"Got: {query_target!r}"
        )

    return aliases[x]


# ============================================================
# Query objective
# ============================================================

def trajectory_noise_deviation_to_target(
    model: nn.Module,
    xt_ref: torch.Tensor,
    t: torch.Tensor,
    cond_q: torch.Tensor,
    eps_target: torch.Tensor,
) -> torch.Tensor:
    """
    Per-snapshot objective:

      f(theta)
        = || eps_theta(xt_ref, t, cond_q)
             - stopgrad(eps_target) ||_2^2

    eps_target may come from:
      - final/reference checkpoint
      - next checkpoint
    """
    eps_pred = model(
        xt_ref,
        t,
        cond_q,
    )

    return (
        eps_pred
        - eps_target.detach()
    ).pow(2).sum()


def grad_dot(
    g1,
    g2,
) -> torch.Tensor:
    s = None

    for a, b in zip(g1, g2):
        if a is None or b is None:
            continue

        v = (a * b).sum()

        s = (
            v
            if s is None
            else s + v
        )

    if s is None:
        device = (
            g1[0].device
            if (
                len(g1) > 0
                and g1[0] is not None
            )
            else "cpu"
        )

        return torch.tensor(
            0.0,
            device=device,
        )

    return s


# ============================================================
# Train loss at fixed trajectory timestep
# ============================================================

def train_loss_mc_at_t(
    model: nn.Module,
    sched,
    x0: torch.Tensor,
    cond: torch.Tensor,
    t: torch.Tensor,
    *,
    num_mc_samples: int = 8,
) -> torch.Tensor:
    """
    Monte Carlo estimate of:

      E_noise [
        MSE(
          eps_theta(x_t, t, cond),
          noise
        )
      ]

    at a fixed diffusion timestep t.
    """
    losses = []

    for _ in range(
        int(num_mc_samples)
    ):
        noise = torch.randn_like(
            x0
        )

        xt = base.q_sample(
            x0,
            t,
            noise,
            sched,
        )

        eps_pred = model(
            xt,
            t,
            cond,
        )

        losses.append(
            F.mse_loss(
                eps_pred,
                noise,
                reduction="mean",
            )
        )

    return torch.stack(
        losses
    ).mean()


# ============================================================
# Fixed reference diffusion trajectory
# ============================================================

@torch.no_grad()
def compute_reference_trajectory(
    model: nn.Module,
    sched,
    cond: torch.Tensor,
    shape: Tuple[int, int, int, int],
    seed: int,
    steps: int,
    eta: float,
    device: torch.device,
    num_keep: int,
) -> Tuple[
    List[torch.Tensor],
    np.ndarray,
    List[int],
]:
    """
    Generate ONE fixed reference DDIM trajectory.

    Returns
    -------
    traj_use:
        saved x_t snapshots

    t_seq:
        diffusion timestep for each saved snapshot

    save_steps:
        DDIM-loop position for each snapshot
    """
    model.eval()

    steps = int(steps)
    num_keep = int(num_keep)

    if num_keep <= 0:
        raise ValueError(
            "num_keep must be > 0"
        )

    num_keep = min(
        num_keep,
        steps,
    )

    save_steps = np.linspace(
        0,
        steps - 1,
        num_keep,
        dtype=np.int64,
    ).tolist()

    traj_use = base.ddim_sample(
        model=model,
        sched=sched,
        cond=cond,
        shape=shape,
        seed=seed,
        steps=steps,
        eta=eta,
        device=str(device),
        save_steps=save_steps,
    )

    T = int(
        sched.T
    )

    ts = np.linspace(
        T - 1,
        0,
        steps,
        dtype=np.int64,
    )

    t_seq = np.array(
        [
            int(ts[k])
            for k in save_steps
        ],
        dtype=np.int64,
    )

    if len(traj_use) != len(t_seq):
        raise RuntimeError(
            "traj_use len "
            f"{len(traj_use)} != "
            f"t_seq len {len(t_seq)}"
        )

    return (
        traj_use,
        t_seq,
        save_steps,
    )


def pick_snapshot_ids(
    K: int,
    num_pick: int,
) -> List[int]:
    K = int(K)
    num_pick = int(num_pick)

    if num_pick >= K:
        return list(range(K))

    return np.linspace(
        0,
        K - 1,
        num_pick,
        dtype=np.int64,
    ).tolist()


def build_xtref_dict_by_snapid(
    traj_use: List[torch.Tensor],
    snap_ids: List[int],
) -> Dict[int, torch.Tensor]:
    out: Dict[
        int,
        torch.Tensor,
    ] = {}

    for i in snap_ids:
        out[int(i)] = (
            traj_use[int(i)]
            .detach()
        )

    return out


# ============================================================
# Target epsilon cache
# ============================================================

@torch.no_grad()
def compute_eps_target_by_snapid(
    target_model: nn.Module,
    xt_ref_dict: Dict[int, torch.Tensor],
    query_cond: torch.Tensor,
    snap_ids: List[int],
    t_seq: np.ndarray,
) -> Dict[int, torch.Tensor]:
    """
    Cache eps_target(x_ref_k, t_k, query_cond).

    The target model can be:
      - reference model
      - next-checkpoint model
    """
    target_model.eval()

    out: Dict[
        int,
        torch.Tensor,
    ] = {}

    for sid in snap_ids:
        sid = int(sid)

        xt_ref = xt_ref_dict[sid]

        t = torch.tensor(
            [int(t_seq[sid])],
            device=xt_ref.device,
            dtype=torch.long,
        )

        out[sid] = (
            target_model(
                xt_ref,
                t,
                query_cond,
            )
            .detach()
        )

    return out


# ============================================================
# Query gradients
# ============================================================

def compute_g_traj(
    model: nn.Module,
    active_params: List[nn.Parameter],
    xt_ref_dict: Dict[int, torch.Tensor],
    eps_target_dict: Dict[int, torch.Tensor],
    query_cond: torch.Tensor,
    snap_ids: List[int],
    t_seq: np.ndarray,
) -> Dict[
    int,
    Tuple[torch.Tensor, ...],
]:
    """
    For each trajectory snapshot:

      g_q[k]
        = grad_theta
            ||eps_theta(x_ref_k,t_k,q)
              - stopgrad(eps_target_k)||^2
    """
    model.eval()

    gq: Dict[
        int,
        Tuple[torch.Tensor, ...],
    ] = {}

    total = len(snap_ids)

    for k, sid in enumerate(
        snap_ids
    ):
        sid = int(sid)

        if should_print_item(
            k,
            total,
            every=10,
        ):
            print_item_progress(
                "query grad",
                k,
                total,
                extra=(
                    f"snap_id={sid} "
                    f"t={int(t_seq[sid])}"
                ),
            )

        xt_ref = (
            xt_ref_dict[sid]
            .detach()
        )

        t_int = int(
            t_seq[sid]
        )

        t = torch.tensor(
            [t_int],
            device=xt_ref.device,
            dtype=torch.long,
        )

        f = (
            trajectory_noise_deviation_to_target(
                model=model,
                xt_ref=xt_ref,
                t=t,
                cond_q=query_cond,
                eps_target=eps_target_dict[sid],
            )
        )

        g = torch.autograd.grad(
            f,
            active_params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        gq[sid] = g

    return gq


# ============================================================
# Score one training point
# ============================================================

def score_one_trainpoint_given_gtraj(
    model: nn.Module,
    active_params: List[nn.Parameter],
    sched,
    gq: Dict[
        int,
        Tuple[torch.Tensor, ...],
    ],
    snap_ids: List[int],
    t_seq: np.ndarray,
    x0_train: torch.Tensor,
    train_cond: torch.Tensor,
    *,
    checkpoint_weight: float = 1.0,
    train_mc_samples: int = 8,
) -> torch.Tensor:
    """
    Old-X3 style Traj-TracIn:

      score_c(z)
        = checkpoint_weight
          * (1/K)
          * sum_k
              <g_q[c,k], g_train[z,c,k]>

    By default checkpoint_weight = 1.
    """
    model.train()

    snapshot_weight = (
        1.0
        / float(len(snap_ids))
    )

    total = 0.0

    for sid in snap_ids:
        sid = int(sid)

        t_int = int(
            t_seq[sid]
        )

        t = torch.tensor(
            [t_int],
            device=x0_train.device,
            dtype=torch.long,
        )

        L_tr = train_loss_mc_at_t(
            model,
            sched,
            x0_train,
            train_cond,
            t,
            num_mc_samples=int(
                train_mc_samples
            ),
        )

        g_tr = torch.autograd.grad(
            L_tr,
            active_params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        total = (
            total
            + checkpoint_weight
            * snapshot_weight
            * grad_dot(
                gq[sid],
                g_tr,
            )
        )

    return total.detach()


# ============================================================
# IO
# ============================================================

def load_index_list(
    path: str,
    col: str = "idx",
) -> List[int]:
    """
    Supports:
      - CSV with header
      - plain text one integer per line
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    idxs = []

    with open(path, "r") as f:
        lines = [
            ln.strip()
            for ln in f.readlines()
            if ln.strip()
        ]

    if (
        len(lines) >= 2
        and lines[0].lower()
        == col.lower()
    ):
        for ln in lines[1:]:
            idxs.append(
                int(
                    ln.split(",")[0]
                )
            )

        return idxs

    for ln in lines:
        idxs.append(
            int(
                ln.split(",")[0]
            )
        )

    return idxs


def ensure_dir(
    path: str,
):
    os.makedirs(
        path,
        exist_ok=True,
    )


def save_json(
    path: str,
    obj,
):
    ensure_dir(
        os.path.dirname(path)
    )

    with open(path, "w") as f:
        json.dump(
            obj,
            f,
            indent=2,
        )


# ============================================================
# Candidate set
# ============================================================

def build_score_items(
    ds,
    CONFIG: Dict[str, Any],
) -> List[Tuple[int, int]]:
    """
    Returns:
      [(src_id, dataset_idx), ...]

    Preserves your original 0/2/3/6-list behavior.
    """
    N = len(ds)

    M_req = min(
        int(
            CONFIG[
                "max_train_points"
            ]
        ),
        N,
    )

    use_six_lists = bool(
        CONFIG.get(
            "use_six_index_lists",
            False,
        )
    )

    use_three_lists = (
        bool(
            CONFIG.get(
                "use_three_index_lists",
                False,
            )
        )
        and not use_six_lists
    )

    use_two_lists = (
        bool(
            CONFIG.get(
                "use_two_index_lists",
                False,
            )
        )
        and not use_three_lists
        and not use_six_lists
    )

    # --------------------------------------------------------
    # No explicit index lists
    # --------------------------------------------------------

    if (
        not use_two_lists
        and not use_three_lists
        and not use_six_lists
    ):
        if CONFIG.get(
            "random_subset",
            False,
        ):
            return [
                (0, i)
                for i in random.sample(
                    range(N),
                    k=M_req,
                )
            ]

        return [
            (0, i)
            for i in range(M_req)
        ]

    idx_col = CONFIG.get(
        "idx_col_name",
        "idx",
    )

    # --------------------------------------------------------
    # Two lists
    # --------------------------------------------------------

    if use_two_lists:
        lists = []

        for k in range(1, 3):
            lst = load_index_list(
                CONFIG[f"idx_list_{k}"],
                col=idx_col,
            )

            lst = sorted(
                set(
                    int(i)
                    for i in lst
                    if 0 <= int(i) < N
                )
            )

            lists.append(lst)

        M1 = min(
            M_req // 2,
            len(lists[0]),
        )

        M2 = min(
            M_req - M1,
            len(lists[1]),
        )

        Ms = [
            M1,
            M2,
        ]

    # --------------------------------------------------------
    # Three lists
    # --------------------------------------------------------

    elif use_three_lists:
        lists = []

        for k in range(1, 4):
            lst = load_index_list(
                CONFIG[f"idx_list_{k}"],
                col=idx_col,
            )

            lst = sorted(
                set(
                    int(i)
                    for i in lst
                    if 0 <= int(i) < N
                )
            )

            lists.append(lst)

        split_base = (
            M_req // 3
        )

        Ms = [
            min(
                split_base,
                len(lists[k]),
            )
            for k in range(3)
        ]

        remainder = (
            M_req
            - sum(Ms)
        )

        while remainder > 0:
            progressed = False

            for k in range(3):
                cap = (
                    len(lists[k])
                    - Ms[k]
                )

                if (
                    cap > 0
                    and remainder > 0
                ):
                    Ms[k] += 1
                    remainder -= 1
                    progressed = True

            if not progressed:
                break

    # --------------------------------------------------------
    # Six lists
    # --------------------------------------------------------

    else:
        lists = []

        for k in range(1, 7):
            key = f"idx_list_{k}"

            if CONFIG.get(
                key,
                None,
            ) is None:
                raise KeyError(
                    f"Missing CONFIG[{key}] "
                    "for six-index mode."
                )

            lst = load_index_list(
                CONFIG[key],
                col=idx_col,
            )

            lst = sorted(
                set(
                    int(i)
                    for i in lst
                    if 0 <= int(i) < N
                )
            )

            lists.append(lst)

        split_base = (
            M_req // 6
        )

        Ms = [
            min(
                split_base,
                len(lists[k]),
            )
            for k in range(6)
        ]

        remainder = (
            M_req
            - sum(Ms)
        )

        while remainder > 0:
            progressed = False

            for k in range(6):
                cap = (
                    len(lists[k])
                    - Ms[k]
                )

                if (
                    cap > 0
                    and remainder > 0
                ):
                    Ms[k] += 1
                    remainder -= 1
                    progressed = True

            if not progressed:
                break

    picks = []

    for k, lst in enumerate(
        lists
    ):
        Mk = Ms[k]

        if Mk <= 0:
            picks.append([])
            continue

        if CONFIG.get(
            "random_subset",
            False,
        ):
            picks.append(
                random.sample(
                    lst,
                    k=Mk,
                )
            )

        else:
            picks.append(
                lst[:Mk]
            )

    score_items = []

    for k, one_pick in enumerate(
        picks
    ):
        src_id = k + 1

        score_items.extend(
            [
                (
                    src_id,
                    int(i),
                )
                for i in one_pick
            ]
        )

    random.shuffle(
        score_items
    )

    return score_items


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # EXPERIMENT IDENTITY
    # ========================================================

    MODEL = "model_109900"
    LORA = "y"

    MAX_TRAIN_POINTS = 2000

    # ========================================================
    # NEW SWITCH 1:
    # raw or ema
    # ========================================================

    PARAM_SOURCE = "raw"
    # PARAM_SOURCE = "ema"

    # Meaning:
    #
    # raw:
    #   use ckpt["model_state"]
    #
    # ema:
    #   use ckpt["ema_model_state"]
    #
    # This switch is applied consistently to:
    #   - current checkpoint
    #   - reference checkpoint
    #   - next checkpoint target
    #   - reference trajectory generator


    # ========================================================
    # NEW SWITCH 2:
    # reference target or next-checkpoint target
    # ========================================================

    QUERY_TARGET = "reference"
    # QUERY_TARGET = "next_checkpoint"

    # reference:
    #
    #   f_c,k =
    #     ||eps_theta_c(x_ref_k)
    #       - eps_theta_ref(x_ref_k)||^2
    #
    #
    # next_checkpoint:
    #
    #   f_c,k =
    #     ||eps_theta_c(x_ref_k)
    #       - eps_theta_{c+1}(x_ref_k)||^2
    #
    #   final checkpoint is automatically skipped.


    # ========================================================
    # Which checkpoint families to use
    # ========================================================

    USE_BASELINE = True
    USE_LORA = False

    # NOTE:
    # next_checkpoint targeting is naturally defined over an
    # ordered checkpoint sequence.
    #
    # This script supports it for baseline checkpoints.
    #
    # LoRA remains supported for reference-target scoring.
    # If you want next-checkpoint target over LoRA checkpoints,
    # this script also handles that sequence, but PARAM_SOURCE
    # only controls the underlying baseline RAW/EMA state because
    # old LoRA checkpoints do not store LoRA EMA parameters.


    PARAM_SOURCE = normalize_param_source(
        PARAM_SOURCE
    )

    QUERY_TARGET = normalize_query_target(
        QUERY_TARGET
    )

    if (
        not USE_BASELINE
        and not USE_LORA
    ):
        raise ValueError(
            "At least one of USE_BASELINE "
            "or USE_LORA must be True."
        )

    param_tag = PARAM_SOURCE

    target_tag = (
        "ref"
        if QUERY_TARGET == "reference"
        else "next"
    )

    run_tag_parts = []

    if USE_BASELINE:
        run_tag_parts.append(
            "baseline"
        )

    if USE_LORA:
        run_tag_parts.append(
            "lora"
        )

    run_tag = "_".join(
        run_tag_parts
    )

    CUR_MODEL = (
        f"{MODEL}_{LORA}"
    )

    # ========================================================
    # Config
    # ========================================================

    CONFIG = dict(
        # ----------------------------------------------------
        # runtime / data
        # ----------------------------------------------------

        device=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),

        seed=808,

        csv_path=(
            "generated_database/49_100000.csv"
        ),

        grid_size=3,

        # ----------------------------------------------------
        # model paths
        # ----------------------------------------------------

        model_root=(
            f"models_checkpoints/"
            f"{LORA}/{MODEL}"
        ),

        baseline_dir=(
            f"models_checkpoints/"
            f"{LORA}/{MODEL}/baseline"
        ),

        lora_update_dir=(
            f"models_checkpoints/"
            f"{LORA}/{MODEL}/{LORA}"
        ),

        # ----------------------------------------------------
        # new switches
        # ----------------------------------------------------

        param_source=PARAM_SOURCE,

        query_target=QUERY_TARGET,

        # ----------------------------------------------------
        # query
        # ----------------------------------------------------

        query_labels=[
            "background_color_red",
            "background_color_blue",
            "background_color_yellow",
        ],

        # ----------------------------------------------------
        # diffusion trajectory
        # ----------------------------------------------------

        ddim_steps=1000,

        eta=0.0,

        num_traj_t=50,

        # ----------------------------------------------------
        # train-gradient Monte Carlo
        # ----------------------------------------------------

        train_mc_samples=8,

        # ----------------------------------------------------
        # scoring
        # ----------------------------------------------------

        max_train_points=(
            MAX_TRAIN_POINTS
        ),

        topk=2000,

        # preserve OLD X3 semantics:
        # every checkpoint equally weighted
        checkpoint_weight=1.0,

        # ----------------------------------------------------
        # checkpoint families
        # ----------------------------------------------------

        use_baseline_ckpts=(
            USE_BASELINE
        ),

        use_lora_ckpts=(
            USE_LORA
        ),

        # None => latest baseline ckpt.
        #
        # This checkpoint is used to:
        # 1) generate the FIXED x_ref trajectory
        # 2) provide eps_ref when QUERY_TARGET="reference"
        reference_ckpt=None,

        # ----------------------------------------------------
        # candidate subset
        # ----------------------------------------------------

        random_subset=True,

        use_two_index_lists=False,
        use_three_index_lists=False,
        use_six_index_lists=True,

        idx_list_1=(
            "generated_database/RBY/"
            "subset/blue_A_idx.csv"
        ),

        idx_list_2=(
            "generated_database/RBY/"
            "subset/blue_B_idx.csv"
        ),

        idx_list_3=(
            "generated_database/RBY/"
            "subset/red_A_idx.csv"
        ),

        idx_list_4=(
            "generated_database/RBY/"
            "subset/red_B_idx.csv"
        ),

        idx_list_5=(
            "generated_database/RBY/"
            "subset/yellow_A_idx.csv"
        ),

        idx_list_6=(
            "generated_database/RBY/"
            "subset/yellow_B_idx.csv"
        ),

        idx_col_name="idx",

        # ----------------------------------------------------
        # output
        # ----------------------------------------------------

        out_dir=(
            f"tracein_traj_runs/"
            f"{CUR_MODEL}_traj_objective"
            f"{MAX_TRAIN_POINTS}/"
            f"{run_tag}_"
            f"{param_tag}_"
            f"{target_tag}"
        ),
    )

    # ========================================================
    # Setup
    # ========================================================

    t0 = time.perf_counter()

    device = torch.device(
        CONFIG["device"]
    )

    set_seed(
        int(
            CONFIG["seed"]
        )
    )

    print("=" * 90)
    print("X3 Traj-TracIn")
    print(f"PARAM_SOURCE : {CONFIG['param_source']}")
    print(f"QUERY_TARGET : {CONFIG['query_target']}")
    print(
        "trajectory   : fixed trajectory from "
        "REFERENCE_CKPT"
    )
    print(
        "ckpt weight  : "
        f"{CONFIG['checkpoint_weight']} "
        "(old-X3 semantics)"
    )
    print("=" * 90)

    print(
        f"[device] {device}",
        flush=True,
    )

    if device.type == "cuda":
        print(
            "[device] GPU count visible: "
            f"{torch.cuda.device_count()}",
            flush=True,
        )

        print(
            "[device] GPU name: "
            f"{torch.cuda.get_device_name(0)}",
            flush=True,
        )

    # ========================================================
    # Dataset
    # ========================================================

    print(
        "[setup] Loading dataset...",
        flush=True,
    )

    if (
        "grid_size"
        in ColorGridDataset.__init__.__code__.co_varnames
    ):
        ds = ColorGridDataset(
            CONFIG["csv_path"],
            grid_size=CONFIG[
                "grid_size"
            ],
        )

    else:
        ds = ColorGridDataset(
            CONFIG["csv_path"]
        )

    x0_ex, cond_ex = (
        ds[0][:2]
    )

    C, H, W = x0_ex.shape

    cond_dim = int(
        cond_ex.numel()
    )

    print(
        "[setup] Dataset loaded: "
        f"N={len(ds)} | "
        f"C={C} H={H} W={W} "
        f"cond_dim={cond_dim}",
        flush=True,
    )

    # ========================================================
    # Baseline checkpoints
    # ========================================================

    baseline_ckpts = (
        list_checkpoints_sorted(
            CONFIG["baseline_dir"]
        )
        if CONFIG[
            "use_baseline_ckpts"
        ]
        else []
    )

    lora_ckpts = (
        list_checkpoints_sorted(
            CONFIG["lora_update_dir"]
        )
        if CONFIG[
            "use_lora_ckpts"
        ]
        else []
    )

    print(
        "[setup] "
        f"baseline_ckpts={len(baseline_ckpts)} | "
        f"lora_ckpts={len(lora_ckpts)}",
        flush=True,
    )

    if (
        CONFIG["use_baseline_ckpts"]
        and not baseline_ckpts
    ):
        raise FileNotFoundError(
            "USE_BASELINE=True but no baseline "
            f"checkpoints found in "
            f"{CONFIG['baseline_dir']}"
        )

    if (
        CONFIG["use_lora_ckpts"]
        and not lora_ckpts
    ):
        raise FileNotFoundError(
            "USE_LORA=True but no LoRA "
            f"checkpoints found in "
            f"{CONFIG['lora_update_dir']}"
        )

    # ========================================================
    # Reference checkpoint
    # ========================================================

    print(
        "[setup] Locating reference checkpoint...",
        flush=True,
    )

    ref_ckpt = (
        CONFIG["reference_ckpt"]
        or latest_checkpoint_in_dir(
            CONFIG["baseline_dir"]
        )
    )

    if ref_ckpt is None:
        raise FileNotFoundError(
            "No reference baseline checkpoint "
            "found in baseline_dir."
        )

    print(
        "[setup] Reference checkpoint: "
        f"{os.path.basename(ref_ckpt)}",
        flush=True,
    )

    # IMPORTANT:
    # Reference model uses the SAME raw/ema switch.
    ref_model, ref_meta = (
        build_model_from_baseline_ckpt(
            ref_ckpt,
            device=device,
            param_source=CONFIG[
                "param_source"
            ],
        )
    )

    T = int(
        ref_meta["T"]
    )

    sched = base.make_linear_schedule(
        T,
        device=device,
    )

    print(
        f"[setup] Diffusion steps T={T}",
        flush=True,
    )

    # ========================================================
    # Query conditioning
    # ========================================================

    vocab = (
        getattr(
            ds,
            "vocab",
            None,
        )
        or ref_meta.get(
            "vocab",
            None,
        )
    )

    if vocab is None:
        raise RuntimeError(
            "No vocab found in dataset or ckpt; "
            "cannot build query cond."
        )

    query_cond = labels_to_cond(
        CONFIG["query_labels"],
        vocab,
        cond_dim,
        device=device,
    )

    # ========================================================
    # FIXED reference diffusion trajectory
    # ========================================================

    print(
        "[setup] Computing FIXED reference trajectory...",
        flush=True,
    )

    traj_use, t_seq, save_steps = (
        compute_reference_trajectory(
            model=ref_model,
            sched=sched,
            cond=query_cond,
            shape=(
                1,
                C,
                H,
                W,
            ),
            seed=int(
                CONFIG["seed"]
            ),
            steps=int(
                CONFIG["ddim_steps"]
            ),
            eta=float(
                CONFIG["eta"]
            ),
            device=device,
            num_keep=int(
                CONFIG["num_traj_t"]
            ),
        )
    )

    K = len(
        traj_use
    )

    snap_ids = pick_snapshot_ids(
        K,
        num_pick=int(
            CONFIG["num_traj_t"]
        ),
    )

    xt_ref_dict = (
        build_xtref_dict_by_snapid(
            traj_use,
            snap_ids,
        )
    )

    print(
        "[setup] Fixed reference trajectory ready: "
        f"K={K} | "
        f"selected_snapshots={len(snap_ids)}",
        flush=True,
    )

    # ========================================================
    # Reference target cache
    # ========================================================

    # Only needed once for reference-target mode.
    reference_eps_target_dict = None

    if (
        CONFIG["query_target"]
        == "reference"
    ):
        print(
            "[setup] Caching reference-target "
            "epsilon predictions...",
            flush=True,
        )

        reference_eps_target_dict = (
            compute_eps_target_by_snapid(
                target_model=ref_model,
                xt_ref_dict=xt_ref_dict,
                query_cond=query_cond,
                snap_ids=snap_ids,
                t_seq=t_seq,
            )
        )

    # ========================================================
    # Candidate set
    # ========================================================

    print(
        "[setup] Building candidate set...",
        flush=True,
    )

    score_items = build_score_items(
        ds,
        CONFIG,
    )

    M_eff = len(
        score_items
    )

    if M_eff == 0:
        raise RuntimeError(
            "No training points selected."
        )

    scores = torch.zeros(
        M_eff,
        dtype=torch.float64,
    )

    print(
        "[candidate-set] "
        f"N={len(ds)} | "
        f"M_selected={M_eff}",
        flush=True,
    )

    def get_one(
        src: int,
        i: int,
    ):
        x0, cond = ds[i][:2]

        x0 = (
            x0
            .unsqueeze(0)
            .to(device)
        )

        if cond.dim() == 1:
            cond = (
                cond
                .unsqueeze(0)
                .to(device)
            )

        else:
            cond = cond.to(
                device
            )

        return x0, cond

    # ========================================================
    # Run metadata
    # ========================================================

    if (
        CONFIG["query_target"]
        == "reference"
    ):
        objective_formula = (
            "mean_k ||eps_theta_c(x_ref_k,t_k)"
            "-eps_theta_ref(x_ref_k,t_k)||_2^2"
        )

    else:
        objective_formula = (
            "mean_k ||eps_theta_c(x_ref_k,t_k)"
            "-eps_theta_c_plus_1(x_ref_k,t_k)||_2^2"
        )

    run_info = {
        "reference_ckpt": ref_ckpt,

        "param_source": CONFIG[
            "param_source"
        ],

        "query_target": CONFIG[
            "query_target"
        ],

        "T": int(T),

        "ddim_steps": int(
            CONFIG["ddim_steps"]
        ),

        "num_snap_used": int(
            len(snap_ids)
        ),

        "snap_ids": [
            int(x)
            for x in snap_ids
        ],

        "save_steps": [
            int(x)
            for x in save_steps
        ],

        "t_seq": [
            int(x)
            for x in t_seq.tolist()
        ],

        "M_requested": int(
            min(
                CONFIG[
                    "max_train_points"
                ],
                len(ds),
            )
        ),

        "M_eff": int(
            M_eff
        ),

        "device": str(
            device
        ),

        "seed": int(
            CONFIG["seed"]
        ),

        "time_started": float(
            t0
        ),

        "query_objective": {
            "name": (
                "trajectory_noise_squared_deviation"
            ),
            "target": CONFIG[
                "query_target"
            ],
            "formula": objective_formula,
            "reference_trajectory": (
                "fixed_from_reference_ckpt"
            ),
            "reference_ckpt": ref_ckpt,
            "snapshot_reduction": "mean",
            "checkpoint_reduction": "sum",
            "checkpoint_weight": float(
                CONFIG[
                    "checkpoint_weight"
                ]
            ),
        },

        "train_mc_samples": int(
            CONFIG[
                "train_mc_samples"
            ]
        ),
    }

    # ========================================================
    # BASELINE checkpoint scoring
    # ========================================================

    if CONFIG[
        "use_baseline_ckpts"
    ]:
        num_baseline_terms = (
            len(baseline_ckpts)
            if CONFIG["query_target"]
            == "reference"
            else max(
                0,
                len(baseline_ckpts) - 1,
            )
        )

        print(
            "\n"
            "[baseline] checkpoints contributing "
            f"to score: {num_baseline_terms}",
            flush=True,
        )

        for ckpt_idx, ckpt_path in enumerate(
            baseline_ckpts
        ):

            # -----------------------------------------------
            # Next-target: final checkpoint has no c+1
            # -----------------------------------------------

            if (
                CONFIG["query_target"]
                == "next_checkpoint"
                and ckpt_idx + 1
                >= len(baseline_ckpts)
            ):
                print(
                    "\n[baseline checkpoint] "
                    f"{ckpt_idx + 1}/"
                    f"{len(baseline_ckpts)} | "
                    f"{os.path.basename(ckpt_path)}"
                )

                print(
                    "[baseline] skip final checkpoint: "
                    "no next checkpoint target.",
                    flush=True,
                )

                continue

            ckpt_t0 = (
                time.perf_counter()
            )

            print(
                "\n[baseline checkpoint] "
                f"{ckpt_idx + 1}/"
                f"{len(baseline_ckpts)} | "
                f"{os.path.basename(ckpt_path)}",
                flush=True,
            )

            # -----------------------------------------------
            # Current checkpoint
            # -----------------------------------------------

            model_k, _ = (
                build_model_from_baseline_ckpt(
                    ckpt_path,
                    device=device,
                    param_source=CONFIG[
                        "param_source"
                    ],
                )
            )

            active = (
                set_active_params_baseline(
                    model_k
                )
            )

            # -----------------------------------------------
            # Select target epsilon predictions
            # -----------------------------------------------

            if (
                CONFIG["query_target"]
                == "reference"
            ):
                eps_target_dict = (
                    reference_eps_target_dict
                )

                if (
                    os.path.abspath(
                        ckpt_path
                    )
                    == os.path.abspath(
                        ref_ckpt
                    )
                ):
                    print(
                        "[warning] current checkpoint "
                        "equals reference checkpoint; "
                        "query objective and query "
                        "gradient are exactly zero.",
                        flush=True,
                    )

            else:
                next_ckpt_path = (
                    baseline_ckpts[
                        ckpt_idx + 1
                    ]
                )

                print(
                    "[baseline] next-checkpoint target: "
                    f"{os.path.basename(next_ckpt_path)}",
                    flush=True,
                )

                next_model, _ = (
                    build_model_from_baseline_ckpt(
                        next_ckpt_path,
                        device=device,
                        param_source=CONFIG[
                            "param_source"
                        ],
                    )
                )

                eps_target_dict = (
                    compute_eps_target_by_snapid(
                        target_model=next_model,
                        xt_ref_dict=xt_ref_dict,
                        query_cond=query_cond,
                        snap_ids=snap_ids,
                        t_seq=t_seq,
                    )
                )

                del next_model

            # -----------------------------------------------
            # Query gradients
            # -----------------------------------------------

            print(
                "[baseline] computing trajectory "
                "query gradients...",
                flush=True,
            )

            gq = compute_g_traj(
                model=model_k,
                active_params=active,
                xt_ref_dict=xt_ref_dict,
                eps_target_dict=eps_target_dict,
                query_cond=query_cond,
                snap_ids=snap_ids,
                t_seq=t_seq,
            )

            # -----------------------------------------------
            # Train-point scores
            # -----------------------------------------------

            for j, (
                src,
                idx,
            ) in enumerate(
                score_items
            ):
                if should_print_item(
                    j,
                    M_eff,
                ):
                    print_item_progress(
                        "baseline score",
                        j,
                        M_eff,
                        extra=(
                            f"ckpt="
                            f"{ckpt_idx + 1}/"
                            f"{len(baseline_ckpts)}"
                        ),
                    )

                x0_train, cond_train = (
                    get_one(
                        src,
                        idx,
                    )
                )

                sc = (
                    score_one_trainpoint_given_gtraj(
                        model=model_k,
                        active_params=active,
                        sched=sched,
                        gq=gq,
                        snap_ids=snap_ids,
                        t_seq=t_seq,
                        x0_train=x0_train,
                        train_cond=cond_train,
                        checkpoint_weight=float(
                            CONFIG[
                                "checkpoint_weight"
                            ]
                        ),
                        train_mc_samples=int(
                            CONFIG[
                                "train_mc_samples"
                            ]
                        ),
                    )
                )

                scores[j] += float(
                    sc.item()
                )

            print(
                "[baseline] done: "
                f"{os.path.basename(ckpt_path)} | "
                f"snapshots={len(snap_ids)} | "
                f"elapsed="
                f"{format_seconds(time.perf_counter() - ckpt_t0)}",
                flush=True,
            )

            del gq
            del model_k

            if (
                device.type
                == "cuda"
            ):
                torch.cuda.empty_cache()

    # ========================================================
    # LoRA checkpoint scoring
    # ========================================================

    if CONFIG[
        "use_lora_ckpts"
    ]:
        num_lora_terms = (
            len(lora_ckpts)
            if CONFIG["query_target"]
            == "reference"
            else max(
                0,
                len(lora_ckpts) - 1,
            )
        )

        print(
            "\n"
            "[lora] checkpoints contributing "
            f"to score: {num_lora_terms}",
            flush=True,
        )

        for ckpt_idx, ckpt_path in enumerate(
            lora_ckpts
        ):

            if (
                CONFIG["query_target"]
                == "next_checkpoint"
                and ckpt_idx + 1
                >= len(lora_ckpts)
            ):
                print(
                    "\n[lora checkpoint] "
                    f"{ckpt_idx + 1}/"
                    f"{len(lora_ckpts)} | "
                    f"{os.path.basename(ckpt_path)}"
                )

                print(
                    "[lora] skip final checkpoint: "
                    "no next checkpoint target.",
                    flush=True,
                )

                continue

            ckpt_t0 = (
                time.perf_counter()
            )

            print(
                "\n[lora checkpoint] "
                f"{ckpt_idx + 1}/"
                f"{len(lora_ckpts)} | "
                f"{os.path.basename(ckpt_path)}",
                flush=True,
            )

            model_k, _ = (
                build_model_from_lora_ckpt(
                    ckpt_path,
                    device=device,
                    baseline_param_source=CONFIG[
                        "param_source"
                    ],
                )
            )

            active = (
                set_active_params_lora(
                    model_k
                )
            )

            if (
                CONFIG["query_target"]
                == "reference"
            ):
                # For LoRA + reference target, compare the
                # LoRA-updated current model against the fixed
                # baseline reference model.
                eps_target_dict = (
                    reference_eps_target_dict
                )

            else:
                next_ckpt_path = (
                    lora_ckpts[
                        ckpt_idx + 1
                    ]
                )

                print(
                    "[lora] next-checkpoint target: "
                    f"{os.path.basename(next_ckpt_path)}",
                    flush=True,
                )

                next_model, _ = (
                    build_model_from_lora_ckpt(
                        next_ckpt_path,
                        device=device,
                        baseline_param_source=CONFIG[
                            "param_source"
                        ],
                    )
                )

                eps_target_dict = (
                    compute_eps_target_by_snapid(
                        target_model=next_model,
                        xt_ref_dict=xt_ref_dict,
                        query_cond=query_cond,
                        snap_ids=snap_ids,
                        t_seq=t_seq,
                    )
                )

                del next_model

            print(
                "[lora] computing trajectory "
                "query gradients...",
                flush=True,
            )

            gq = compute_g_traj(
                model=model_k,
                active_params=active,
                xt_ref_dict=xt_ref_dict,
                eps_target_dict=eps_target_dict,
                query_cond=query_cond,
                snap_ids=snap_ids,
                t_seq=t_seq,
            )

            for j, (
                src,
                idx,
            ) in enumerate(
                score_items
            ):
                if should_print_item(
                    j,
                    M_eff,
                ):
                    print_item_progress(
                        "lora score",
                        j,
                        M_eff,
                        extra=(
                            f"ckpt="
                            f"{ckpt_idx + 1}/"
                            f"{len(lora_ckpts)}"
                        ),
                    )

                x0_train, cond_train = (
                    get_one(
                        src,
                        idx,
                    )
                )

                sc = (
                    score_one_trainpoint_given_gtraj(
                        model=model_k,
                        active_params=active,
                        sched=sched,
                        gq=gq,
                        snap_ids=snap_ids,
                        t_seq=t_seq,
                        x0_train=x0_train,
                        train_cond=cond_train,
                        checkpoint_weight=float(
                            CONFIG[
                                "checkpoint_weight"
                            ]
                        ),
                        train_mc_samples=int(
                            CONFIG[
                                "train_mc_samples"
                            ]
                        ),
                    )
                )

                scores[j] += float(
                    sc.item()
                )

            print(
                "[lora] done: "
                f"{os.path.basename(ckpt_path)} | "
                f"snapshots={len(snap_ids)} | "
                f"elapsed="
                f"{format_seconds(time.perf_counter() - ckpt_t0)}",
                flush=True,
            )

            del gq
            del model_k

            if (
                device.type
                == "cuda"
            ):
                torch.cuda.empty_cache()

    # ========================================================
    # Top-k
    # ========================================================

    topk = min(
        int(
            CONFIG["topk"]
        ),
        M_eff,
    )

    vals, ord_idx = torch.topk(
        scores,
        k=topk,
        largest=True,
    )

    top = []

    for r in range(
        topk
    ):
        j = int(
            ord_idx[r].item()
        )

        src, train_idx = (
            score_items[j]
        )

        src = int(src)
        train_idx = int(
            train_idx
        )

        idx_tag = (
            f"{train_idx}_{src}"
            if src != 0
            else str(train_idx)
        )

        top.append(
            {
                "idx": train_idx,
                "src": src,
                "idx_tag": idx_tag,
                "score": float(
                    vals[r].item()
                ),
            }
        )

    # ========================================================
    # Save
    # ========================================================

    print(
        "[save] Writing outputs...",
        flush=True,
    )

    ensure_dir(
        CONFIG["out_dir"]
    )

    run_info[
        "elapsed_sec"
    ] = float(
        time.perf_counter()
        - t0
    )

    save_json(
        os.path.join(
            CONFIG["out_dir"],
            "run_config.json",
        ),
        CONFIG,
    )

    save_json(
        os.path.join(
            CONFIG["out_dir"],
            "run_info.json",
        ),
        run_info,
    )

    score_indices_payload = {
        "N_eff": int(
            M_eff
        ),

        "items": [
            {
                "src": int(s),
                "idx": int(i),
            }
            for (
                s,
                i,
            ) in score_items
        ],

        "idx_list_1": CONFIG.get(
            "idx_list_1",
            None,
        ),

        "idx_list_2": CONFIG.get(
            "idx_list_2",
            None,
        ),

        "idx_list_3": CONFIG.get(
            "idx_list_3",
            None,
        ),

        "idx_list_4": CONFIG.get(
            "idx_list_4",
            None,
        ),

        "idx_list_5": CONFIG.get(
            "idx_list_5",
            None,
        ),

        "idx_list_6": CONFIG.get(
            "idx_list_6",
            None,
        ),

        "idx_col_name": CONFIG.get(
            "idx_col_name",
            "idx",
        ),

        "use_two_index_lists": bool(
            CONFIG.get(
                "use_two_index_lists",
                False,
            )
        ),

        "use_three_index_lists": bool(
            CONFIG.get(
                "use_three_index_lists",
                False,
            )
        ),

        "use_six_index_lists": bool(
            CONFIG.get(
                "use_six_index_lists",
                False,
            )
        ),
    }

    save_json(
        os.path.join(
            CONFIG["out_dir"],
            "score_indices.json",
        ),
        score_indices_payload,
    )

    save_json(
        os.path.join(
            CONFIG["out_dir"],
            "result_topk.json",
        ),
        {
            "N_eff": int(
                M_eff
            ),
            "topk": int(
                topk
            ),
            "top": top,
        },
    )

    np.save(
        os.path.join(
            CONFIG["out_dir"],
            "scores.npy",
        ),
        scores.cpu().numpy(),
    )

    dt = (
        time.perf_counter()
        - t0
    )

    print(
        "\n[saved] "
        f"{CONFIG['out_dir']}/run_config.json",
        flush=True,
    )

    print(
        "[saved] "
        f"{CONFIG['out_dir']}/run_info.json",
        flush=True,
    )

    print(
        "[saved] "
        f"{CONFIG['out_dir']}/score_indices.json",
        flush=True,
    )

    print(
        "[saved] "
        f"{CONFIG['out_dir']}/result_topk.json",
        flush=True,
    )

    print(
        "[saved] "
        f"{CONFIG['out_dir']}/scores.npy",
        flush=True,
    )

    print(
        "\n(done) "
        f"elapsed={format_seconds(dt)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
