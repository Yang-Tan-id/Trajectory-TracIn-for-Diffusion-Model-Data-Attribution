# x3_endpoint_das_jax_logic_pytorch.py
"""
PyTorch X3 Endpoint Projected DAS, aligned to the later JAX DAS logic.

Method alignment with the later JAX version:
1. Fixed query endpoint x0_ref from a reference checkpoint.
2. For each checkpoint / diffusion timestep / MC sample:
   a) sample one random output probe v
   b) define scalar projected epsilon:
          s(theta) = <eps_theta(x_t,t,c), v> / sqrt(m)
   c) compute projected parameter gradient:
          phi = CountSketch(grad_theta s(theta))
   d) optionally L2-normalize phi (default True)
   e) compute scalar residual:
          r = <eps_theta(x_t,t,c) - noise, v> / sqrt(m)
3. Build projected Gram:
        H = sum_i phi_i phi_i^T
4. For each damping lambda:
        u = (H + lambda I)^(-1) phi_q
5. Candidate score:
        raw_i = (phi_i^T u) * r_i
   optionally:
        raw_i /= 1 - phi_i^T (H + lambda I)^(-1) phi_i
   then:
        score_i = raw_i^2
6. Average scores over checkpoint x timestep x MC terms.

Additional X3 conveniences:
- PARAM_SOURCE switch: "ema" or "raw"
- six-list candidate selection preserved from the old X3 experiments
- lambda sweep is a first-class CONFIG option (no environment variable required)

This intentionally matches the *logic* of the later JAX DAS more closely than
the older X3 endpoint_das_projected.py implementation.
"""

import os
import glob
import json
import random
import time
import math
import hashlib
import re
from typing import Dict, Any, List, Tuple, Optional, Set, Sequence

import numpy as np
import torch
import torch.nn as nn

import x3pixel_DM_training as base
from dataset_loader import ColorGridDataset


# ============================================================
# Repro / seeds
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def _stable_int_seed(*parts: Any) -> int:
    s = "|".join(map(str, parts)).encode("utf-8")
    h = hashlib.sha256(s).digest()
    v = int.from_bytes(h[:8], "little", signed=False)
    return int(v % (2**63 - 1))


def make_torch_generator(
    device: torch.device,
    *seed_parts: Any,
) -> torch.Generator:
    g = torch.Generator(device=str(device))
    g.manual_seed(_stable_int_seed(*seed_parts))
    return g


# ============================================================
# General helpers
# ============================================================

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


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def save_json(path: str, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def cleanup_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def damping_output_tag(value: float) -> str:
    text = (
        ("%g" % float(value))
        .replace("+", "_")
        .replace("-", "neg_")
        .replace(".", "p")
    )
    return text or "0"


# ============================================================
# Checkpoint discovery
# ============================================================

def _natural_key(path: str):
    name = os.path.basename(path)
    parts = re.split(r"(\d+)", name)
    return [int(p) if p.isdigit() else p for p in parts]


def list_checkpoints_sorted(
    dir_path: str,
    pattern: str = "*.pt",
) -> List[str]:
    if not os.path.isdir(dir_path):
        return []
    paths = glob.glob(os.path.join(dir_path, pattern))
    paths.sort(key=_natural_key)
    return paths


def filter_checkpoints(
    ckpts: List[str],
    ckpt_stride: int,
    max_num_ckpts: Optional[int],
) -> List[str]:
    if int(ckpt_stride) <= 0:
        raise ValueError(
            f"ckpt_stride must be >=1, got {ckpt_stride}"
        )

    out = ckpts[:: int(ckpt_stride)]

    if max_num_ckpts is not None:
        n = int(max_num_ckpts)
        if n <= 0:
            raise ValueError(
                f"max_num_ckpts must be positive or None, got {n}"
            )
        out = out[-n:]

    if not out:
        raise RuntimeError(
            "Checkpoint filtering removed all checkpoints."
        )

    return out


# ============================================================
# Vocab / conditioning
# ============================================================

def vocab_to_index(vocab: Any) -> Dict[str, int]:
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
        lab for lab in labels
        if lab not in m
    ]

    if missing:
        raise KeyError(
            f"Query labels not found in vocab: {missing[:10]}"
        )

    for lab in labels:
        cond[m[lab]] = 1.0

    return cond.unsqueeze(0)


# ============================================================
# RAW / EMA checkpoint selection
# ============================================================

def normalize_param_source(x: str) -> str:
    x = str(x).strip().lower()

    aliases = {
        "raw": "raw",
        "model_state": "raw",
        "params": "raw",
        "ema": "ema",
        "ema_model_state": "ema",
    }

    if x not in aliases:
        raise ValueError(
            f"param_source must be 'raw' or 'ema', got {x!r}"
        )

    return aliases[x]


def select_model_state(
    ckpt: Dict[str, Any],
    *,
    param_source: str,
    ckpt_path: str,
):
    source = normalize_param_source(param_source)

    if source == "raw":
        if "model_state" not in ckpt:
            raise KeyError(
                f"RAW requested but model_state is missing: {ckpt_path}"
            )
        return ckpt["model_state"]

    if "ema_model_state" not in ckpt:
        raise KeyError(
            "EMA requested but ema_model_state is missing: "
            f"{ckpt_path}\n"
            "Use param_source='raw' for older checkpoints, "
            "or retrain with the updated X3 trainer that saves EMA."
        )

    return ckpt["ema_model_state"]


def build_model_from_baseline_ckpt(
    baseline_ckpt_path: str,
    device: torch.device,
    *,
    param_source: str = "ema",
) -> Tuple[nn.Module, Dict[str, Any]]:
    ckpt = torch.load(
        baseline_ckpt_path,
        map_location=str(device),
    )

    for k in ("T", "cond_dim"):
        if k not in ckpt:
            raise KeyError(
                f"Baseline ckpt missing {k}: {baseline_ckpt_path}"
            )

    cond_dim = int(ckpt["cond_dim"])
    base_ch = int(ckpt.get("base_ch", 64))
    time_dim = int(ckpt.get("time_dim", 128))
    grid_size = int(ckpt.get("grid_size", 3))

    model = base.CondEpsModel(
        in_ch=3,
        cond_dim=cond_dim,
        base_ch=base_ch,
        time_dim=time_dim,
    ).to(device)

    sd = select_model_state(
        ckpt,
        param_source=param_source,
        ckpt_path=baseline_ckpt_path,
    )

    model.load_state_dict(
        sd,
        strict=True,
    )
    model.eval()

    meta = {
        "T": int(ckpt["T"]),
        "cond_dim": cond_dim,
        "vocab": ckpt.get("vocab", None),
        "base_ch": base_ch,
        "time_dim": time_dim,
        "grid_size": grid_size,
        "param_source": normalize_param_source(param_source),
        "epoch": ckpt.get("epoch", None),
        "global_step": ckpt.get("global_step", None),
    }

    return model, meta


# ============================================================
# Active parameters
# ============================================================

def set_active_params_baseline(
    model: nn.Module,
) -> List[nn.Parameter]:
    active = []

    for p in model.parameters():
        p.requires_grad_(True)
        active.append(p)

    return active


# ============================================================
# DDIM reference endpoint
# ============================================================

@torch.no_grad()
def compute_reference_endpoint(
    model: nn.Module,
    sched,
    cond: torch.Tensor,
    shape: Tuple[int, int, int, int],
    seed: int,
    steps: int,
    eta: float,
    device: torch.device,
) -> torch.Tensor:
    model.eval()

    traj = base.ddim_sample(
        model=model,
        sched=sched,
        cond=cond,
        shape=shape,
        seed=seed,
        steps=steps,
        eta=eta,
        device=str(device),
    )

    return traj[-1].detach()


# ============================================================
# Candidate-set helpers
# ============================================================

def load_index_list(
    path: str,
    col: str = "idx",
) -> List[int]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    with open(path, "r") as f:
        lines = [
            ln.strip()
            for ln in f.readlines()
            if ln.strip()
        ]

    if not lines:
        return []

    if lines[0].lower() == col.lower():
        lines = lines[1:]

    return [
        int(ln.split(",")[0])
        for ln in lines
    ]


def build_candidate_items_from_lists(
    N: int,
    idx_lists: List[List[int]],
    extra_random_points: int,
    seed: int,
    shuffle: bool = True,
) -> List[Tuple[int, int]]:
    """
    Keeps old X3 six-list bookkeeping.

    src=1..K: listed/case-line points
    src=0: extra random points
    """
    clean_lists: List[List[int]] = []

    for lst in idx_lists:
        clean = sorted(
            {
                int(i)
                for i in lst
                if 0 <= int(i) < N
            }
        )
        clean_lists.append(clean)

    seen: Set[int] = set()
    base_items: List[Tuple[int, int]] = []

    for k, lst in enumerate(clean_lists):
        src_id = k + 1

        for i in lst:
            if i in seen:
                continue

            seen.add(i)
            base_items.append(
                (src_id, i)
            )

    remaining = [
        i
        for i in range(N)
        if i not in seen
    ]

    extras: List[int] = []

    if extra_random_points == -1:
        extras = remaining

    elif extra_random_points > 0:
        rnd = random.Random(seed)
        k = min(
            int(extra_random_points),
            len(remaining),
        )

        if k > 0:
            extras = rnd.sample(
                remaining,
                k=k,
            )

    items = (
        base_items
        + [
            (0, int(i))
            for i in extras
        ]
    )

    if shuffle:
        rnd = random.Random(
            seed + 999
        )
        rnd.shuffle(items)

    return items


def get_one(
    ds,
    idx: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x0, cond = ds[idx][:2]

    x0 = (
        x0
        .unsqueeze(0)
        .to(device)
    )

    cond = (
        cond
        .unsqueeze(0)
        .to(device)
        if cond.dim() == 1
        else cond.to(device)
    )

    return x0, cond


# ============================================================
# CountSketch
# ============================================================

def build_countsketch_specs(
    active: List[nn.Parameter],
    d: int,
    *,
    device: torch.device,
    seed_parts: Tuple[Any, ...],
):
    """
    Build ONE shared CountSketch map for the entire DAS term.
    This matches the later JAX logic more closely than using
    different projection hashes per example/output row.
    """
    specs = []

    for tensor_idx, p in enumerate(active):
        n = p.numel()

        gen = make_torch_generator(
            device,
            *seed_parts,
            "tensor",
            tensor_idx,
            n,
            d,
        )

        idx = torch.randint(
            0,
            d,
            (n,),
            generator=gen,
            device=device,
            dtype=torch.int64,
        )

        sign = (
            torch.randint(
                0,
                2,
                (n,),
                generator=gen,
                device=device,
                dtype=torch.int8,
            )
            * 2
            - 1
        ).to(torch.float32)

        specs.append(
            (idx, sign)
        )

    return specs


def project_grad_with_specs(
    grads: Tuple[Optional[torch.Tensor], ...],
    specs,
    d: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    out = torch.zeros(
        d,
        device=device,
        dtype=torch.float32,
    )

    for g, (idx, sign) in zip(
        grads,
        specs,
    ):
        if g is None:
            continue

        flat = (
            g
            .reshape(-1)
            .to(
                device=device,
                dtype=torch.float32,
            )
        )

        out.index_add_(
            0,
            idx,
            sign * flat,
        )

    return (
        out
        / math.sqrt(float(d))
    )


def maybe_normalize_phi(
    phi: torch.Tensor,
    normalize: bool,
    eps: float,
) -> torch.Tensor:
    if not normalize:
        return phi

    return (
        phi
        / (
            torch.linalg.vector_norm(phi)
            + float(eps)
        )
    )


# ============================================================
# JAX-aligned epsilon-probe feature
# ============================================================

def sample_noise(
    x0: torch.Tensor,
    *,
    rng: torch.Generator,
) -> torch.Tensor:
    return torch.randn(
        x0.shape,
        generator=rng,
        device=x0.device,
        dtype=x0.dtype,
    )


def sample_output_probe(
    shape: Tuple[int, ...],
    *,
    device: torch.device,
    rng: torch.Generator,
) -> torch.Tensor:
    return torch.randn(
        shape,
        generator=rng,
        device=device,
        dtype=torch.float32,
    )


def compute_projected_eps_feature(
    model: nn.Module,
    active: List[nn.Parameter],
    sched,
    x0: torch.Tensor,
    cond: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
    output_probe: torch.Tensor,
    *,
    projection_specs,
    proj_dim: int,
    device: torch.device,
    normalize_projected_grads: bool,
    normalize_eps: float,
) -> Tuple[float, torch.Tensor]:
    """
    Matches later JAX make_projected_eps_grad_fn:

      xt = q_sample(...)
      pred = eps_theta(xt,t,cond)
      residual = pred - noise

      scalar_eps(theta)
        = sum(pred * output_probe) / sqrt(pred.numel())

      phi
        = CountSketch(grad_theta scalar_eps)

      residual_scalar
        = sum(residual * output_probe) / sqrt(residual.numel())
    """
    xt = base.q_sample(
        x0,
        t,
        noise,
        sched,
    )

    pred = model(
        xt,
        t,
        cond,
    )

    denom = math.sqrt(
        float(pred.numel())
    )

    scalar_eps = (
        (pred * output_probe).sum()
        / denom
    )

    grads = torch.autograd.grad(
        scalar_eps,
        active,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    phi = project_grad_with_specs(
        grads,
        projection_specs,
        proj_dim,
        device=device,
    )

    phi = maybe_normalize_phi(
        phi,
        normalize_projected_grads,
        normalize_eps,
    )

    residual_scalar = (
        (
            (pred.detach() - noise)
            * output_probe
        ).sum()
        / denom
    )

    return (
        float(
            residual_scalar.item()
        ),
        phi.detach(),
    )


# ============================================================
# One DAS term, all lambdas
# ============================================================

def compute_one_das_term_all_lambdas(
    *,
    model: nn.Module,
    active: List[nn.Parameter],
    sched,
    ds,
    score_items: List[Tuple[int, int]],
    x0_ref: torch.Tensor,
    query_cond: torch.Tensor,
    t_tensor: torch.Tensor,
    ckpt_i: int,
    t_value: int,
    mc_i: int,
    config: Dict[str, Any],
    device: torch.device,
    damping_values: Sequence[float],
) -> Tuple[Dict[float, np.ndarray], float, float]:
    """
    Compute projected train/query features ONCE, then reuse them
    for every lambda in damping_values.
    """
    proj_dim = int(
        config["proj_dim"]
    )

    normalize_projected_grads = bool(
        config["normalize_projected_grads"]
    )

    normalize_eps = float(
        config["normalize_eps"]
    )

    # --------------------------------------------------------
    # Shared projection map for this checkpoint/t/MC term
    # --------------------------------------------------------

    projection_specs = build_countsketch_specs(
        active,
        proj_dim,
        device=device,
        seed_parts=(
            config["seed"],
            "pdas_gradient_projection",
            ckpt_i,
            t_value,
            mc_i,
        ),
    )

    # --------------------------------------------------------
    # Shared random output probe for query + all train points
    # --------------------------------------------------------

    rng_probe = make_torch_generator(
        device,
        config["seed"],
        "pdas_output_probe",
        ckpt_i,
        t_value,
        mc_i,
    )

    output_probe = sample_output_probe(
        tuple(x0_ref.shape),
        device=device,
        rng=rng_probe,
    )

    # --------------------------------------------------------
    # Query feature
    # --------------------------------------------------------

    rng_q = make_torch_generator(
        device,
        config["seed"],
        "pdas_q",
        ckpt_i,
        t_value,
        mc_i,
    )

    noise_q = sample_noise(
        x0_ref,
        rng=rng_q,
    )

    query_residual, phi_q = (
        compute_projected_eps_feature(
            model=model,
            active=active,
            sched=sched,
            x0=x0_ref,
            cond=query_cond,
            t=t_tensor,
            noise=noise_q,
            output_probe=output_probe,
            projection_specs=projection_specs,
            proj_dim=proj_dim,
            device=device,
            normalize_projected_grads=normalize_projected_grads,
            normalize_eps=normalize_eps,
        )
    )

    # --------------------------------------------------------
    # Train features + Gram
    # --------------------------------------------------------

    M = len(
        score_items
    )

    phi_cache = np.empty(
        (M, proj_dim),
        dtype=np.float32,
    )

    residual_cache = np.empty(
        (M,),
        dtype=np.float32,
    )

    H_base = np.zeros(
        (proj_dim, proj_dim),
        dtype=np.float32,
    )

    progress_every = max(
        1,
        int(
            config[
                "progress_every_points"
            ]
        ),
    )

    train_started = (
        time.perf_counter()
    )

    for j, (
        _src,
        idx,
    ) in enumerate(
        score_items
    ):
        x0_i, cond_i = get_one(
            ds,
            idx,
            device,
        )

        rng_i = make_torch_generator(
            device,
            config["seed"],
            "pdas_tr",
            ckpt_i,
            t_value,
            mc_i,
            idx,
        )

        noise_i = sample_noise(
            x0_i,
            rng=rng_i,
        )

        residual_i, phi_i = (
            compute_projected_eps_feature(
                model=model,
                active=active,
                sched=sched,
                x0=x0_i,
                cond=cond_i,
                t=t_tensor,
                noise=noise_i,
                output_probe=output_probe,
                projection_specs=projection_specs,
                proj_dim=proj_dim,
                device=device,
                normalize_projected_grads=normalize_projected_grads,
                normalize_eps=normalize_eps,
            )
        )

        phi_np = (
            phi_i
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        phi_cache[j] = (
            phi_np
        )

        residual_cache[j] = (
            residual_i
        )

        H_base += np.outer(
            phi_np,
            phi_np,
        ).astype(
            np.float32
        )

        if (
            j == 0
            or (j + 1) % progress_every == 0
            or (j + 1) == M
        ):
            elapsed_feat = time.perf_counter() - train_started
            done_feat = j + 1
            eta_feat = (
                elapsed_feat / float(done_feat) * float(M - done_feat)
                if done_feat > 0 else 0.0
            )
            print(
                "    [features] "
                f"{done_feat}/{M} ({100.0*done_feat/M:.1f}%) | "
                f"elapsed={format_seconds(elapsed_feat)} | "
                f"eta≈{format_seconds(eta_feat)}",
                flush=True,
            )

        del (
            x0_i,
            cond_i,
            noise_i,
            phi_i,
        )

    # --------------------------------------------------------
    # Lambda sweep
    # --------------------------------------------------------

    eye = np.eye(
        proj_dim,
        dtype=np.float32,
    )

    phi_q_np = (
        phi_q
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32,
            copy=False,
        )
    )

    scores_by_lambda: Dict[
        float,
        np.ndarray,
    ] = {}

    use_sm = bool(
        config[
            "use_sherman_morrison_denominator"
        ]
    )

    for damping in damping_values:
        damping = float(
            damping
        )

        print(
            f"    [lambda] solving lambda={damping:g}",
            flush=True,
        )

        H = (
            H_base
            + damping * eye
        )

        # u = (H + lambda I)^-1 phi_q
        u = np.linalg.solve(
            H,
            phi_q_np,
        )

        # raw_i = (phi_i^T u) * residual_i
        raw = (
            phi_cache @ u
        ).astype(
            np.float64
        )

        raw *= (
            residual_cache
            .astype(np.float64)
        )

        # Optional same denominator used by later JAX DAS.
        if use_sm:
            denominator = np.empty(
                (M,),
                dtype=np.float64,
            )

            denom_batch = max(
                1,
                int(
                    config[
                        "denominator_solve_batch_size"
                    ]
                ),
            )

            for start in range(
                0,
                M,
                denom_batch,
            ):
                end = min(
                    start
                    + denom_batch,
                    M,
                )

                phi_chunk = (
                    phi_cache[
                        start:end
                    ]
                )

                # Solve many RHS together:
                # solved[b] = H^-1 phi_b
                solved = np.linalg.solve(
                    H,
                    phi_chunk.T,
                ).T

                leverage = np.einsum(
                    "bi,bi->b",
                    phi_chunk,
                    solved,
                    dtype=np.float64,
                )

                denom = (
                    1.0
                    - leverage
                )

                denom = np.where(
                    np.abs(denom) < 1e-6,
                    np.where(
                        denom >= 0.0,
                        1e-6,
                        -1e-6,
                    ),
                    denom,
                )

                denominator[
                    start:end
                ] = denom

            raw /= denominator

        scores_by_lambda[
            damping
        ] = np.square(
            raw
        )

    return (
        scores_by_lambda,
        float(
            np.mean(
                residual_cache
            )
        ),
        float(
            query_residual
        ),
    )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Experiment identity
    # --------------------------------------------------------

    MODEL = "model_109900"
    LORA = "r"
    CUR_MODEL = f"{MODEL}_{LORA}"

    # --------------------------------------------------------
    # IMPORTANT:
    # Match later JAX DAS default parameter source = EMA.
    # Switch to "raw" only for a raw-vs-EMA ablation.
    # --------------------------------------------------------

    PARAM_SOURCE = "ema"
    # PARAM_SOURCE = "raw"

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

        baseline_dir=(
            f"models_checkpoints/"
            f"{LORA}/{MODEL}/baseline"
        ),

        param_source=PARAM_SOURCE,

        # ----------------------------------------------------
        # query / endpoint
        # ----------------------------------------------------

        query_labels=[
            "background_color_red",
            "background_color_blue",
            "background_color_yellow",
        ],

        ddim_steps=1000,
        eta=0.0,

        # ----------------------------------------------------
        # DAS terms
        # Match later JAX-style settings as desired.
        # ----------------------------------------------------

        timesteps=[
            0,
        ],

        num_mc_noise=1,

        # ----------------------------------------------------
        # Projection / normalization
        # ----------------------------------------------------

        proj_dim=4096,

        # IMPORTANT:
        # later JAX DAS default is True
        normalize_projected_grads=True,

        normalize_eps=1e-8,

        # IMPORTANT:
        # later JAX DAS default is False
        use_sherman_morrison_denominator=False,

        denominator_solve_batch_size=64,

        # ----------------------------------------------------
        # Lambda / damping
        # ----------------------------------------------------

        damping=2.0,

        damping_sweep_enabled=True,

        damping_sweep_values=[
            0.1,
            0.2,
            0.5,
            1.0,
            2.0,
            5.0,
            10.0,
            20.0,
            50.0,
        ],

        # ----------------------------------------------------
        # Checkpoint selection
        # later JAX DAS defaults to last 1 checkpoint.
        # ----------------------------------------------------

        ckpt_stride=1,
        max_num_ckpts=1,

        # ----------------------------------------------------
        # Candidate set
        # Preserve X3 six-list bookkeeping.
        # ----------------------------------------------------

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

        # 0 = only listed points
        # -1 = all remaining dataset points too
        extra_random_points=0,

        shuffle_candidates=True,

        # ----------------------------------------------------
        # progress / save
        # ----------------------------------------------------

        progress_every_points=50,

        topk=2000,

        out_dir=(
            "the_other_runs/"
            f"{CUR_MODEL}_endpoint_das_"
            f"jax_logic_pytorch/"
            f"{PARAM_SOURCE}"
        ),
    )

    # ========================================================
    # Setup
    # ========================================================

    t0 = (
        time.perf_counter()
    )

    device = torch.device(
        CONFIG["device"]
    )

    set_seed(
        int(
            CONFIG["seed"]
        )
    )

    CONFIG["param_source"] = (
        normalize_param_source(
            CONFIG["param_source"]
        )
    )

    print("=" * 90)
    print(
        "X3 Endpoint Projected DAS "
        "(PyTorch, aligned to later JAX logic)"
    )
    print(
        f"param_source              : "
        f"{CONFIG['param_source']}"
    )
    print(
        f"proj_dim                  : "
        f"{CONFIG['proj_dim']}"
    )
    print(
        f"normalize_projected_grads : "
        f"{CONFIG['normalize_projected_grads']}"
    )
    print(
        f"sherman_morrison_denom    : "
        f"{CONFIG['use_sherman_morrison_denominator']}"
    )
    print(
        f"lambda_sweep              : "
        f"{CONFIG['damping_sweep_enabled']}"
    )
    print("=" * 90)

    # ========================================================
    # Dataset
    # ========================================================

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

    C, H, W = (
        x0_ex.shape
    )

    cond_dim = int(
        cond_ex.numel()
    )

    print(
        f"[dataset] N={len(ds)} | "
        f"C={C} H={H} W={W} | "
        f"cond_dim={cond_dim}",
        flush=True,
    )

    # ========================================================
    # Checkpoints
    # ========================================================

    baseline_ckpts = (
        list_checkpoints_sorted(
            CONFIG[
                "baseline_dir"
            ]
        )
    )

    if not baseline_ckpts:
        raise FileNotFoundError(
            "No baseline checkpoints found in: "
            f"{CONFIG['baseline_dir']}"
        )

    baseline_ckpts = (
        filter_checkpoints(
            baseline_ckpts,
            ckpt_stride=int(
                CONFIG[
                    "ckpt_stride"
                ]
            ),
            max_num_ckpts=CONFIG[
                "max_num_ckpts"
            ],
        )
    )

    print(
        f"[checkpoints] selected={len(baseline_ckpts)}",
        flush=True,
    )

    for p in baseline_ckpts:
        print(
            "   ",
            os.path.basename(p),
            flush=True,
        )

    # Same as later JAX DAS:
    # reference defaults to selected last checkpoint.
    ref_ckpt = (
        baseline_ckpts[-1]
    )

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

    # ========================================================
    # Query condition + endpoint
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
            "No vocab found."
        )

    query_cond = (
        labels_to_cond(
            CONFIG[
                "query_labels"
            ],
            vocab,
            cond_dim,
            device,
        )
    )

    print(
        "[reference] computing endpoint...",
        flush=True,
    )

    x0_ref = (
        compute_reference_endpoint(
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
                CONFIG[
                    "ddim_steps"
                ]
            ),
            eta=float(
                CONFIG[
                    "eta"
                ]
            ),
            device=device,
        )
    )

    print(
        "[reference] endpoint ready",
        flush=True,
    )

    # ========================================================
    # Candidate set
    # ========================================================

    idx_lists = [
        load_index_list(
            CONFIG[
                f"idx_list_{k}"
            ],
            col=CONFIG[
                "idx_col_name"
            ],
        )
        for k in range(
            1,
            7,
        )
    ]

    score_items = (
        build_candidate_items_from_lists(
            N=len(ds),
            idx_lists=idx_lists,
            extra_random_points=int(
                CONFIG[
                    "extra_random_points"
                ]
            ),
            seed=int(
                CONFIG["seed"]
            ),
            shuffle=bool(
                CONFIG[
                    "shuffle_candidates"
                ]
            ),
        )
    )

    M = len(
        score_items
    )

    if M == 0:
        raise RuntimeError(
            "No candidate points selected."
        )

    print(
        f"[candidate-set] M={M}",
        flush=True,
    )

    # ========================================================
    # Lambda configuration
    # ========================================================

    if CONFIG[
        "damping_sweep_enabled"
    ]:
        damping_values = tuple(
            float(v)
            for v in CONFIG[
                "damping_sweep_values"
            ]
        )
    else:
        damping_values = (
            float(
                CONFIG[
                    "damping"
                ]
            ),
        )

    if not damping_values:
        damping_values = (
            float(
                CONFIG[
                    "damping"
                ]
            ),
        )

    scores_by_lambda = {
        float(lam): np.zeros(
            (M,),
            dtype=np.float64,
        )
        for lam in damping_values
    }

    timesteps = [
        int(t)
        for t in CONFIG[
            "timesteps"
        ]
    ]

    num_mc_noise = int(
        CONFIG[
            "num_mc_noise"
        ]
    )

    total_terms = (
        len(baseline_ckpts)
        * len(timesteps)
        * num_mc_noise
    )

    terms_done = 0

    # ========================================================
    # DAS loop
    # ========================================================

    for ckpt_i, ckpt_path in enumerate(
        baseline_ckpts
    ):
        ckpt_name = os.path.basename(
            ckpt_path
        )

        print(
            f"\n[checkpoint] "
            f"{ckpt_i + 1}/"
            f"{len(baseline_ckpts)} | "
            f"{ckpt_name}",
            flush=True,
        )

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

        model_k.eval()

        for t_value in timesteps:
            if not (
                0 <= t_value < T
            ):
                raise ValueError(
                    f"Invalid timestep {t_value}; T={T}"
                )

            t_tensor = torch.tensor(
                [t_value],
                device=device,
                dtype=torch.long,
            )

            for mc_i in range(
                num_mc_noise
            ):
                term_t0 = (
                    time.perf_counter()
                )

                terms_done += 1

                print(
                    f"\n[term] "
                    f"{terms_done}/{total_terms} | "
                    f"ckpt={ckpt_i + 1} | "
                    f"t={t_value} | "
                    f"mc={mc_i + 1}/{num_mc_noise}",
                    flush=True,
                )

                term_scores, avg_train_resid, query_resid = (
                    compute_one_das_term_all_lambdas(
                        model=model_k,
                        active=active,
                        sched=sched,
                        ds=ds,
                        score_items=score_items,
                        x0_ref=x0_ref,
                        query_cond=query_cond,
                        t_tensor=t_tensor,
                        ckpt_i=ckpt_i,
                        t_value=t_value,
                        mc_i=mc_i,
                        config=CONFIG,
                        device=device,
                        damping_values=damping_values,
                    )
                )

                for (
                    lam,
                    term_score,
                ) in term_scores.items():
                    scores_by_lambda[
                        float(lam)
                    ] += term_score

                print(
                    f"[term done] "
                    f"query_resid={query_resid:.6f} | "
                    f"avg_train_resid={avg_train_resid:.6f} | "
                    f"elapsed="
                    f"{format_seconds(time.perf_counter() - term_t0)}",
                    flush=True,
                )

                cleanup_cuda()

        del (
            model_k,
            active,
        )

        cleanup_cuda()

    # ========================================================
    # Aggregate + save one folder per lambda
    # ========================================================

    save_root = CONFIG[
        "out_dir"
    ]

    for (
        lam,
        scores,
    ) in scores_by_lambda.items():
        if total_terms > 0:
            scores = (
                scores
                / float(total_terms)
            )

        if CONFIG[
            "damping_sweep_enabled"
        ]:
            out_dir = os.path.join(
                save_root,
                f"lambda_"
                f"{damping_output_tag(lam)}",
            )
        else:
            out_dir = (
                save_root
            )

        ensure_dir(
            out_dir
        )

        topk = min(
            int(
                CONFIG[
                    "topk"
                ]
            ),
            M,
        )

        order = np.argsort(
            -scores
        )[:topk]

        top = []

        for rank_idx in order:
            src, train_idx = (
                score_items[
                    int(rank_idx)
                ]
            )

            top.append(
                {
                    "idx": int(
                        train_idx
                    ),
                    "src": int(
                        src
                    ),
                    "score": float(
                        scores[
                            int(rank_idx)
                        ]
                    ),
                }
            )

        run_info = {
            "method": (
                "projected_eps_probe_das_"
                "pytorch_jax_logic"
            ),

            "reference_ckpt": (
                ref_ckpt
            ),

            "param_source": (
                CONFIG[
                    "param_source"
                ]
            ),

            "used_ckpts": [
                os.path.basename(p)
                for p in baseline_ckpts
            ],

            "num_ckpts": int(
                len(
                    baseline_ckpts
                )
            ),

            "T": int(T),

            "ddim_steps": int(
                CONFIG[
                    "ddim_steps"
                ]
            ),

            "timesteps": (
                timesteps
            ),

            "num_mc_noise": int(
                num_mc_noise
            ),

            "proj_dim": int(
                CONFIG[
                    "proj_dim"
                ]
            ),

            "normalize_projected_grads": bool(
                CONFIG[
                    "normalize_projected_grads"
                ]
            ),

            "normalize_eps": float(
                CONFIG[
                    "normalize_eps"
                ]
            ),

            "use_sherman_morrison_denominator": bool(
                CONFIG[
                    "use_sherman_morrison_denominator"
                ]
            ),

            "damping": float(
                lam
            ),

            "damping_sweep_enabled": bool(
                CONFIG[
                    "damping_sweep_enabled"
                ]
            ),

            "damping_sweep_values": [
                float(v)
                for v in damping_values
            ],

            "num_terms_total": int(
                total_terms
            ),

            "M_scored": int(M),

            "elapsed_sec": float(
                time.perf_counter()
                - t0
            ),
        }

        save_json(
            os.path.join(
                out_dir,
                "run_config.json",
            ),
            CONFIG,
        )

        save_json(
            os.path.join(
                out_dir,
                "run_info.json",
            ),
            run_info,
        )

        save_json(
            os.path.join(
                out_dir,
                "score_indices.json",
            ),
            {
                "M_selected": int(M),
                "items": [
                    {
                        "src": int(src),
                        "idx": int(idx),
                    }
                    for (
                        src,
                        idx,
                    ) in score_items
                ],
            },
        )

        save_json(
            os.path.join(
                out_dir,
                "result_topk.json",
            ),
            {
                "M_selected": int(M),
                "topk": int(topk),
                "top": top,
            },
        )

        np.save(
            os.path.join(
                out_dir,
                "scores.npy",
            ),
            scores,
        )

        print(
            f"[saved lambda={lam:g}] "
            f"{out_dir}",
            flush=True,
        )

    print(
        "\n(done) total elapsed="
        f"{format_seconds(time.perf_counter() - t0)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
