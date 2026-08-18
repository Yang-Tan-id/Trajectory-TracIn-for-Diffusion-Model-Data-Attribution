import argparse
import os
import glob
import json
import random
import time
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


def print_item_progress(phase: str, idx: int, total: int, extra: str = ""):
    msg = f"    [{phase}] {idx+1}/{total}"
    if extra:
        msg += f" | {extra}"
    print(msg, flush=True)


def should_print_item(idx: int, total: int, every: int = 100) -> bool:
    return idx == 0 or (idx + 1) % every == 0 or (idx + 1) == total


# ============================================================
# Helpers: checkpoint discovery
# ============================================================
def list_checkpoints_sorted(dir_path: str, pattern: str = "*.pt") -> List[str]:
    if not os.path.isdir(dir_path):
        return []
    paths = glob.glob(os.path.join(dir_path, pattern))
    paths.sort()
    return paths


def latest_checkpoint_in_dir(dir_path: str, pattern: str = "*.pt") -> Optional[str]:
    paths = list_checkpoints_sorted(dir_path, pattern)
    return paths[-1] if paths else None


# ============================================================
# Vocab / cond builder (for query labels)
# ============================================================
def vocab_to_index(vocab: Any) -> Dict[str, int]:
    if vocab is None:
        raise ValueError("Checkpoint/dataset has no vocab; cannot build cond from label strings.")
    if isinstance(vocab, dict):
        return {str(k): int(v) for k, v in vocab.items()}
    if isinstance(vocab, (list, tuple)):
        return {str(lbl): i for i, lbl in enumerate(vocab)}
    raise TypeError(f"Unsupported vocab type: {type(vocab)}")


def labels_to_cond(labels: List[str], vocab: Any, cond_dim: int, device: torch.device) -> torch.Tensor:
    m = vocab_to_index(vocab)
    cond = torch.zeros(cond_dim, device=device, dtype=torch.float32)
    missing = [lab for lab in labels if lab not in m]
    if missing:
        raise KeyError(f"Query labels not found in vocab: {missing[:10]}")
    for lab in labels:
        cond[m[lab]] = 1.0
    return cond.unsqueeze(0)  # [1, cond_dim]


# ============================================================
# Load checkpoint into a ready-to-run model
# ============================================================
def build_model_from_ckpt(
    ckpt_path: str, device: torch.device
) -> Tuple[nn.Module, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=str(device))
    need = ["model_state", "T", "cond_dim"]
    for k in need:
        if k not in ckpt:
            raise KeyError(f"Checkpoint missing {k}: {ckpt_path}")

    cond_dim = int(ckpt["cond_dim"])
    base_ch = int(ckpt.get("base_ch", 64))
    time_dim = int(ckpt.get("time_dim", 128))
    grid_size = int(ckpt.get("grid_size", 3))

    model = base.CondEpsModel(
        in_ch=3, cond_dim=cond_dim, base_ch=base_ch, time_dim=time_dim
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    meta = {
        "T": int(ckpt["T"]),
        "grid_size": grid_size,
        "cond_dim": cond_dim,
        "vocab": ckpt.get("vocab", None),
        "base_ch": base_ch,
        "time_dim": time_dim,
    }
    return model, meta


# ============================================================
# Active-parameter switching
# ============================================================
def set_active_params(model: nn.Module) -> List[nn.Parameter]:
    active = []
    for p in model.parameters():
        p.requires_grad_(True)
        active.append(p)
    return active


# ============================================================
# Trajectory query objective: squared epsilon deviation from theta_ref
# ============================================================
def trajectory_noise_deviation(
    model: nn.Module,
    xt_ref: torch.Tensor,   # [1,3,H,W]
    t: torch.Tensor,        # [1] long
    cond_q: torch.Tensor,   # [1,cond_dim]
    eps_ref: torch.Tensor,  # detached eps prediction from theta_ref
) -> torch.Tensor:
    """
    Per-snapshot objective:
        || eps_theta(xt_ref,t,cond_q) - eps_theta_ref(xt_ref,t,cond_q) ||_2^2
    """
    eps_pred = model(xt_ref, t, cond_q)
    return (eps_pred - eps_ref.detach()).pow(2).sum()


def grad_dot(g1, g2) -> torch.Tensor:
    s = None
    for a, b in zip(g1, g2):
        if a is None or b is None:
            continue
        v = (a * b).sum()
        s = v if s is None else (s + v)
    if s is None:
        return torch.tensor(0.0, device=g1[0].device if len(g1) > 0 and g1[0] is not None else "cpu")
    return s


# ============================================================
# Train loss at fixed t (Monte Carlo alignment with trajectory timestep)
# ============================================================
def train_loss_mc_at_t(
    model: nn.Module,
    sched,
    x0: torch.Tensor,         # [1,3,H,W]
    cond: torch.Tensor,       # [1,cond_dim]
    t: torch.Tensor,          # [1] long
    *,
    num_mc_samples: int = 8,
) -> torch.Tensor:
    """
    Monte Carlo estimate of:
        E_eps || eps_theta(x_t,t,cond) - eps ||^2
    at a fixed timestep t.
    """
    losses = []
    for _ in range(int(num_mc_samples)):
        noise = torch.randn_like(x0)
        xt = base.q_sample(x0, t, noise, sched)
        eps_pred = model(xt, t, cond)
        losses.append(F.mse_loss(eps_pred, noise, reduction="mean"))
    return torch.stack(losses).mean()


# ============================================================
# Reference trajectory: snapshots (key by snapshot index)
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
) -> Tuple[List[torch.Tensor], np.ndarray, List[int]]:
    """
    Returns:
      traj_use: saved snapshots, length K=num_keep
      t_seq:    diffusion timestep corresponding to each snapshot
      save_steps: step-index in DDIM loop corresponding to each snapshot
    """
    model.eval()
    steps = int(steps)
    num_keep = int(num_keep)
    if num_keep <= 0:
        raise ValueError("num_keep must be > 0")
    num_keep = min(num_keep, steps)

    save_steps = np.linspace(0, steps - 1, num_keep, dtype=np.int64).tolist()

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

    T = int(sched.T)
    ts = np.linspace(T - 1, 0, steps, dtype=np.int64)
    t_seq = np.array([int(ts[k]) for k in save_steps], dtype=np.int64)

    if len(traj_use) != len(t_seq):
        raise RuntimeError(f"traj_use len {len(traj_use)} != t_seq len {len(t_seq)} (unexpected)")

    return traj_use, t_seq, save_steps


def pick_snapshot_ids(K: int, num_pick: int) -> List[int]:
    K = int(K)
    num_pick = int(num_pick)
    if num_pick >= K:
        return list(range(K))
    return np.linspace(0, K - 1, num_pick, dtype=np.int64).tolist()


def build_xtref_dict_by_snapid(traj_use: List[torch.Tensor], snap_ids: List[int]) -> Dict[int, torch.Tensor]:
    out: Dict[int, torch.Tensor] = {}
    for i in snap_ids:
        out[int(i)] = traj_use[int(i)].detach()
    return out


@torch.no_grad()
def compute_target_eps_by_snapid(
    target_model: nn.Module,
    xt_ref_dict: Dict[int, torch.Tensor],
    query_cond: torch.Tensor,
    snap_ids: List[int],
    t_seq: np.ndarray,
) -> Dict[int, torch.Tensor]:
    """Cache stop-grad eps predictions for a query target checkpoint."""
    target_model.eval()
    out: Dict[int, torch.Tensor] = {}
    for sid in snap_ids:
        sid = int(sid)
        xt_ref = xt_ref_dict[sid]
        t = torch.tensor([int(t_seq[sid])], device=xt_ref.device, dtype=torch.long)
        out[sid] = target_model(xt_ref, t, query_cond).detach()
    return out


# ============================================================
# Compute query gradients for all snapshots (per checkpoint)
# ============================================================
def compute_g_traj(
    model: nn.Module,
    active_params: List[nn.Parameter],
    xt_ref_dict: Dict[int, torch.Tensor],   # key=snap_id
    target_eps_dict: Dict[int, torch.Tensor],  # stop-grad target prediction per snap_id
    query_cond: torch.Tensor,
    snap_ids: List[int],                    # list of snap_id
    t_seq: np.ndarray,                      # diffusion timestep per snap_id
) -> Dict[int, Tuple[torch.Tensor, ...]]:
    """
    g_q[snap_id] = grad_theta f(theta; x_ref(snap_id), t_seq[snap_id], cond_q)
    """
    model.eval()
    gq: Dict[int, Tuple[torch.Tensor, ...]] = {}

    total = len(snap_ids)
    for k, sid in enumerate(snap_ids):
        sid = int(sid)
        if should_print_item(k, total, every=10):
            print_item_progress("query grad", k, total, extra=f"snap_id={sid} t={int(t_seq[sid])}")

        xt_ref = xt_ref_dict[sid].detach()
        t_int = int(t_seq[sid])
        t = torch.tensor([t_int], device=xt_ref.device, dtype=torch.long)

        f = trajectory_noise_deviation(
            model=model,
            xt_ref=xt_ref,
            t=t,
            cond_q=query_cond,
            eps_ref=target_eps_dict[sid],
        )
        g = torch.autograd.grad(f, active_params, retain_graph=False, create_graph=False, allow_unused=True)
        gq[sid] = g

    return gq


def score_one_trainpoint_given_gtraj(
    model: nn.Module,
    active_params: List[nn.Parameter],
    sched,
    gq: Dict[int, Tuple[torch.Tensor, ...]],  # keyed by snap_id
    snap_ids: List[int],
    t_seq: np.ndarray,
    x0_train: torch.Tensor,
    train_cond: torch.Tensor,
    *,
    eta_k: float = 1.0,
    train_mc_samples: int = 8,
) -> torch.Tensor:
    """
    score(z) = avg_{snap_id} eta_k * < g_q[snap_id], grad_theta L_train(t_seq[snap_id]) >
    """
    model.train()
    w = 1.0 / float(len(snap_ids))
    total = 0.0

    for sid in snap_ids:
        sid = int(sid)
        t_int = int(t_seq[sid])
        t = torch.tensor([t_int], device=x0_train.device, dtype=torch.long)

        L_tr = train_loss_mc_at_t(
            model,
            sched,
            x0_train,
            train_cond,
            t,
            num_mc_samples=int(train_mc_samples),
        )
        g_tr = torch.autograd.grad(
            L_tr, active_params, retain_graph=False, create_graph=False, allow_unused=True
        )

        total = total + (eta_k * w * grad_dot(gq[sid], g_tr))

    return total.detach()


# ============================================================
# IO
# ============================================================
def load_index_list(path: str, col: str = "idx") -> List[int]:
    """
    Support formats:
      - CSV with header
      - Plain text, one integer per line
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    idxs = []
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    if len(lines) >= 2 and lines[0].lower() == col.lower():
        for ln in lines[1:]:
            idxs.append(int(ln.split(",")[0]))
        return idxs

    for ln in lines:
        idxs.append(int(ln.split(",")[0]))
    return idxs


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def target_mode_out_name(mode: str) -> str:
    mode = str(mode).strip().lower()
    if mode in {"ref", "reference"}:
        return "ref"
    if mode in {"next", "next_ckpt", "next_checkpoint"}:
        return "next"
    raise ValueError(f"Unsupported query target mode: {mode!r}. Expected 'ref' or 'next'.")


# ============================================================
# main
# ============================================================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    parser = argparse.ArgumentParser(
        description="Trajectory TracIn attribution over full model parameters (no LoRA)."
    )
    parser.add_argument("--model", default="model_109900")
    parser.add_argument(
        "--variant",
        default="y",
        help="Model variant under models_checkpoints/<variant>/<model>, e.g. b/by/r/rb/y/yr.",
    )
    parser.add_argument(
        "--target-modes",
        default="ref,next",
        help="Comma-separated query targets to run: ref,next.",
    )
    parser.add_argument("--max-train-points", type=int, default=2000)
    parser.add_argument("--topk", type=int, default=2000)
    parser.add_argument("--csv-path", default="generated_database/49_100000.csv")
    parser.add_argument("--grid-size", type=int, default=3)
    parser.add_argument("--ddim-steps", type=int, default=2000)
    parser.add_argument("--num-traj-t", type=int, default=50)
    parser.add_argument("--train-mc-samples", type=int, default=8)
    parser.add_argument("--checkpoint-limit", type=int, default=-1)
    parser.add_argument("--checkpoint-stride", type=int, default=1)
    parser.add_argument("--reference-ckpt", default=None)
    parser.add_argument("--out-root", default="tracein_traj_runs")
    args = parser.parse_args()

    MODEL = args.model
    MODEL_VARIANT = args.variant
    CUR_MODEL = f"{MODEL}_{MODEL_VARIANT}"
    MAX_TRAIN_POINTS = int(args.max_train_points)
    target_modes = [target_mode_out_name(x) for x in parse_csv_list(args.target_modes)]
    if not target_modes:
        raise ValueError("--target-modes must include at least one of: ref,next")

    CONFIG = dict(
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=808,
        csv_path=args.csv_path,
        grid_size=int(args.grid_size),

        model_root=f"models_checkpoints/{MODEL_VARIANT}/{MODEL}",
        checkpoint_dir=f"models_checkpoints/{MODEL_VARIANT}/{MODEL}/baseline",
        checkpoint_pattern="*.pt",
        checkpoint_limit=int(args.checkpoint_limit),
        checkpoint_stride=int(args.checkpoint_stride),

        # query
        query_labels=[
            "background_color_red",
            "background_color_blue",
            "background_color_yellow",
        ],
        ddim_steps=int(args.ddim_steps),
        eta=0.0,

        # trajectory subset
        num_traj_t=int(args.num_traj_t),

        # Monte Carlo expectation controls
        train_mc_samples=int(args.train_mc_samples),

        # scoring
        max_train_points=MAX_TRAIN_POINTS,
        topk=int(args.topk),

        # query target modes:
        #   ref  -> target eps comes from reference_ckpt
        #   next -> target eps comes from the next full-model checkpoint
        query_target_modes=target_modes,

        reference_ckpt=args.reference_ckpt,
        random_subset=True,

        use_two_index_lists=False,
        use_six_index_lists=True,
        idx_list_1="generated_database/RBY/subset/blue_A_idx.csv",
        idx_list_2="generated_database/RBY/subset/blue_B_idx.csv",
        idx_list_3="generated_database/RBY/subset/red_A_idx.csv",
        idx_list_4="generated_database/RBY/subset/red_B_idx.csv",
        idx_list_5="generated_database/RBY/subset/yellow_A_idx.csv",
        idx_list_6="generated_database/RBY/subset/yellow_B_idx.csv",
        idx_col_name="idx",

        out_root=os.path.join(args.out_root, f"{CUR_MODEL}_full_model_traj_attribution{MAX_TRAIN_POINTS}"),
    )

    t0 = time.perf_counter()
    device = torch.device(CONFIG["device"])
    set_seed(int(CONFIG["seed"]))

    print(f"[device] {device}", flush=True)
    if device.type == "cuda":
        print(f"[device] GPU count visible: {torch.cuda.device_count()}", flush=True)
        print(f"[device] GPU name: {torch.cuda.get_device_name(0)}", flush=True)

    # ---- dataset ----
    print("[setup] Loading dataset...", flush=True)
    if "grid_size" in ColorGridDataset.__init__.__code__.co_varnames:
        ds = ColorGridDataset(CONFIG["csv_path"], grid_size=CONFIG["grid_size"])
    else:
        ds = ColorGridDataset(CONFIG["csv_path"])

    x0_ex, cond_ex = ds[0][:2]
    C, H, W = x0_ex.shape
    cond_dim = int(cond_ex.numel())
    print(f"[setup] Dataset loaded: N={len(ds)} | C={C} H={H} W={W} cond_dim={cond_dim}", flush=True)

    # ---- reference ckpt ----
    print("[setup] Locating reference checkpoint...", flush=True)
    ref_ckpt = CONFIG["reference_ckpt"] or latest_checkpoint_in_dir(
        CONFIG["checkpoint_dir"], pattern=CONFIG["checkpoint_pattern"]
    )
    if ref_ckpt is None:
        raise FileNotFoundError("No reference full-model checkpoint found in checkpoint_dir.")

    print(f"[setup] Reference checkpoint: {os.path.basename(ref_ckpt)}", flush=True)
    ref_model, ref_meta = build_model_from_ckpt(ref_ckpt, device=device)
    T = int(ref_meta["T"])
    sched = base.make_linear_schedule(T, device=device)
    print(f"[setup] Diffusion steps T={T}", flush=True)

    # ---- query cond ----
    vocab = getattr(ds, "vocab", None) or ref_meta.get("vocab", None)
    if vocab is None:
        raise RuntimeError("No vocab found in dataset or ckpt; cannot build query cond from label strings.")
    query_cond = labels_to_cond(CONFIG["query_labels"], vocab, cond_dim, device=device)

    # ---- reference trajectory snapshots ----
    print("[setup] Computing reference trajectory...", flush=True)
    traj_use, t_seq, save_steps = compute_reference_trajectory(
        model=ref_model,
        sched=sched,
        cond=query_cond,
        shape=(1, C, H, W),
        seed=int(CONFIG["seed"]),
        steps=int(CONFIG["ddim_steps"]),
        eta=float(CONFIG["eta"]),
        device=device,
        num_keep=int(CONFIG["num_traj_t"]),
    )

    K = len(traj_use)
    snap_ids = pick_snapshot_ids(K, num_pick=int(CONFIG["num_traj_t"]))
    xt_ref_dict = build_xtref_dict_by_snapid(traj_use, snap_ids)
    ref_target_eps_dict = compute_target_eps_by_snapid(
        target_model=ref_model,
        xt_ref_dict=xt_ref_dict,
        query_cond=query_cond,
        snap_ids=snap_ids,
        t_seq=t_seq,
    )
    print(f"[setup] Reference trajectory ready: K={K} | selected_snapshots={len(snap_ids)}", flush=True)

    # ---- collect checkpoints ----
    ckpts = list_checkpoints_sorted(CONFIG["checkpoint_dir"], pattern=CONFIG["checkpoint_pattern"])
    stride = max(1, int(CONFIG.get("checkpoint_stride", 1)))
    if stride > 1:
        ckpts = ckpts[::stride]
    limit = int(CONFIG.get("checkpoint_limit", -1))
    if limit > 0:
        ckpts = ckpts[:limit]
    if not ckpts:
        raise FileNotFoundError(f"No full-model checkpoints found in {CONFIG['checkpoint_dir']}")
    print(
        f"[setup] full_model_ckpts={len(ckpts)} | target_modes={','.join(CONFIG['query_target_modes'])}",
        flush=True,
    )

    # ---- choose training points to score ----
    print("[setup] Building candidate set...", flush=True)
    use_six_lists = bool(CONFIG.get("use_six_index_lists", False))
    use_three_lists = bool(CONFIG.get("use_three_index_lists", False)) and (not use_six_lists)
    use_two_lists = bool(CONFIG.get("use_two_index_lists", False)) and (not use_three_lists) and (not use_six_lists)

    N = len(ds)
    M_req = min(int(CONFIG["max_train_points"]), N)

    if (not use_two_lists) and (not use_three_lists) and (not use_six_lists):
        if CONFIG.get("random_subset", False):
            score_items = [(0, i) for i in random.sample(range(N), k=M_req)]
        else:
            score_items = [(0, i) for i in range(M_req)]

    elif use_two_lists:
        idxs1 = load_index_list(CONFIG["idx_list_1"], col=CONFIG.get("idx_col_name", "idx"))
        idxs2 = load_index_list(CONFIG["idx_list_2"], col=CONFIG.get("idx_col_name", "idx"))

        idxs1 = sorted(set(int(i) for i in idxs1 if 0 <= int(i) < N))
        idxs2 = sorted(set(int(i) for i in idxs2 if 0 <= int(i) < N))

        M1 = min(M_req // 2, len(idxs1))
        M2 = min(M_req - M1, len(idxs2))

        if CONFIG.get("random_subset", False):
            pick1 = random.sample(idxs1, k=M1) if M1 > 0 else []
            pick2 = random.sample(idxs2, k=M2) if M2 > 0 else []
        else:
            pick1 = idxs1[:M1]
            pick2 = idxs2[:M2]

        score_items = [(1, int(i)) for i in pick1] + [(2, int(i)) for i in pick2]
        random.shuffle(score_items)

    elif use_three_lists:
        idxs1 = load_index_list(CONFIG["idx_list_1"], col=CONFIG.get("idx_col_name", "idx"))
        idxs2 = load_index_list(CONFIG["idx_list_2"], col=CONFIG.get("idx_col_name", "idx"))
        idxs3 = load_index_list(CONFIG["idx_list_3"], col=CONFIG.get("idx_col_name", "idx"))

        idxs1 = sorted(set(int(i) for i in idxs1 if 0 <= int(i) < N))
        idxs2 = sorted(set(int(i) for i in idxs2 if 0 <= int(i) < N))
        idxs3 = sorted(set(int(i) for i in idxs3 if 0 <= int(i) < N))

        split_base = M_req // 3
        M1 = min(split_base, len(idxs1))
        M2 = min(split_base, len(idxs2))
        M3 = min(M_req - M1 - M2, len(idxs3))

        leftover = M_req - (M1 + M2 + M3)
        if leftover > 0:
            cap1 = len(idxs1) - M1
            cap2 = len(idxs2) - M2
            cap3 = len(idxs3) - M3
            add1 = min(leftover, max(0, cap1))
            M1 += add1
            leftover -= add1
            add2 = min(leftover, max(0, cap2))
            M2 += add2
            leftover -= add2
            add3 = min(leftover, max(0, cap3))
            M3 += add3
            leftover -= add3

        if CONFIG.get("random_subset", False):
            pick1 = random.sample(idxs1, k=M1) if M1 > 0 else []
            pick2 = random.sample(idxs2, k=M2) if M2 > 0 else []
            pick3 = random.sample(idxs3, k=M3) if M3 > 0 else []
        else:
            pick1 = idxs1[:M1]
            pick2 = idxs2[:M2]
            pick3 = idxs3[:M3]

        score_items = (
            [(1, int(i)) for i in pick1] +
            [(2, int(i)) for i in pick2] +
            [(3, int(i)) for i in pick3]
        )
        random.shuffle(score_items)

    else:
        idxs = []
        for k in range(1, 7):
            key = f"idx_list_{k}"
            idx_path = CONFIG.get(key, None)
            if idx_path is None:
                raise KeyError(f"Missing CONFIG[{key}] for six-index mode.")
            lst = load_index_list(idx_path, col=CONFIG.get("idx_col_name", "idx"))
            lst = sorted(set(int(i) for i in lst if 0 <= int(i) < N))
            idxs.append(lst)

        split_base = M_req // 6
        Ms = [min(split_base, len(idxs[k])) for k in range(6)]
        remainder = M_req - sum(Ms)

        if remainder > 0:
            while remainder > 0:
                progressed = False
                for k in range(6):
                    cap = len(idxs[k]) - Ms[k]
                    if cap > 0 and remainder > 0:
                        Ms[k] += 1
                        remainder -= 1
                        progressed = True
                if not progressed:
                    break

        picks = []
        for k in range(6):
            Mk = Ms[k]
            if Mk <= 0:
                picks.append([])
                continue
            if CONFIG.get("random_subset", False):
                picks.append(random.sample(idxs[k], k=Mk))
            else:
                picks.append(idxs[k][:Mk])

        score_items = []
        for k in range(6):
            src_id = k + 1
            score_items.extend([(src_id, int(i)) for i in picks[k]])

        random.shuffle(score_items)

    M_eff = len(score_items)
    if M_eff == 0:
        raise RuntimeError("No training points selected ...")
    scores = torch.zeros(M_eff, dtype=torch.float64)
    print(f"[candidate-set] N={N} | M_selected={M_eff}", flush=True)

    def get_one(src: int, i: int):
        x0, cond = ds[i][:2]
        x0 = x0.unsqueeze(0).to(device)
        cond = cond.unsqueeze(0).to(device) if cond.dim() == 1 else cond.to(device)
        return x0, cond

    score_indices_payload = {
        "N_eff": int(M_eff),
        "items": [{"src": int(s), "idx": int(i)} for (s, i) in score_items],
        "idx_list_1": CONFIG.get("idx_list_1", None),
        "idx_list_2": CONFIG.get("idx_list_2", None),
        "idx_list_3": CONFIG.get("idx_list_3", None),
        "idx_list_4": CONFIG.get("idx_list_4", None),
        "idx_list_5": CONFIG.get("idx_list_5", None),
        "idx_list_6": CONFIG.get("idx_list_6", None),
        "idx_col_name": CONFIG.get("idx_col_name", "idx"),
        "use_six_index_lists": bool(CONFIG.get("use_six_index_lists", False)),
    }

    base_run_info = {
        "ref_ckpt": ref_ckpt,
        "T": int(T),
        "ddim_steps": int(CONFIG["ddim_steps"]),
        "num_snap_used": int(len(snap_ids)),
        "snap_ids": [int(x) for x in snap_ids],
        "save_steps": [int(x) for x in save_steps],
        "t_seq": [int(x) for x in t_seq.tolist()],
        "M_requested": int(M_req),
        "M_eff": int(M_eff),
        "device": str(device),
        "seed": int(CONFIG["seed"]),
        "time_started": float(t0),
        "train_mc_samples": int(CONFIG["train_mc_samples"]),
        "parameter_scope": "full_model",
        "lora": False,
        "trajectory_source": {
            "name": "reference_checkpoint_ddim_trajectory",
            "checkpoint": ref_ckpt,
        },
        "checkpoint_reduction": "sum",
        "snapshot_reduction": "mean",
        "candidate_set": score_indices_payload,
    }

    for target_mode in CONFIG["query_target_modes"]:
        mode_t0 = time.perf_counter()
        mode = target_mode_out_name(target_mode)
        mode_out_dir = os.path.join(CONFIG["out_root"], mode)
        scores = torch.zeros(M_eff, dtype=torch.float64)
        used_ckpts: List[Dict[str, Any]] = []
        skipped_ckpts: List[Dict[str, Any]] = []

        print(f"\n[target mode] {mode} | output={mode_out_dir}", flush=True)

        for ckpt_idx, ckpt_path in enumerate(ckpts):
            ckpt_t0 = time.perf_counter()
            target_ckpt = ref_ckpt
            target_eps_dict = ref_target_eps_dict

            if mode == "next":
                if ckpt_idx + 1 >= len(ckpts):
                    skipped_ckpts.append({
                        "checkpoint": ckpt_path,
                        "reason": "no next-checkpoint query target",
                    })
                    print(
                        f"[next checkpoint] skipping {os.path.basename(ckpt_path)}: no next checkpoint",
                        flush=True,
                    )
                    continue
                target_ckpt = ckpts[ckpt_idx + 1]
                target_model, _ = build_model_from_ckpt(target_ckpt, device=device)
                target_eps_dict = compute_target_eps_by_snapid(
                    target_model=target_model,
                    xt_ref_dict=xt_ref_dict,
                    query_cond=query_cond,
                    snap_ids=snap_ids,
                    t_seq=t_seq,
                )
                del target_model

            print(
                f"\n[full checkpoint] {ckpt_idx+1}/{len(ckpts)} | "
                f"{os.path.basename(ckpt_path)} | target={os.path.basename(target_ckpt)}",
                flush=True,
            )

            model_k, _ = build_model_from_ckpt(ckpt_path, device=device)
            active = set_active_params(model_k)
            eta_k = 1.0

            if mode == "ref" and os.path.abspath(ckpt_path) == os.path.abspath(ref_ckpt):
                print(
                    "[warning] checkpoint equals theta_ref; query gradient is exactly zero.",
                    flush=True,
                )

            print("[full-model] computing trajectory query gradients...", flush=True)
            gq = compute_g_traj(
                model=model_k,
                active_params=active,
                xt_ref_dict=xt_ref_dict,
                target_eps_dict=target_eps_dict,
                query_cond=query_cond,
                snap_ids=snap_ids,
                t_seq=t_seq,
            )

            for j, (src, idx) in enumerate(score_items):
                if should_print_item(j, M_eff):
                    print_item_progress(
                        f"{mode} score",
                        j,
                        M_eff,
                        extra=f"ckpt={ckpt_idx+1}/{len(ckpts)}",
                    )

                x0_train, cond_train = get_one(src, idx)
                sc = score_one_trainpoint_given_gtraj(
                    model=model_k,
                    active_params=active,
                    sched=sched,
                    gq=gq,
                    snap_ids=snap_ids,
                    t_seq=t_seq,
                    x0_train=x0_train,
                    train_cond=cond_train,
                    eta_k=eta_k,
                    train_mc_samples=int(CONFIG["train_mc_samples"]),
                )
                scores[j] += float(sc.item())

            used_ckpts.append({
                "checkpoint": ckpt_path,
                "query_target_checkpoint": target_ckpt,
            })
            print(
                f"[full-model] done: {os.path.basename(ckpt_path)} |snap|={len(snap_ids)} "
                f"| elapsed={format_seconds(time.perf_counter() - ckpt_t0)}",
                flush=True,
            )
            del model_k

        topk = min(int(CONFIG["topk"]), M_eff)
        vals, ord_idx = torch.topk(scores, k=topk, largest=True)

        top = []
        for r in range(topk):
            j = int(ord_idx[r].item())
            src, train_idx = score_items[j]
            src = int(src)
            train_idx = int(train_idx)
            idx_tag = f"{train_idx}_{src}" if src != 0 else str(train_idx)

            top.append({
                "idx": train_idx,
                "src": src,
                "idx_tag": idx_tag,
                "score": float(vals[r].item()),
            })

        run_info = dict(base_run_info)
        run_info["elapsed_sec"] = float(time.perf_counter() - mode_t0)
        run_info["query_objective"] = {
            "name": (
                "trajectory_noise_squared_deviation"
                if mode == "ref"
                else "trajectory_next_checkpoint_noise_mse"
            ),
            "target_mode": mode,
            "formula": (
                "mean_k ||eps_theta(x_ref_k,k)-eps_theta_ref(x_ref_k,k)||_2^2"
                if mode == "ref"
                else "mean_k mean((eps_theta_c(x_ref_k,k)-stopgrad(eps_theta_c_plus_1(x_ref_k,k)))^2)"
            ),
            "reference_ckpt": ref_ckpt,
        }
        run_info["used_checkpoints"] = used_ckpts
        run_info["skipped_checkpoints"] = skipped_ckpts

        print(f"[save] Writing {mode} outputs...", flush=True)
        ensure_dir(mode_out_dir)
        mode_config = dict(CONFIG)
        mode_config["out_dir"] = mode_out_dir
        mode_config["query_target_mode"] = mode

        save_json(os.path.join(mode_out_dir, "run_config.json"), mode_config)
        save_json(os.path.join(mode_out_dir, "run_info.json"), run_info)
        save_json(os.path.join(mode_out_dir, "score_indices.json"), score_indices_payload)
        save_json(
            os.path.join(mode_out_dir, "result_topk.json"),
            {"N_eff": int(M_eff), "topk": int(topk), "top": top},
        )
        np.save(os.path.join(mode_out_dir, "scores.npy"), scores.cpu().numpy())

        print(f"[saved] {mode_out_dir}/run_config.json", flush=True)
        print(f"[saved] {mode_out_dir}/run_info.json", flush=True)
        print(f"[saved] {mode_out_dir}/score_indices.json", flush=True)
        print(f"[saved] {mode_out_dir}/result_topk.json", flush=True)
        print(f"[saved] {mode_out_dir}/scores.npy", flush=True)

    dt = time.perf_counter() - t0
    print(f"\n(done) elapsed={format_seconds(dt)}", flush=True)


if __name__ == "__main__":
    main()
