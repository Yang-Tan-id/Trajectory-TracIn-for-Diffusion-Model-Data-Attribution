# endpoint_tracein_main.py
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
# Small progress helpers
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
# LoRA wrapper (same as your training file)
# ============================================================
class LoRAConv2d(nn.Module):
    def __init__(self, base_conv: nn.Conv2d, r: int = 4, alpha: float = 1.0):
        super().__init__()
        if not isinstance(base_conv, nn.Conv2d):
            raise TypeError("LoRAConv2d expects nn.Conv2d")
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")

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

        self.lora_down = nn.Conv2d(in_ch, self.r, kernel_size=1, bias=False)
        self.lora_up = nn.Conv2d(
            self.r, out_ch, kernel_size=k, stride=s, padding=p, dilation=d, groups=g, bias=False
        )

        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.base(x) + self.scale * self.lora_up(self.lora_down(x))


def inject_lora_into_selected_convs(model: nn.Module, r: int, alpha: float, target_names):
    target_names = set(target_names) if target_names is not None else {"out_conv"}
    for full_name, module in list(model.named_modules()):
        if full_name in target_names and isinstance(module, nn.Conv2d):
            parent = model
            parts = full_name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], LoRAConv2d(module, r=r, alpha=alpha))


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
# Load baseline OR LoRA-only ckpt into a ready-to-run model
# ============================================================
def build_model_from_baseline_ckpt(
    baseline_ckpt_path: str, device: torch.device
) -> Tuple[nn.Module, Dict[str, Any]]:
    ckpt = torch.load(baseline_ckpt_path, map_location=str(device))
    need = ["model_state", "T", "cond_dim"]
    for k in need:
        if k not in ckpt:
            raise KeyError(f"Baseline ckpt missing {k}: {baseline_ckpt_path}")

    cond_dim = int(ckpt["cond_dim"])
    base_ch = int(ckpt.get("base_ch", 64))
    time_dim = int(ckpt.get("time_dim", 128))
    grid_size = int(ckpt.get("grid_size", 3))

    model = base.CondEpsModel(in_ch=3, cond_dim=cond_dim, base_ch=base_ch, time_dim=time_dim).to(device)
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


def build_model_from_lora_ckpt(
    lora_ckpt_path: str, device: torch.device
) -> Tuple[nn.Module, Dict[str, Any]]:
    payload = torch.load(lora_ckpt_path, map_location="cpu")
    lora_sd = payload.get("lora_state", payload)

    baseline_ckpt_path = payload.get("baseline_ckpt", None)
    if baseline_ckpt_path is None:
        raise KeyError(f"LoRA ckpt missing baseline_ckpt: {lora_ckpt_path}")

    base_model, meta = build_model_from_baseline_ckpt(baseline_ckpt_path, device=device)

    r = int(payload.get("lora_r", 4))
    alpha = float(payload.get("lora_alpha", 4.0))
    targets = payload.get("lora_targets", ["out_conv"])

    inject_lora_into_selected_convs(base_model, r=r, alpha=alpha, target_names=targets)
    base_model.load_state_dict(lora_sd, strict=False)
    base_model.eval()

    meta = dict(meta)
    meta.update({"lora_r": r, "lora_alpha": alpha, "lora_targets": list(targets)})
    return base_model, meta


# ============================================================
# Active-parameter switching
# ============================================================
def set_active_params_baseline(model: nn.Module) -> List[nn.Parameter]:
    """
    baseline checkpoint: differentiate non-LoRA params; freeze LoRA params if present
    """
    active = []
    for name, p in model.named_parameters():
        if ".lora_" in name:
            p.requires_grad_(False)
        else:
            p.requires_grad_(True)
            active.append(p)
    return active


def set_active_params_lora(model: nn.Module) -> List[nn.Parameter]:
    """
    LoRA checkpoint: differentiate only LoRA params; freeze base params
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
# Loss definitions (Monte Carlo estimates of expectations)
# ============================================================
def endpoint_anchored_loss_mc(
    model: nn.Module,
    sched,
    x0_ref: torch.Tensor,      # [B,3,H,W], detached target
    cond: torch.Tensor,        # [B,cond_dim]
    *,
    t_min: int = 0,
    t_max: Optional[int] = None,
    num_mc_samples: int = 8,
) -> torch.Tensor:
    """
    Monte Carlo estimate of:
        E_{t,eps} || eps_theta(x_t_ref,t,y) - eps ||^2
    where x_t_ref is forward-noised from x0_ref via q_sample.
    """
    device = x0_ref.device
    B = x0_ref.shape[0]

    if t_max is None:
        t_max = sched.T - 1
    t_min = max(0, min(sched.T - 1, int(t_min)))
    t_max = max(0, min(sched.T - 1, int(t_max)))
    if t_max < t_min:
        t_max = t_min

    losses = []
    for _ in range(int(num_mc_samples)):
        t = torch.randint(low=t_min, high=t_max + 1, size=(B,), device=device, dtype=torch.long)
        noise = torch.randn_like(x0_ref)
        xt_ref = base.q_sample(x0_ref, t, noise, sched)
        eps_pred = model(xt_ref, t, cond)
        losses.append(F.mse_loss(eps_pred, noise, reduction="mean"))

    return torch.stack(losses).mean()


def train_loss_mc(
    model: nn.Module,
    sched,
    x0: torch.Tensor,          # [1,3,H,W]
    cond: torch.Tensor,        # [1,cond_dim]
    *,
    num_mc_samples: int = 8,
) -> torch.Tensor:
    """
    Monte Carlo estimate of:
        E_{t,eps} || eps_theta(x_t,t,y) - eps ||^2
    for a single training point.
    """
    device = x0.device

    losses = []
    for _ in range(int(num_mc_samples)):
        t = torch.randint(0, sched.T, (1,), device=device, dtype=torch.long)
        noise = torch.randn_like(x0)
        xt = base.q_sample(x0, t, noise, sched)
        eps_pred = model(xt, t, cond)
        losses.append(F.mse_loss(eps_pred, noise, reduction="mean"))

    return torch.stack(losses).mean()


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


def compute_g_end(
    model,
    active_params,
    sched,
    x0_ref,
    query_cond,
    t_min_end,
    t_max_end,
    endpoint_mc_samples,
):
    model.train()
    L_end = endpoint_anchored_loss_mc(
        model=model,
        sched=sched,
        x0_ref=x0_ref.detach(),
        cond=query_cond,
        t_min=t_min_end,
        t_max=t_max_end,
        num_mc_samples=endpoint_mc_samples,
    )
    g_end = torch.autograd.grad(
        L_end, active_params, retain_graph=False, create_graph=False, allow_unused=True
    )
    return g_end, L_end.detach()


def score_one_trainpoint_given_gend(
    model,
    active_params,
    sched,
    g_end,
    x0_train,
    train_cond,
    eta_k=1.0,
    train_mc_samples=8,
):
    model.train()
    L_tr = train_loss_mc(
        model=model,
        sched=sched,
        x0=x0_train,
        cond=train_cond,
        num_mc_samples=train_mc_samples,
    )
    g_tr = torch.autograd.grad(
        L_tr, active_params, retain_graph=False, create_graph=False, allow_unused=True
    )

    sc = eta_k * grad_dot(g_end, g_tr)
    return sc.detach(), L_tr.detach()


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
    """
    Use your base.ddim_sample(...) which returns a trajectory list; endpoint = last element.
    """
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
    x0_ref = traj[-1]
    return x0_ref.detach()


# ============================================================
# IO helpers
# ============================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_index_list(path: str, col: str = "idx") -> List[int]:
    """
    Support formats:
      1) CSV with header: idx\\n5110\\n26558\\n...
      2) Plain text: one integer per line
      3) CSV without header: 5110\\n26558\\n...
    We always take the first field before comma.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    with open(path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    if not lines:
        return []

    if lines[0].lower() == col.lower():
        lines = lines[1:]

    idxs = []
    for ln in lines:
        idxs.append(int(ln.split(",")[0]))
    return idxs


def infer_run_tag(use_baseline_ckpts: bool, use_lora_ckpts: bool) -> str:
    if use_baseline_ckpts and use_lora_ckpts:
        return "combine"
    if use_baseline_ckpts and (not use_lora_ckpts):
        return "baseline"
    if (not use_baseline_ckpts) and use_lora_ckpts:
        return "lora"
    raise ValueError("Invalid setting: both use_baseline_ckpts and use_lora_ckpts are False.")


def main():
    # -----------------------------
    # user-friendly path knobs
    # -----------------------------
    MODEL = "model_109900"
    LORA = "by"
    CUR_MODEL = f"{MODEL}_{LORA}"
    MAX_TRAIN_POINTS = 2000

    use_baseline = True
    use_lora = False

    run_tag = infer_run_tag(use_baseline, use_lora)

    CONFIG = dict(
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=808,
        csv_path="generated_database/49_100000.csv",
        grid_size=3,

        # dirs
        model_root=f"models_checkpoints/{LORA}/{MODEL}",
        baseline_dir=f"models_checkpoints/{LORA}/{MODEL}/baseline",
        lora_update_dir=f"models_checkpoints/{LORA}/{MODEL}/{LORA}",

        # query
        query_labels=[
            "background_color_red",
            "background_color_blue",
            "background_color_yellow",
        ],
        ddim_steps=2000,
        eta=0.0,

        # endpoint-anchored loss focus
        t_min_end=0,
        t_max_end_frac=0.2,

        # Monte Carlo expectation controls
        endpoint_mc_samples=8,
        train_mc_samples=8,

        # scoring
        max_train_points=MAX_TRAIN_POINTS,
        topk=2000,

        # which checkpoints to use
        use_baseline_ckpts=use_baseline,
        use_lora_ckpts=use_lora,

        # reference checkpoint (None -> latest baseline)
        reference_ckpt=None,
        random_subset=True,

        # index-lists mode
        use_two_index_lists=False,
        use_six_index_lists=True,
        idx_list_1="generated_database/RBY/subset/blue_A_idx.csv",
        idx_list_2="generated_database/RBY/subset/blue_B_idx.csv",
        idx_list_3="generated_database/RBY/subset/red_A_idx.csv",
        idx_list_4="generated_database/RBY/subset/red_B_idx.csv",
        idx_list_5="generated_database/RBY/subset/yellow_A_idx.csv",
        idx_list_6="generated_database/RBY/subset/yellow_B_idx.csv",
        idx_col_name="idx",

        # output
        out_dir=f"tracein_end_runs/{CUR_MODEL}_endpoint_loss{MAX_TRAIN_POINTS}/{run_tag}",
    )

    # -----------------------------
    # run
    # -----------------------------
    t0 = time.perf_counter()
    device = torch.device(CONFIG["device"])
    set_seed(int(CONFIG["seed"]))

    print(f"[device] Using device: {device}", flush=True)
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

    print(f"[setup] Dataset loaded: N={len(ds)}", flush=True)
    print(f"[setup] Example shape: C={C}, H={H}, W={W}, cond_dim={cond_dim}", flush=True)

    # ---- choose reference ckpt ----
    print("[setup] Locating reference checkpoint...", flush=True)
    ref_ckpt = CONFIG["reference_ckpt"]
    if ref_ckpt is None:
        ref_ckpt = latest_checkpoint_in_dir(CONFIG["baseline_dir"])
    if ref_ckpt is None:
        raise FileNotFoundError("No reference baseline checkpoint found in baseline_dir.")

    print(f"[setup] Reference checkpoint: {os.path.basename(ref_ckpt)}", flush=True)
    ref_model, ref_meta = build_model_from_baseline_ckpt(ref_ckpt, device=device)
    T = int(ref_meta["T"])
    sched = base.make_linear_schedule(T, device=device)

    # ---- query cond ----
    vocab = getattr(ds, "vocab", None) or ref_meta.get("vocab", None)
    if vocab is None:
        raise RuntimeError("No vocab found in dataset or ckpt; cannot build query cond from label strings.")
    query_cond = labels_to_cond(CONFIG["query_labels"], vocab, cond_dim, device=device)

    # ---- reference endpoint x0_ref ----
    print("[setup] Computing reference endpoint...", flush=True)
    x0_ref = compute_reference_endpoint(
        model=ref_model,
        sched=sched,
        cond=query_cond,
        shape=(1, C, H, W),
        seed=int(CONFIG["seed"]),
        steps=int(CONFIG["ddim_steps"]),
        eta=float(CONFIG["eta"]),
        device=device,
    )
    print("[setup] Reference endpoint ready.", flush=True)

    # ---- collect checkpoints ----
    baseline_ckpts = list_checkpoints_sorted(CONFIG["baseline_dir"]) if CONFIG["use_baseline_ckpts"] else []
    lora_ckpts = list_checkpoints_sorted(CONFIG["lora_update_dir"]) if CONFIG["use_lora_ckpts"] else []

    print(f"[setup] baseline_ckpts={len(baseline_ckpts)} | lora_ckpts={len(lora_ckpts)}", flush=True)

    # ---- choose training points to score ----
    print("[setup] Building candidate set...", flush=True)
    N = len(ds)
    M_req = min(int(CONFIG["max_train_points"]), N)

    use_six_lists = bool(CONFIG.get("use_six_index_lists", False))
    use_two_lists = bool(CONFIG.get("use_two_index_lists", False)) and (not use_six_lists)

    if (not use_two_lists) and (not use_six_lists):
        if CONFIG.get("random_subset", False):
            picked = random.sample(range(N), k=M_req)
        else:
            picked = list(range(M_req))
        score_items = [(0, int(i)) for i in picked]

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

    else:
        idxs = []
        for k in range(1, 7):
            key = f"idx_list_{k}"
            pth = CONFIG.get(key, None)
            if pth is None:
                raise KeyError(f"Missing CONFIG[{key}] for six-index mode.")
            lst = load_index_list(pth, col=CONFIG.get("idx_col_name", "idx"))
            lst = sorted(set(int(i) for i in lst if 0 <= int(i) < N))
            idxs.append(lst)

        split_base = M_req // 6
        Ms = [min(split_base, len(idxs[k])) for k in range(6)]
        remainder = M_req - sum(Ms)

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

    M = len(score_items)
    if M == 0:
        raise RuntimeError("No training points selected for scoring (M=0). Check idx_list files / N / filters.")

    print(f"[candidate-set] N={N} | M_selected={M}", flush=True)

    # ---- scoring accumulator ----
    scores = torch.zeros(M, dtype=torch.float64)

    t_max_end = int(float(CONFIG["t_max_end_frac"]) * T)
    t_max_end = max(0, min(T - 1, t_max_end))

    run_info = {
        "ref_ckpt": ref_ckpt,
        "T": int(T),
        "ddim_steps": int(CONFIG["ddim_steps"]),
        "M_scored": int(M),
        "device": str(device),
        "seed": int(CONFIG["seed"]),
        "time_started": float(t0),
        "use_six_index_lists": bool(CONFIG.get("use_six_index_lists", False)),
        "endpoint_mc_samples": int(CONFIG["endpoint_mc_samples"]),
        "train_mc_samples": int(CONFIG["train_mc_samples"]),
    }

    def get_one(src: int, i: int):
        x0, cond = ds[i][:2]
        x0 = x0.unsqueeze(0).to(device)
        cond = cond.unsqueeze(0).to(device) if cond.dim() == 1 else cond.to(device)
        return x0, cond

    # ---- baseline ckpts ----
    for ckpt_idx, ckpt_path in enumerate(baseline_ckpts):
        ckpt_t0 = time.perf_counter()
        print(f"\n[baseline checkpoint] {ckpt_idx+1}/{len(baseline_ckpts)} | {os.path.basename(ckpt_path)}", flush=True)

        model_k, _ = build_model_from_baseline_ckpt(ckpt_path, device=device)
        active = set_active_params_baseline(model_k)
        eta_k = 1.0

        print("[baseline] computing g_end (Monte Carlo expectation)...", flush=True)
        g_end, Lend0 = compute_g_end(
            model=model_k,
            active_params=active,
            sched=sched,
            x0_ref=x0_ref,
            query_cond=query_cond,
            t_min_end=int(CONFIG["t_min_end"]),
            t_max_end=t_max_end,
            endpoint_mc_samples=int(CONFIG["endpoint_mc_samples"]),
        )

        for j, (src, idx) in enumerate(score_items):
            if should_print_item(j, M):
                print_item_progress(
                    "baseline score",
                    j,
                    M,
                    extra=f"ckpt={ckpt_idx+1}/{len(baseline_ckpts)}",
                )

            x0_train, cond_train = get_one(src, idx)
            sc, _ = score_one_trainpoint_given_gend(
                model=model_k,
                active_params=active,
                sched=sched,
                g_end=g_end,
                x0_train=x0_train,
                train_cond=cond_train,
                eta_k=eta_k,
                train_mc_samples=int(CONFIG["train_mc_samples"]),
            )
            scores[j] += float(sc.item())

        print(
            f"[baseline] done: {os.path.basename(ckpt_path)} | "
            f"L_end_mc={float(Lend0):.6f} | elapsed={format_seconds(time.perf_counter() - ckpt_t0)}",
            flush=True,
        )

    # ---- LoRA ckpts ----
    for ckpt_idx, ckpt_path in enumerate(lora_ckpts):
        ckpt_t0 = time.perf_counter()
        print(f"\n[lora checkpoint] {ckpt_idx+1}/{len(lora_ckpts)} | {os.path.basename(ckpt_path)}", flush=True)

        model_k, _ = build_model_from_lora_ckpt(ckpt_path, device=device)
        active = set_active_params_lora(model_k)
        eta_k = 1.0

        print("[lora] computing g_end (Monte Carlo expectation)...", flush=True)
        g_end, Lend0 = compute_g_end(
            model=model_k,
            active_params=active,
            sched=sched,
            x0_ref=x0_ref,
            query_cond=query_cond,
            t_min_end=int(CONFIG["t_min_end"]),
            t_max_end=t_max_end,
            endpoint_mc_samples=int(CONFIG["endpoint_mc_samples"]),
        )

        for j, (src, idx) in enumerate(score_items):
            if should_print_item(j, M):
                print_item_progress(
                    "lora score",
                    j,
                    M,
                    extra=f"ckpt={ckpt_idx+1}/{len(lora_ckpts)}",
                )

            x0_train, cond_train = get_one(src, idx)
            sc, _ = score_one_trainpoint_given_gend(
                model=model_k,
                active_params=active,
                sched=sched,
                g_end=g_end,
                x0_train=x0_train,
                train_cond=cond_train,
                eta_k=eta_k,
                train_mc_samples=int(CONFIG["train_mc_samples"]),
            )
            scores[j] += float(sc.item())

        print(
            f"[lora] done: {os.path.basename(ckpt_path)} | "
            f"L_end_mc={float(Lend0):.6f} | elapsed={format_seconds(time.perf_counter() - ckpt_t0)}",
            flush=True,
        )

    # ---- top-k ----
    topk = min(int(CONFIG["topk"]), M)
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

    # -----------------------------
    # SAVE OUTPUTS
    # -----------------------------
    print("[save] Writing outputs...", flush=True)
    ensure_dir(CONFIG["out_dir"])

    run_info["elapsed_sec"] = float(time.perf_counter() - t0)

    save_json(os.path.join(CONFIG["out_dir"], "run_config.json"), CONFIG)
    save_json(os.path.join(CONFIG["out_dir"], "run_info.json"), run_info)

    save_json(
        os.path.join(CONFIG["out_dir"], "score_indices.json"),
        {
            "N_eff": int(M),
            "items": [{"src": int(s), "idx": int(i)} for (s, i) in score_items],
            "idx_list_1": CONFIG.get("idx_list_1", None),
            "idx_list_2": CONFIG.get("idx_list_2", None),
            "idx_list_3": CONFIG.get("idx_list_3", None),
            "idx_list_4": CONFIG.get("idx_list_4", None),
            "idx_list_5": CONFIG.get("idx_list_5", None),
            "idx_list_6": CONFIG.get("idx_list_6", None),
            "idx_col_name": CONFIG.get("idx_col_name", "idx"),
            "use_six_index_lists": bool(CONFIG.get("use_six_index_lists", False)),
        },
    )

    payload = {
        "N_eff": int(M),
        "topk": int(topk),
        "top": top,
    }
    save_json(os.path.join(CONFIG["out_dir"], "result_topk.json"), payload)
    np.save(os.path.join(CONFIG["out_dir"], "scores.npy"), scores.cpu().numpy())

    print(f"\n[saved] {CONFIG['out_dir']}/run_config.json", flush=True)
    print(f"[saved] {CONFIG['out_dir']}/run_info.json", flush=True)
    print(f"[saved] {CONFIG['out_dir']}/score_indices.json", flush=True)
    print(f"[saved] {CONFIG['out_dir']}/result_topk.json", flush=True)
    print(f"[saved] {CONFIG['out_dir']}/scores.npy", flush=True)
    print(f"\n(done) total elapsed={format_seconds(time.perf_counter() - t0)}", flush=True)


if __name__ == "__main__":
    main()