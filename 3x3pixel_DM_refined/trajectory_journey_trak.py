import os
import glob
import json
import random
import time
import hashlib
from typing import Dict, Any, List, Tuple, Optional, Set

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


def _stable_int_seed(*parts: Any) -> int:
    s = "|".join(map(str, parts)).encode("utf-8")
    h = hashlib.sha256(s).digest()
    v = int.from_bytes(h[:8], "little", signed=False)
    return int(v % (2**63 - 1))


def make_torch_generator(device: torch.device, *seed_parts: Any) -> torch.Generator:
    g = torch.Generator(device=str(device))
    g.manual_seed(_stable_int_seed(*seed_parts))
    return g


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


def print_batch_progress(
    phase: str,
    batch_idx: int,
    num_batches: int,
    *,
    extra: str = "",
):
    msg = f"    [{phase}] batch {batch_idx+1}/{num_batches}"
    if extra:
        msg += f" | {extra}"
    print(msg, flush=True)


def should_print_batch(batch_idx: int, num_batches: int, every: int = 10) -> bool:
    return batch_idx == 0 or (batch_idx + 1) % every == 0 or (batch_idx + 1) == num_batches


def cleanup_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Helpers: checkpoint discovery
# ============================================================
def list_checkpoints_sorted(dir_path: str, pattern: str = "*.pt") -> List[str]:
    if not os.path.isdir(dir_path):
        return []
    paths = glob.glob(os.path.join(dir_path, pattern))
    paths.sort()
    return paths


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
# Load baseline ckpt into a ready-to-run model
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

    model = base.CondEpsModel(
        in_ch=3, cond_dim=cond_dim, base_ch=base_ch, time_dim=time_dim
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    meta = {
        "T": int(ckpt["T"]),
        "cond_dim": cond_dim,
        "vocab": ckpt.get("vocab", None),
        "base_ch": base_ch,
        "time_dim": time_dim,
        "grid_size": int(ckpt.get("grid_size", 3)),
    }
    return model, meta


# ============================================================
# Active parameters (baseline-only)
# ============================================================
def set_active_params_baseline(model: nn.Module) -> List[nn.Parameter]:
    active = []
    for _, p in model.named_parameters():
        p.requires_grad_(True)
        active.append(p)
    return active


# ============================================================
# Schedule / x0 helpers
# ============================================================
def _get_alphabar_at_t(sched, t: torch.Tensor) -> torch.Tensor:
    candidates = [
        "alphabar",
        "alpha_bar",
        "alpha_bars",
        "alphas_bar",
        "alphas_cumprod",
    ]
    arr = None
    for name in candidates:
        if hasattr(sched, name):
            arr = getattr(sched, name)
            break
    if arr is None:
        raise AttributeError(
            "Could not find alpha-bar / alphas_cumprod on sched. "
            "Please expose it on your schedule object."
        )

    if not torch.is_tensor(arr):
        arr = torch.tensor(arr, device=t.device, dtype=torch.float32)
    else:
        arr = arr.to(device=t.device, dtype=torch.float32)

    return arr[t].view(-1, 1, 1, 1)


def predict_x0_from_xt(
    model: nn.Module,
    sched,
    xt: torch.Tensor,
    t: torch.Tensor,
    cond: torch.Tensor,
) -> torch.Tensor:
    eps_pred = model(xt, t, cond)
    ab_t = _get_alphabar_at_t(sched, t).to(dtype=xt.dtype)
    return (xt - torch.sqrt(1.0 - ab_t) * eps_pred) / torch.sqrt(ab_t.clamp_min(1e-12))


# ============================================================
# Losses / features
# ============================================================
def diffusion_train_loss_expected_at_t(
    model: nn.Module,
    sched,
    x0: torch.Tensor,
    cond: torch.Tensor,
    t: torch.Tensor,
    *,
    rng: torch.Generator,
    num_expectation_samples: int,
) -> torch.Tensor:
    """
    Monte Carlo estimate of E_eps ||eps_theta(x_t,t,cond)-eps||^2
    at a fixed timestep t.
    """
    losses = []
    for _ in range(int(num_expectation_samples)):
        noise = torch.randn(x0.shape, generator=rng, device=x0.device, dtype=x0.dtype)
        xt = base.q_sample(x0, t, noise, sched)
        eps_pred = model(xt, t, cond)
        losses.append(F.mse_loss(eps_pred, noise, reduction="mean"))
    return torch.stack(losses).mean()


def trajectory_query_loss_expected_at_step(
    model: nn.Module,
    sched,
    xt: torch.Tensor,
    t: torch.Tensor,
    cond: torch.Tensor,
    *,
    rng: torch.Generator,
    num_expectation_samples: int,
) -> torch.Tensor:
    """
    Journey-side query loss at a fixed trajectory step:
      1) infer xhat0 from xt
      2) re-noise xhat0 at the same t
      3) evaluate denoising loss
    """
    xhat0 = predict_x0_from_xt(model, sched, xt, t, cond).detach()

    losses = []
    for _ in range(int(num_expectation_samples)):
        noise = torch.randn(xhat0.shape, generator=rng, device=xhat0.device, dtype=xhat0.dtype)
        xt_recon = base.q_sample(xhat0, t, noise, sched)
        eps_pred = model(xt_recon, t, cond)
        losses.append(F.mse_loss(eps_pred, noise, reduction="mean"))

    return torch.stack(losses).mean()


# ============================================================
# Deterministic projected gradient features
# ============================================================
def _countsketch_project_grad(
    grads: Tuple[Optional[torch.Tensor], ...],
    d: int,
    *,
    device: torch.device,
    seed_parts: Tuple[Any, ...],
) -> torch.Tensor:
    out = torch.zeros(d, device=device, dtype=torch.float32)

    for t_idx, g in enumerate(grads):
        if g is None:
            continue
        flat = g.detach().reshape(-1).to(device=device, dtype=torch.float32)
        n = flat.numel()
        if n == 0:
            continue

        gen = make_torch_generator(device, *seed_parts, "tensor", t_idx, n, d)
        idx = torch.randint(0, d, (n,), generator=gen, device=device, dtype=torch.int64)
        sign = torch.randint(0, 2, (n,), generator=gen, device=device, dtype=torch.int8) * 2 - 1
        sign = sign.to(dtype=torch.float32)

        out.index_add_(0, idx, sign * flat)

    return out / (float(d) ** 0.5)


def grad_feature_phi(
    active_params: List[nn.Parameter],
    loss: torch.Tensor,
    *,
    proj_dim: int,
    device: torch.device,
    seed_parts: Tuple[Any, ...],
) -> torch.Tensor:
    grads = torch.autograd.grad(
        loss,
        active_params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    return _countsketch_project_grad(grads, proj_dim, device=device, seed_parts=seed_parts)


# ============================================================
# DDIM reference trajectory
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
      save_steps: step indices in DDIM loop
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
        raise RuntimeError(f"traj_use len {len(traj_use)} != t_seq len {len(t_seq)}")

    return [x.detach() for x in traj_use], t_seq, save_steps


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
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    if not lines:
        return []
    if lines[0].lower() == col.lower():
        lines = lines[1:]
    return [int(ln.split(",")[0]) for ln in lines]


# ============================================================
# Candidate-set construction
# ============================================================
def build_candidate_items(
    N: int,
    idx_lists: List[List[int]],
    extra_random_points: int,
    seed: int,
    shuffle: bool = True,
) -> List[Tuple[int, int]]:
    clean_lists: List[List[int]] = []
    for lst in idx_lists:
        clean = sorted({int(i) for i in lst if 0 <= int(i) < N})
        clean_lists.append(clean)

    seen: Set[int] = set()
    base_items: List[Tuple[int, int]] = []
    for k, lst in enumerate(clean_lists):
        src_id = k + 1
        for i in lst:
            if i in seen:
                continue
            seen.add(i)
            base_items.append((src_id, i))

    remaining = [i for i in range(N) if i not in seen]

    extras: List[int] = []
    if extra_random_points == -1:
        extras = remaining
    elif extra_random_points > 0:
        rnd = random.Random(seed)
        kk = min(int(extra_random_points), len(remaining))
        extras = rnd.sample(remaining, k=kk) if kk > 0 else []

    items = base_items + [(0, int(i)) for i in extras]
    if shuffle:
        rnd = random.Random(seed + 999)
        rnd.shuffle(items)
    return items


# ============================================================
# Batch helpers
# ============================================================
def iter_batches(items: List[Tuple[int, int]], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield i, items[i:i + batch_size]


def get_one(ds, idx: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    x0, cond = ds[idx][:2]
    x0 = x0.unsqueeze(0).to(device)
    cond = cond.unsqueeze(0).to(device) if cond.dim() == 1 else cond.to(device)
    return x0, cond


# ============================================================
# Main
# ============================================================
def main():
    MODEL = "model_109900"
    LORA = "by"
    CUR_MODEL = f"{MODEL}_{LORA}"

    CONFIG = dict(
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=808,
        csv_path="generated_database/49_100000.csv",
        grid_size=3,

        baseline_dir=f"models_checkpoints/{LORA}/{MODEL}/baseline",

        query_labels=["background_color_red", "background_color_blue", "background_color_yellow"],
        ddim_steps=2000,
        eta=0.0,

        idx_list_1="generated_database/RBY/subset/blue_A_idx.csv",
        idx_list_2="generated_database/RBY/subset/blue_B_idx.csv",
        idx_list_3="generated_database/RBY/subset/red_A_idx.csv",
        idx_list_4="generated_database/RBY/subset/red_B_idx.csv",
        idx_list_5="generated_database/RBY/subset/yellow_A_idx.csv",
        idx_list_6="generated_database/RBY/subset/yellow_B_idx.csv",
        idx_col_name="idx",

        extra_random_points=0,

        proj_dim=4096,
        damping=1e-3,
        num_samples=1,
        batch_size=64,

        train_expectation_samples=8,
        query_expectation_samples=8,

        num_query_traj_steps=50,

        ckpt_stride=1,
        max_num_ckpts=1,

        topk=2000,
        out_dir=f"the_other_runs/{CUR_MODEL}_endpoint_journey_trak/baseline",
    )

    t0 = time.perf_counter()
    device = torch.device(CONFIG["device"])
    set_seed(int(CONFIG["seed"]))

    print(f"[device] Using device: {device}", flush=True)
    if device.type == "cuda":
        print(f"[device] GPU count visible: {torch.cuda.device_count()}", flush=True)
        print(f"[device] GPU name: {torch.cuda.get_device_name(0)}", flush=True)

    # dataset
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

    # checkpoints
    print("[setup] Scanning checkpoints...", flush=True)
    baseline_ckpts = list_checkpoints_sorted(CONFIG["baseline_dir"])
    if not baseline_ckpts:
        raise FileNotFoundError(f"No baseline checkpoints found in: {CONFIG['baseline_dir']}")

    ckpt_stride = int(CONFIG.get("ckpt_stride", 1))
    max_num_ckpts = CONFIG.get("max_num_ckpts", None)
    if ckpt_stride <= 0:
        raise ValueError(f"ckpt_stride must be >= 1, got {ckpt_stride}")
    baseline_ckpts = baseline_ckpts[::ckpt_stride]
    if max_num_ckpts is not None:
        max_num_ckpts = int(max_num_ckpts)
        if max_num_ckpts <= 0:
            raise ValueError(f"max_num_ckpts must be positive or None, got {max_num_ckpts}")
        baseline_ckpts = baseline_ckpts[-max_num_ckpts:]
    if not baseline_ckpts:
        raise RuntimeError("Checkpoint filtering removed all checkpoints.")

    print(f"[checkpoints] selected={len(baseline_ckpts)}", flush=True)
    for p in baseline_ckpts:
        print("  ", os.path.basename(p), flush=True)

    ref_ckpt = baseline_ckpts[-1]
    print(f"[setup] Reference checkpoint: {os.path.basename(ref_ckpt)}", flush=True)
    ref_model, ref_meta = build_model_from_baseline_ckpt(ref_ckpt, device=device)
    T = int(ref_meta["T"])
    sched = base.make_linear_schedule(T, device=device)
    print(f"[setup] Diffusion steps T={T}", flush=True)

    vocab = getattr(ds, "vocab", None) or ref_meta.get("vocab", None)
    if vocab is None:
        raise RuntimeError("No vocab found in dataset or ckpt; cannot build query cond from label strings.")
    query_cond = labels_to_cond(CONFIG["query_labels"], vocab, cond_dim, device=device)

    # reference generation trajectory from final checkpoint
    print("[setup] Computing reference trajectory...", flush=True)
    ref_traj, t_seq, save_steps = compute_reference_trajectory(
        model=ref_model,
        sched=sched,
        cond=query_cond,
        shape=(1, C, H, W),
        seed=int(CONFIG["seed"]),
        steps=int(CONFIG["ddim_steps"]),
        eta=float(CONFIG["eta"]),
        device=device,
        num_keep=int(CONFIG["num_query_traj_steps"]),
    )
    print("[setup] Reference trajectory ready.", flush=True)

    traj_len = len(ref_traj)
    selected_traj_idx = list(range(traj_len))
    selected_diff_t = [int(t_seq[i]) for i in selected_traj_idx]
    K = len(selected_traj_idx)

    # candidate set
    print("[setup] Building candidate set...", flush=True)
    N_total = len(ds)
    idx_lists = [load_index_list(CONFIG[f"idx_list_{k}"], col=CONFIG["idx_col_name"]) for k in range(1, 7)]
    score_items = build_candidate_items(
        N=N_total,
        idx_lists=idx_lists,
        extra_random_points=int(CONFIG["extra_random_points"]),
        seed=int(CONFIG["seed"]),
        shuffle=True,
    )
    M = len(score_items)
    if M == 0:
        raise RuntimeError("No training points selected. Check idx lists and extra_random_points.")

    print(f"[candidate-set] N_total={N_total}  M_selected={M}  extra_random_points={CONFIG['extra_random_points']}", flush=True)
    print(f"[trajectory] len={traj_len}  selected={K}  traj_idx={selected_traj_idx}", flush=True)
    print(f"[trajectory] diffusion t = {selected_diff_t}", flush=True)

    d = int(CONFIG["proj_dim"])
    lam = float(CONFIG["damping"])
    S = int(CONFIG["num_samples"])
    bs = int(CONFIG["batch_size"])
    num_batches = (M + bs - 1) // bs

    # aggregate overall score + per-step score
    scores_overall = torch.zeros(M, dtype=torch.float64)
    scores_per_step = torch.zeros((M, K), dtype=torch.float64)

    total_s = 0
    for ckpt_i, ckpt_path in enumerate(baseline_ckpts):
        ckpt_t0 = time.perf_counter()
        print(
            f"\n[checkpoint] {ckpt_i+1}/{len(baseline_ckpts)} | {os.path.basename(ckpt_path)}",
            flush=True,
        )

        model_k, _ = build_model_from_baseline_ckpt(ckpt_path, device=device)
        active = set_active_params_baseline(model_k)
        model_k.train()

        for s_i in range(S):
            sample_t0 = time.perf_counter()
            total_s += 1

            print(
                f"[sample] ckpt {ckpt_i+1}/{len(baseline_ckpts)} | sample {s_i+1}/{S} | building query-side features",
                flush=True,
            )

            # ----------------------------------------------------
            # build selected query-side features Gq = [g_1; ...; g_K]
            # ----------------------------------------------------
            g_list = []
            q_losses_dbg = []

            for q_pos, traj_idx in enumerate(selected_traj_idx):
                t_val = int(selected_diff_t[q_pos])

                if q_pos == 0 or (q_pos + 1) % 5 == 0 or (q_pos + 1) == K:
                    print(
                        f"    [query-step] {q_pos+1}/{K} | traj_idx={traj_idx} | t={t_val}",
                        flush=True,
                    )

                xt_q = ref_traj[traj_idx].to(device)
                t_tensor = torch.tensor([t_val], device=device, dtype=torch.long)

                rng_q = make_torch_generator(
                    device, CONFIG["seed"], "journey_q", ckpt_i, s_i, traj_idx, t_val
                )
                Lq_t = trajectory_query_loss_expected_at_step(
                    model=model_k,
                    sched=sched,
                    xt=xt_q,
                    t=t_tensor,
                    cond=query_cond,
                    rng=rng_q,
                    num_expectation_samples=int(CONFIG["query_expectation_samples"]),
                )
                g_t = grad_feature_phi(
                    active_params=active,
                    loss=Lq_t,
                    proj_dim=d,
                    device=device,
                    seed_parts=(CONFIG["seed"], "journey_g", ckpt_i, s_i, traj_idx, t_val),
                )

                g_list.append(g_t)
                q_losses_dbg.append(float(Lq_t.detach().item()))

                del xt_q, t_tensor, Lq_t
                cleanup_cuda()

            Gq = torch.stack(g_list, dim=1)  # [d, K]

            print(
                f"[sample] query-side features ready | mean_query_loss={float(np.mean(q_losses_dbg)):.6f}",
                flush=True,
            )

            # ----------------------------------------------------
            # For each query-step k:
            #   build step-aligned Gram
            #   solve U_k
            #   score all candidates at same t_k
            # ----------------------------------------------------
            for q_pos, traj_idx in enumerate(selected_traj_idx):
                t_val = int(selected_diff_t[q_pos])
                t_tensor = torch.tensor([t_val], device=device, dtype=torch.long)

                print(
                    f"[sample-step] {q_pos+1}/{K} | traj_idx={traj_idx} | t={t_val} | building Gram",
                    flush=True,
                )

                G = torch.zeros((d, d), device=device, dtype=torch.float32)

                for batch_idx, (_, batch) in enumerate(iter_batches(score_items, bs)):
                    if should_print_batch(batch_idx, num_batches):
                        print_batch_progress(
                            "PASS1/Gram",
                            batch_idx,
                            num_batches,
                            extra=f"ckpt={ckpt_i+1}/{len(baseline_ckpts)}, sample={s_i+1}/{S}, qstep={q_pos+1}/{K}, t={t_val}",
                        )

                    phis = []
                    for (_, idx) in batch:
                        x0, cond = get_one(ds, idx, device)
                        rng_i = make_torch_generator(
                            device, CONFIG["seed"], "tr_step", ckpt_i, s_i, q_pos, t_val, idx
                        )
                        Li = diffusion_train_loss_expected_at_t(
                            model_k,
                            sched,
                            x0,
                            cond,
                            t_tensor,
                            rng=rng_i,
                            num_expectation_samples=int(CONFIG["train_expectation_samples"]),
                        )
                        phi_i = grad_feature_phi(
                            active_params=active,
                            loss=Li,
                            proj_dim=d,
                            device=device,
                            seed_parts=(CONFIG["seed"], "phi_tr_step", ckpt_i, s_i, q_pos, t_val, idx),
                        )
                        phis.append(phi_i)

                        del x0, cond, Li
                        cleanup_cuda()

                    Phi = torch.stack(phis, dim=0)  # [B, d]
                    G += Phi.t().matmul(Phi)

                    del phis, Phi
                    cleanup_cuda()

                G += lam * torch.eye(d, device=device, dtype=torch.float32)
                u_k = torch.linalg.solve(G, Gq[:, q_pos])  # [d]

                print(
                    f"[sample-step] {q_pos+1}/{K} | t={t_val} | solve complete, starting scoring",
                    flush=True,
                )

                for batch_idx, (start, batch) in enumerate(iter_batches(score_items, bs)):
                    if should_print_batch(batch_idx, num_batches):
                        print_batch_progress(
                            "PASS2/Score",
                            batch_idx,
                            num_batches,
                            extra=f"ckpt={ckpt_i+1}/{len(baseline_ckpts)}, sample={s_i+1}/{S}, qstep={q_pos+1}/{K}, t={t_val}",
                        )

                    phis = []
                    for (_, idx) in batch:
                        x0, cond = get_one(ds, idx, device)
                        rng_i = make_torch_generator(
                            device, CONFIG["seed"], "tr_step", ckpt_i, s_i, q_pos, t_val, idx
                        )
                        Li = diffusion_train_loss_expected_at_t(
                            model_k,
                            sched,
                            x0,
                            cond,
                            t_tensor,
                            rng=rng_i,
                            num_expectation_samples=int(CONFIG["train_expectation_samples"]),
                        )
                        phi_i = grad_feature_phi(
                            active_params=active,
                            loss=Li,
                            proj_dim=d,
                            device=device,
                            seed_parts=(CONFIG["seed"], "phi_tr_step", ckpt_i, s_i, q_pos, t_val, idx),
                        )
                        phis.append(phi_i)

                        del x0, cond, Li
                        cleanup_cuda()

                    Phi = torch.stack(phis, dim=0)  # [B,d]
                    batch_scores_step = Phi.matmul(u_k)  # [B]

                    scores_per_step[start:start + len(batch), q_pos] += (
                        batch_scores_step.detach().to(torch.float64).cpu()
                    )
                    scores_overall[start:start + len(batch)] += (
                        (batch_scores_step / float(K)).detach().to(torch.float64).cpu()
                    )

                    del phis, Phi, batch_scores_step
                    cleanup_cuda()

                del G, u_k, t_tensor
                cleanup_cuda()

            elapsed_sample = time.perf_counter() - sample_t0
            print(
                f"[done] ckpt {ckpt_i+1}/{len(baseline_ckpts)} | sample {s_i+1}/{S} | "
                f"{os.path.basename(ckpt_path)} | "
                f"mean_query_loss={float(np.mean(q_losses_dbg)):.6f} | "
                f"batches={num_batches} | "
                f"elapsed={format_seconds(elapsed_sample)}",
                flush=True,
            )

            del Gq, g_list
            cleanup_cuda()

        elapsed_ckpt = time.perf_counter() - ckpt_t0
        print(
            f"[checkpoint done] {os.path.basename(ckpt_path)} | elapsed={format_seconds(elapsed_ckpt)}",
            flush=True,
        )

    if total_s > 0:
        scores_overall /= float(total_s)
        scores_per_step /= float(total_s)

    # top-k by overall averaged score
    topk = min(int(CONFIG["topk"]), M)
    vals, ord_idx = torch.topk(scores_overall, k=topk, largest=True)

    top = []
    for r in range(topk):
        j = int(ord_idx[r].item())
        src, train_idx = score_items[j]
        top.append({"idx": int(train_idx), "src": int(src), "score": float(vals[r].item())})

    # save
    print("[save] Writing outputs...", flush=True)
    ensure_dir(CONFIG["out_dir"])
    run_info = {
        "ref_ckpt": ref_ckpt,
        "num_ckpts": int(len(baseline_ckpts)),
        "ckpt_stride": int(CONFIG["ckpt_stride"]),
        "max_num_ckpts": CONFIG["max_num_ckpts"],
        "used_ckpts": [os.path.basename(p) for p in baseline_ckpts],
        "T": int(T),
        "ddim_steps": int(CONFIG["ddim_steps"]),
        "trajectory_length": int(traj_len),
        "num_query_traj_steps_requested": int(CONFIG["num_query_traj_steps"]),
        "num_query_traj_steps_used": int(K),
        "selected_traj_indices": [int(i) for i in selected_traj_idx],
        "selected_diffusion_timesteps": [int(t) for t in selected_diff_t],
        "save_steps": [int(s) for s in save_steps],
        "M_scored": int(M),
        "device": str(device),
        "seed": int(CONFIG["seed"]),
        "proj_dim": int(d),
        "damping": float(lam),
        "num_samples_total": int(total_s),
        "elapsed_sec": float(time.perf_counter() - t0),
        "extra_random_points": int(CONFIG["extra_random_points"]),
        "solver": "journey_trak_step_aligned_projected",
    }

    save_json(os.path.join(CONFIG["out_dir"], "run_config.json"), CONFIG)
    save_json(os.path.join(CONFIG["out_dir"], "run_info.json"), run_info)
    save_json(
        os.path.join(CONFIG["out_dir"], "score_indices.json"),
        {
            "N_total": int(N_total),
            "M_selected": int(M),
            "items": [{"src": int(s), "idx": int(i)} for (s, i) in score_items],
        },
    )
    save_json(
        os.path.join(CONFIG["out_dir"], "result_topk.json"),
        {"N_total": int(N_total), "M_selected": int(M), "topk": int(topk), "top": top},
    )

    np.save(os.path.join(CONFIG["out_dir"], "scores_overall.npy"), scores_overall.numpy())
    np.save(os.path.join(CONFIG["out_dir"], "scores_per_step.npy"), scores_per_step.numpy())

    print(f"\n[saved] {CONFIG['out_dir']}/run_config.json", flush=True)
    print(f"[saved] {CONFIG['out_dir']}/run_info.json", flush=True)
    print(f"[saved] {CONFIG['out_dir']}/score_indices.json", flush=True)
    print(f"[saved] {CONFIG['out_dir']}/result_topk.json", flush=True)
    print(f"[saved] {CONFIG['out_dir']}/scores_overall.npy", flush=True)
    print(f"[saved] {CONFIG['out_dir']}/scores_per_step.npy", flush=True)
    print(f"\n(done) total elapsed={format_seconds(time.perf_counter() - t0)}", flush=True)


if __name__ == "__main__":
    main()