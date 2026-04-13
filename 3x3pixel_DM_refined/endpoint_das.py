# endpoint_das_projected.py

import os
import glob
import json
import random
import time
import hashlib
from dataclasses import dataclass
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


def print_cuda_mem(tag: str):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        reserv = torch.cuda.memory_reserved() / (1024 ** 3)
        max_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(
            f"[mem] {tag} | allocated={alloc:.2f} GB | reserved={reserv:.2f} GB | max_alloc={max_alloc:.2f} GB",
            flush=True,
        )


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
    return cond.unsqueeze(0)


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
# Active parameters
# ============================================================
def set_active_params_baseline(model: nn.Module) -> List[nn.Parameter]:
    active = []
    for _, p in model.named_parameters():
        p.requires_grad_(True)
        active.append(p)
    return active


def num_active_params(active: List[nn.Parameter]) -> int:
    return sum(p.numel() for p in active)


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
        k = min(int(extra_random_points), len(remaining))
        extras = rnd.sample(remaining, k=k) if k > 0 else []

    items = base_items + [(0, int(i)) for i in extras]
    if shuffle:
        rnd = random.Random(seed + 999)
        rnd.shuffle(items)
    return items


def iter_batches(items: List[Tuple[int, int]], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield i, items[i:i + batch_size]


def get_one(ds, idx: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    x0, cond = ds[idx][:2]
    x0 = x0.unsqueeze(0).to(device)
    cond = cond.unsqueeze(0).to(device) if cond.dim() == 1 else cond.to(device)
    return x0, cond


# ============================================================
# Projected DAS helpers
# ============================================================
@dataclass
class ProjectedDASBundle:
    B: torch.Tensor      # [m, d]
    BtB: torch.Tensor    # [d, d]
    Btr: torch.Tensor    # [d]
    resid_norm2: float


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


def sample_xt_and_noise(
    x0: torch.Tensor,
    sched,
    *,
    t: torch.Tensor,
    rng: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    noise = torch.randn(x0.shape, generator=rng, device=x0.device, dtype=x0.dtype)
    xt = base.q_sample(x0, t, noise, sched)
    return xt, noise


def compute_projected_jacobian_bundle(
    model: nn.Module,
    active: List[nn.Parameter],
    xt: torch.Tensor,
    t: torch.Tensor,
    cond: torch.Tensor,
    noise: torch.Tensor,
    *,
    proj_dim: int,
    device: torch.device,
    seed_parts: Tuple[Any, ...],
) -> ProjectedDASBundle:
    pred = model(xt, t, cond)
    pred_flat = pred.reshape(-1)
    noise_flat = noise.reshape(-1)
    resid = (pred_flat - noise_flat).to(device=device, dtype=torch.float32)

    rows = []
    for k in range(pred_flat.numel()):
        grads = torch.autograd.grad(
            pred_flat[k],
            active,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )

        bk = _countsketch_project_grad(
            grads,
            proj_dim,
            device=device,
            seed_parts=(*seed_parts, "row", k),
        )  # [d]
        rows.append(bk)

    B = torch.stack(rows, dim=0)      # [m, d]
    BtB = B.t().matmul(B)             # [d, d]
    Btr = B.t().matmul(resid)         # [d]

    return ProjectedDASBundle(
        B=B.detach(),
        BtB=BtB.detach(),
        Btr=Btr.detach(),
        resid_norm2=float(resid.pow(2).sum().item()),
    )


def build_reference_set_for_H(
    score_items: List[Tuple[int, int]],
    hessian_mode: str,
) -> List[Tuple[int, int]]:
    if hessian_mode == "selected":
        return score_items
    if hessian_mode == "caselines":
        return [(src, idx) for (src, idx) in score_items if src != 0]
    raise ValueError(f"Unknown hessian_mode: {hessian_mode}")


# ============================================================
# Main
# ============================================================
def main():
    MODEL = "model_109900"
    LORA = "r"
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
        batch_size=64,

        # projected DAS-specific
        timesteps=[0, 400, 800, 1200, 1600, 1999],
        num_mc_noise=8,
        damping=1e-3,
        hessian_mode="selected",
        proj_dim=4069,

        # optional cap on H reference set
        max_ref_points_for_H=200,

        # checkpoint control
        ckpt_stride=1,
        max_num_ckpts=1,

        topk=2000,
        out_dir=f"the_other_runs/{CUR_MODEL}_endpoint_das_projected/baseline",
    )

    t0 = time.perf_counter()
    device = torch.device(CONFIG["device"])
    set_seed(int(CONFIG["seed"]))

    print(f"[device] Using device: {device}", flush=True)
    if device.type == "cuda":
        print(f"[device] GPU count visible: {torch.cuda.device_count()}", flush=True)
        print(f"[device] GPU name: {torch.cuda.get_device_name(0)}", flush=True)

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

    ref_items_for_H = build_reference_set_for_H(score_items, CONFIG["hessian_mode"])

    max_ref = CONFIG.get("max_ref_points_for_H", None)
    if max_ref is not None and len(ref_items_for_H) > int(max_ref):
        rnd = random.Random(int(CONFIG["seed"]) + 12345)
        ref_items_for_H = rnd.sample(ref_items_for_H, k=int(max_ref))

    print(f"[candidate-set] N_total={N_total}  M_selected={M}  extra_random_points={CONFIG['extra_random_points']}", flush=True)
    print(f"[H-set] mode={CONFIG['hessian_mode']}  M_ref={len(ref_items_for_H)}", flush=True)

    scores = torch.zeros(M, dtype=torch.float64)

    damping = float(CONFIG["damping"])
    timesteps = [int(t) for t in CONFIG["timesteps"]]
    num_mc_noise = int(CONFIG["num_mc_noise"])
    proj_dim = int(CONFIG["proj_dim"])
    topk = min(int(CONFIG["topk"]), M)

    num_ref_batches = (len(ref_items_for_H) + int(CONFIG["batch_size"]) - 1) // int(CONFIG["batch_size"])
    num_score_batches = (M + int(CONFIG["batch_size"]) - 1) // int(CONFIG["batch_size"])

    total_terms = 0

    for ckpt_i, ckpt_path in enumerate(baseline_ckpts):
        ckpt_t0 = time.perf_counter()
        print(
            f"\n[checkpoint] {ckpt_i+1}/{len(baseline_ckpts)} | {os.path.basename(ckpt_path)}",
            flush=True,
        )

        model_k, _ = build_model_from_baseline_ckpt(ckpt_path, device=device)
        active = set_active_params_baseline(model_k)
        P = num_active_params(active)
        model_k.train()

        print(f"[checkpoint] active params P={P} | proj_dim={proj_dim}", flush=True)

        for t_idx, t_value in enumerate(timesteps):
            if not (0 <= t_value < T):
                raise ValueError(f"Invalid timestep {t_value}; T={T}")

            print(
                f"[timestep] ckpt {ckpt_i+1}/{len(baseline_ckpts)} | "
                f"timestep {t_idx+1}/{len(timesteps)} | t={t_value}",
                flush=True,
            )

            t_tensor = torch.tensor([t_value], device=device, dtype=torch.long)

            for mc_i in range(num_mc_noise):
                term_t0 = time.perf_counter()
                total_terms += 1

                print(
                    f"[mc] ckpt {ckpt_i+1}/{len(baseline_ckpts)} | t={t_value} | "
                    f"mc {mc_i+1}/{num_mc_noise} | building query bundle",
                    flush=True,
                )

                rng_q = make_torch_generator(device, CONFIG["seed"], "pdas_q", ckpt_i, t_value, mc_i)
                xt_q, noise_q = sample_xt_and_noise(x0_ref, sched, t=t_tensor, rng=rng_q)
                bundle_q = compute_projected_jacobian_bundle(
                    model=model_k,
                    active=active,
                    xt=xt_q,
                    t=t_tensor,
                    cond=query_cond,
                    noise=noise_q,
                    proj_dim=proj_dim,
                    device=device,
                    seed_parts=(CONFIG["seed"], "pdas_q_bundle", ckpt_i, t_value, mc_i),
                )

                del xt_q, noise_q
                cleanup_cuda()
                print_cuda_mem("after query bundle")

                print(
                    f"[mc] query bundle ready | query_resid2={bundle_q.resid_norm2:.6f}",
                    flush=True,
                )

                # ----------------------------------------------------
                # PASS 1: build projected Hessian
                # ----------------------------------------------------
                H_proj = damping * torch.eye(proj_dim, device=device, dtype=torch.float32)
                bundle_cache: Dict[int, ProjectedDASBundle] = {}

                for batch_idx, (_, batch) in enumerate(iter_batches(ref_items_for_H, int(CONFIG["batch_size"]))):
                    if should_print_batch(batch_idx, num_ref_batches):
                        print_batch_progress(
                            "PASS1/H-build",
                            batch_idx,
                            num_ref_batches,
                            extra=f"ckpt={ckpt_i+1}/{len(baseline_ckpts)}, t={t_value}, mc={mc_i+1}/{num_mc_noise}",
                        )

                    for (_, idx) in batch:
                        x0_i, cond_i = get_one(ds, idx, device)
                        rng_i = make_torch_generator(device, CONFIG["seed"], "pdas_tr", ckpt_i, t_value, mc_i, idx)
                        xt_i, noise_i = sample_xt_and_noise(x0_i, sched, t=t_tensor, rng=rng_i)

                        bundle_i = compute_projected_jacobian_bundle(
                            model=model_k,
                            active=active,
                            xt=xt_i,
                            t=t_tensor,
                            cond=cond_i,
                            noise=noise_i,
                            proj_dim=proj_dim,
                            device=device,
                            seed_parts=(CONFIG["seed"], "pdas_tr_bundle", ckpt_i, t_value, mc_i, idx),
                        )

                        H_proj += bundle_i.BtB
                        bundle_cache[idx] = bundle_i

                        del x0_i, cond_i, xt_i, noise_i
                        cleanup_cuda()

                print("[mc] PASS1 complete. Starting PASS2 scoring...", flush=True)
                print_cuda_mem("after PASS1 projected H")

                # ----------------------------------------------------
                # PASS 2: compute candidate projected DAS scores
                # ----------------------------------------------------
                batch_losses_dbg = []

                for batch_idx, (start, batch) in enumerate(iter_batches(score_items, int(CONFIG["batch_size"]))):
                    if should_print_batch(batch_idx, num_score_batches):
                        print_batch_progress(
                            "PASS2/Score",
                            batch_idx,
                            num_score_batches,
                            extra=f"ckpt={ckpt_i+1}/{len(baseline_ckpts)}, t={t_value}, mc={mc_i+1}/{num_mc_noise}",
                        )

                    batch_scores = []
                    for (_, idx) in batch:
                        if idx in bundle_cache:
                            bundle_i = bundle_cache[idx]
                            temporary_bundle = False
                        else:
                            x0_i, cond_i = get_one(ds, idx, device)
                            rng_i = make_torch_generator(device, CONFIG["seed"], "pdas_tr", ckpt_i, t_value, mc_i, idx)
                            xt_i, noise_i = sample_xt_and_noise(x0_i, sched, t=t_tensor, rng=rng_i)

                            bundle_i = compute_projected_jacobian_bundle(
                                model=model_k,
                                active=active,
                                xt=xt_i,
                                t=t_tensor,
                                cond=cond_i,
                                noise=noise_i,
                                proj_dim=proj_dim,
                                device=device,
                                seed_parts=(CONFIG["seed"], "pdas_tr_bundle", ckpt_i, t_value, mc_i, idx),
                            )
                            temporary_bundle = True

                            del x0_i, cond_i, xt_i, noise_i
                            cleanup_cuda()

                        H_loo = H_proj - bundle_i.BtB
                        delta_z_i = torch.linalg.solve(H_loo, bundle_i.Btr)           # [d]
                        delta_eps_q = bundle_q.B.matmul(delta_z_i)                    # [m]
                        score_i = float(delta_eps_q.pow(2).sum().item())

                        batch_scores.append(score_i)
                        batch_losses_dbg.append(bundle_i.resid_norm2)

                        del H_loo, delta_z_i, delta_eps_q
                        if temporary_bundle:
                            del bundle_i

                    scores[start:start + len(batch)] += torch.tensor(batch_scores, dtype=torch.float64)

                avg_resid = float(np.mean(batch_losses_dbg)) if batch_losses_dbg else 0.0
                elapsed_term = time.perf_counter() - term_t0
                print(
                    f"[done] ckpt {ckpt_i+1}/{len(baseline_ckpts)} | t={t_value} | mc {mc_i+1}/{num_mc_noise} | "
                    f"query_resid2={bundle_q.resid_norm2:.6f} | avg_train_resid2={avg_resid:.6f} | "
                    f"elapsed={format_seconds(elapsed_term)}",
                    flush=True,
                )

                bundle_cache.clear()
                del H_proj, bundle_q, bundle_cache
                cleanup_cuda()
                print_cuda_mem("after cleanup")

        elapsed_ckpt = time.perf_counter() - ckpt_t0
        print(
            f"[checkpoint done] {os.path.basename(ckpt_path)} | elapsed={format_seconds(elapsed_ckpt)}",
            flush=True,
        )

    if total_terms > 0:
        scores /= float(total_terms)

    vals, ord_idx = torch.topk(scores, k=topk, largest=True)

    top = []
    for r in range(topk):
        j = int(ord_idx[r].item())
        src, train_idx = score_items[j]
        top.append({"idx": int(train_idx), "src": int(src), "score": float(vals[r].item())})

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
        "M_scored": int(M),
        "M_ref_for_H": int(len(ref_items_for_H)),
        "device": str(device),
        "seed": int(CONFIG["seed"]),
        "timesteps": timesteps,
        "num_mc_noise": int(num_mc_noise),
        "damping": float(damping),
        "proj_dim": int(proj_dim),
        "num_terms_total": int(total_terms),
        "elapsed_sec": float(time.perf_counter() - t0),
        "extra_random_points": int(CONFIG["extra_random_points"]),
        "hessian_mode": CONFIG["hessian_mode"],
        "solver": "projected_das_countsketch",
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
    np.save(os.path.join(CONFIG["out_dir"], "scores.npy"), scores.numpy())

    print(f"\n[saved] {CONFIG['out_dir']}/run_config.json", flush=True)
    print(f"[saved] {CONFIG['out_dir']}/run_info.json", flush=True)
    print(f"[saved] {CONFIG['out_dir']}/score_indices.json", flush=True)
    print(f"[saved] {CONFIG['out_dir']}/result_topk.json", flush=True)
    print(f"[saved] {CONFIG['out_dir']}/scores.npy", flush=True)
    print(f"\n(done) total elapsed={format_seconds(time.perf_counter() - t0)}", flush=True)


if __name__ == "__main__":
    main()