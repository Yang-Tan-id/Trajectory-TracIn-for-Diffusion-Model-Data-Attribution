# sample_fixed_sv.py
import os
import re
import numpy as np
import torch
import math
import matplotlib.pyplot as plt

# Import from your training script (must contain these)
from x3pixel_DM_training import CondEpsModel, make_linear_schedule, ddim_sample

import torch.nn as nn


# =========================
# CONFIG (edit these)
# =========================
MODEL = 'b_base_0350'
#CKPT_PATH = f"models_checkpoints/model_100000//lora_update_N10000_bpurple10000/{MODEL}.pt"
CKPT_PATH = f"models_checkpoints/b/model_109900/baseline/baseline_0350.pt"
DEVICE = "cpu"

SEED = 808
# change this for different random starting noise / hue pattern
DDIM_STEPS = 2000                # sampling steps (more = slower, often better)
ETA = 0.0                       # keep 0.0 for deterministic DDIM

BASE_CH = 64                    # must match what you used in training (if you changed it)
TIME_DIM = 128                  # must match training

SAMPLE_LABELS = [
            "background_color_red",
            "background_color_blue",
            "background_color_yellow"
        ]

class LoRAConv2d(nn.Module):
    """LoRA wrapper for Conv2d: y = base(x) + scale * up(down(x))"""
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
        nn.init.zeros_(self.lora_up.weight)  # start as no-op

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


def latest_checkpoint_in_dir(dir_path: str, suffix: str = ".pt"):
    if not os.path.isdir(dir_path):
        return None
    files = [f for f in os.listdir(dir_path) if f.endswith(suffix)]
    if not files:
        return None
    files.sort()
    return os.path.join(dir_path, files[-1])



# Save 10 evenly spaced snapshots INCLUDING initial(0) and final(steps-1)
NUM_SNAPSHOTS = 30

CLIP_TO_01_FOR_DISPLAY = True   # only affects plotting

# ---- requested: fixed saturation + fixed "density/value", fixed tile size ----
FIXED_SATURATION = 0.9
FIXED_VALUE = 0.9
TILE_SIZE = 3  # 3x3 pixel grid


def _infer_baseline_from_lora_path(lora_ckpt_path: str) -> str:
    """
    兜底：如果 LoRA ckpt 里没有 baseline_ckpt，就从路径推测：
    models_checkpoints/model_100000/lora_update_xxx/lora_0040.pt
    -> models_checkpoints/model_100000/baseline/<latest>.pt
    """
    lora_dir = os.path.dirname(lora_ckpt_path)
    model_root = os.path.dirname(lora_dir)  # .../model_100000
    baseline_dir = os.path.join(model_root, "baseline")
    guess = latest_checkpoint_in_dir(baseline_dir)
    if guess is None:
        raise FileNotFoundError(f"Cannot infer baseline ckpt. baseline_dir not found or empty: {baseline_dir}")
    return guess


def load_checkpoint(ckpt_path: str, device: str):
    """
    支持两种 ckpt：
    1) baseline ckpt: contains ["model_state","T","cond_dim"]
    2) lora ckpt: contains ["lora_state"] (and maybe baseline_ckpt, lora_r, lora_alpha, lora_targets)

    返回： (merged_ckpt, T, grid_size, cond_dim, vocab)
    其中 merged_ckpt 一定包含 "model_state"（已合并 LoRA 到模型结构里）
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    # ---- Case A: baseline checkpoint (has full model_state) ----
    if "model_state" in ckpt:
        required = ["model_state", "T", "cond_dim"]
        for k in required:
            if k not in ckpt:
                raise KeyError(f"Checkpoint missing key: {k}")

        T = int(ckpt["T"])
        grid_size = int(ckpt.get("grid_size", 3))
        cond_dim = int(ckpt["cond_dim"])
        vocab = ckpt.get("vocab", None)
        return ckpt, T, grid_size, cond_dim, vocab

    # ---- Case B: LoRA-only checkpoint ----
    if "lora_state" in ckpt or ("lora_up" in str(list(ckpt.keys())[:10]) or "lora_down" in str(list(ckpt.keys())[:10])):
        # lora payload can be either {"lora_state": {...}, ...} or directly the state dict
        lora_sd = ckpt.get("lora_state", ckpt)

        baseline_ckpt_path = ckpt.get("baseline_ckpt", None)
        if baseline_ckpt_path is None:
            baseline_ckpt_path = _infer_baseline_from_lora_path(ckpt_path)

        base_ckpt = torch.load(baseline_ckpt_path, map_location=device)
        for k in ["model_state", "T", "cond_dim"]:
            if k not in base_ckpt:
                raise KeyError(f"Baseline checkpoint missing key: {k}  (path={baseline_ckpt_path})")

        T = int(base_ckpt["T"])
        grid_size = int(base_ckpt.get("grid_size", 3))
        cond_dim = int(base_ckpt["cond_dim"])
        vocab = base_ckpt.get("vocab", None)

        # read LoRA meta (fallback to defaults)
        lora_r = int(ckpt.get("lora_r", 4))
        lora_alpha = float(ckpt.get("lora_alpha", 4.0))
        lora_targets = ckpt.get("lora_targets", ["out_conv"])

        # Build model, load baseline weights, inject LoRA modules, load LoRA weights
        model = CondEpsModel(in_ch=3, cond_dim=cond_dim, base_ch=BASE_CH, time_dim=TIME_DIM).to(device)
        model.load_state_dict(base_ckpt["model_state"], strict=True)
        inject_lora_into_selected_convs(model, r=lora_r, alpha=lora_alpha, target_names=lora_targets)
        model.load_state_dict(lora_sd, strict=False)  # load only lora params

        # We return a "merged_ckpt-like" object with model_state ready for build_model()
        merged = dict(base_ckpt)
        merged["model_state"] = model.state_dict()
        merged["__merged_from_lora__"] = True
        merged["__lora_ckpt_path__"] = ckpt_path
        merged["__baseline_ckpt_path__"] = baseline_ckpt_path
        merged["lora_r"] = lora_r
        merged["lora_alpha"] = lora_alpha
        merged["lora_targets"] = list(lora_targets)
        return merged, T, grid_size, cond_dim, vocab

    raise KeyError(
        f"Unknown checkpoint format. Expect baseline with 'model_state' or LoRA with 'lora_state'. Keys={list(ckpt.keys())[:20]}"
    )



def build_model(cond_dim: int, base_ch: int, time_dim: int, device: str, state_dict):
    model = CondEpsModel(in_ch=3, cond_dim=cond_dim, base_ch=base_ch, time_dim=time_dim).to(device)

    # ---- auto-detect LoRA-wrapped state_dict ----
    has_lora_keys = any(k.startswith("out_conv.base.") or ".lora_" in k for k in state_dict.keys())
    if has_lora_keys:
        # infer targets from keys (works for out_conv; can be extended)
        targets = set()
        for k in state_dict.keys():
            # e.g. "out_conv.base.weight" -> "out_conv"
            if k.endswith(".base.weight") or k.endswith(".base.bias"):
                targets.add(k.split(".base.")[0])
        if len(targets) == 0:
            targets = {"out_conv"}

        # infer r/alpha: r from lora_down weight shape; alpha not stored in weights, so just keep your CONFIG values
        # r is the out_channels of lora_down (in_ch -> r)
        any_down = None
        for k, v in state_dict.items():
            if k.endswith("lora_down.weight"):
                any_down = v
                break
        r = int(any_down.shape[0]) if any_down is not None else 4
        alpha = 4.0  # use your script's intended alpha (must match training)

        inject_lora_into_selected_convs(model, r=r, alpha=alpha, target_names=targets)

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model



def labels_to_cond(labels, vocab, cond_dim: int, device: str) -> torch.Tensor:
    if vocab is None:
        raise RuntimeError(
            "Checkpoint has no vocab. Re-save your checkpoint with ds.vocab included, "
            "or provide a label->index mapping."
        )

    cond = torch.zeros(cond_dim, device=device)
    missing = [lab for lab in labels if lab not in vocab]
    if missing:
        print("❌ Labels not found in vocab:", missing)
        keys = list(vocab.keys())
        print("Some vocab keys:", keys[:60])
        raise SystemExit(1)

    for lab in labels:
        cond[vocab[lab]] = 1.0

    return cond.unsqueeze(0)  # (1, cond_dim)


def tensor_to_image(x: torch.Tensor) -> np.ndarray:
    """
    x: (1,3,H,W) tensor
    returns: (H,W,3) numpy
    """
    img = x[0].detach().cpu().permute(1, 2, 0).numpy()
    return img


# =========================
# Fixed-SV initial state helpers (HSV -> RGB)
# =========================
def hsv_to_rgb_torch(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Vectorized HSV->RGB.
    h,s,v: tensors of same shape, h in [0,1), s,v in [0,1]
    returns: rgb tensor with shape (..., 3) in [0,1]
    """
    h6 = h * 6.0
    i = torch.floor(h6).to(torch.int64)  # 0..5
    f = h6 - i

    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    i_mod = i % 6

    r = torch.zeros_like(h)
    g = torch.zeros_like(h)
    b = torch.zeros_like(h)

    mask = (i_mod == 0)
    r[mask], g[mask], b[mask] = v[mask], t[mask], p[mask]
    mask = (i_mod == 1)
    r[mask], g[mask], b[mask] = q[mask], v[mask], p[mask]
    mask = (i_mod == 2)
    r[mask], g[mask], b[mask] = p[mask], v[mask], t[mask]
    mask = (i_mod == 3)
    r[mask], g[mask], b[mask] = p[mask], q[mask], v[mask]
    mask = (i_mod == 4)
    r[mask], g[mask], b[mask] = t[mask], p[mask], v[mask]
    mask = (i_mod == 5)
    r[mask], g[mask], b[mask] = v[mask], p[mask], q[mask]

    return torch.stack([r, g, b], dim=-1)


def make_fixed_sv_init(
    batch: int,
    H: int,
    W: int,
    seed: int,
    fixed_s: float,
    fixed_v: float,
    device: str,
) -> torch.Tensor:
    """
    Create x_T with random hue but fixed saturation/value.
    Output: (batch, 3, H, W) in [0,1]
    """
    dev = torch.device(device)
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)

    h = torch.rand((batch, H, W), generator=gen, device=dev)  # [0,1)
    s = torch.full((batch, H, W), float(fixed_s), device=dev)
    v = torch.full((batch, H, W), float(fixed_v), device=dev)

    rgb = hsv_to_rgb_torch(h, s, v)                 # (batch, H, W, 3)
    rgb = rgb.permute(0, 3, 1, 2).contiguous()      # (batch, 3, H, W)
    return rgb


def get_evenly_spaced_steps(num: int, steps: int):
    if num <= 1:
        return [0]
    arr = np.linspace(0, steps - 1, num=num)
    idx = np.round(arr).astype(int).tolist()

    # ensure strictly valid and endpoints included
    idx[0] = 0
    idx[-1] = steps - 1

    # optional: make unique while preserving order (linspace+round can duplicate)
    seen = set()
    out = []
    for k in idx:
        if k not in seen:
            out.append(k)
            seen.add(k)

    # if duplicates caused fewer than num, fill by adding missing nearest integers
    if len(out) < num:
        for k in range(steps):
            if k not in seen:
                out.append(k)
                seen.add(k)
            if len(out) == num:
                
                break

    return out


def save_snapshot_image(x: torch.Tensor, save_path: str, clip01: bool = True):
    img = tensor_to_image(x)
    if clip01:
        img = np.clip(img, 0.0, 1.0)

    # Use matplotlib to save without axes
    plt.figure(figsize=(2, 2))
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()

def sanitize_for_path(s: str) -> str:
    # keep letters/numbers/_- and turn everything else into _
    s = re.sub(r"[^a-zA-Z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def rgb_to_hsv_torch(rgb: torch.Tensor):
    """
    rgb: (..., 3) in [0,1]
    returns: h,s,v each shape (...)
    h in [0,1)
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc, _ = torch.max(rgb, dim=-1)
    minc, _ = torch.min(rgb, dim=-1)
    v = maxc
    delta = maxc - minc

    # s
    s = torch.zeros_like(maxc)
    nonzero_v = v > 0
    s[nonzero_v] = delta[nonzero_v] / v[nonzero_v]

    # h
    h = torch.zeros_like(maxc)
    nonzero_delta = delta > 1e-12

    # masks for which channel is max
    mask_r = (maxc == r) & nonzero_delta
    mask_g = (maxc == g) & nonzero_delta
    mask_b = (maxc == b) & nonzero_delta

    # Hue formula in units of "sextants", then /6
    # For red max: (g-b)/delta mod 6
    h_r = ((g - b) / delta) % 6.0
    h_g = ((b - r) / delta) + 2.0
    h_b = ((r - g) / delta) + 4.0

    h[mask_r] = h_r[mask_r]
    h[mask_g] = h_g[mask_g]
    h[mask_b] = h_b[mask_b]

    h = (h / 6.0) % 1.0
    return h, s, v


def compute_hue_series(traj_full, device="cpu"):
    """
    traj_full: list of tensors, each (1,3,H,W)
    returns: hue array (steps,H,W) in numpy
    """
    with torch.no_grad():
        hs = []
        for x in traj_full:
            # (1,3,H,W) -> (H,W,3)
            img = x[0].detach().to(device).permute(1, 2, 0).clamp(0, 1)
            h, _, _ = rgb_to_hsv_torch(img)
            hs.append(h.cpu())
        hue = torch.stack(hs, dim=0).numpy()  # (steps,H,W)
    return hue


def compute_rgb_l2_tilewise(trajA, trajB):
    """
    trajA/trajB: list of tensors length steps, each (1,3,H,W)
    returns:
      dist_tiles: (steps,H,W) per-tile L2 in RGB
      dist_overall: (steps,) overall L2 over all tiles/channels
    """
    with torch.no_grad():
        d_tiles = []
        d_all = []
        for xa, xb in zip(trajA, trajB):
            a = xa[0].detach().cpu()  # (3,H,W)
            b = xb[0].detach().cpu()
            diff = (a - b)  # (3,H,W)

            # per-tile L2 over channel
            tile = torch.sqrt(torch.sum(diff * diff, dim=0))  # (H,W)
            d_tiles.append(tile)

            # overall L2 over everything
            overall = torch.sqrt(torch.sum(diff * diff))
            d_all.append(overall)

        dist_tiles = torch.stack(d_tiles, dim=0).numpy()  # (steps,H,W)
        dist_overall = torch.stack(d_all, dim=0).numpy()  # (steps,)
    return dist_tiles, dist_overall


def plot_hue_tiles(hue1, hue2, hue3, save_path, steps):
    """
    hue*: (steps,H,W)
    saves a 3x3 grid of line plots, each with 3 lines.
    """
    H, W = hue1.shape[1], hue1.shape[2]
    x = np.arange(steps)

    plt.figure(figsize=(14, 10))
    for i in range(H):
        for j in range(W):
            ax = plt.subplot(H, W, i * W + j + 1)
            ax.plot(x, hue1[:, i, j], label="Label1", linewidth=1.0)
            ax.plot(x, hue2[:, i, j], label="Label2", linewidth=1.0)
            ax.plot(x, hue3[:, i, j], label="Label3", linewidth=1.0)
            ax.set_title(f"tile({i},{j})")
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.2)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    plt.suptitle("Hue over diffusion steps (per tile)", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_dist_tiles(dist31_tiles, dist32_tiles, save_path, steps):
    """
    dist3*_tiles: (steps,H,W)
    saves a 3x3 grid of line plots, each with 2 lines (L3-L1 and L3-L2).
    """
    H, W = dist31_tiles.shape[1], dist31_tiles.shape[2]
    x = np.arange(steps)

    plt.figure(figsize=(14, 10))
    for i in range(H):
        for j in range(W):
            ax = plt.subplot(H, W, i * W + j + 1)
            ax.plot(x, dist31_tiles[:, i, j], label="Label3 - Label1", linewidth=1.0)
            ax.plot(x, dist32_tiles[:, i, j], label="Label3 - Label2", linewidth=1.0)
            ax.set_title(f"tile({i},{j}) RGB L2")
            ax.grid(True, alpha=0.2)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    plt.suptitle("Per-tile RGB L2 distance: Label3 vs (Label1, Label2)", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_overall_dist(dist31_all, dist32_all, save_path, steps):
    """
    dist3*_all: (steps,)
    saves a single plot with 2 lines.
    """
    x = np.arange(steps)
    plt.figure(figsize=(10, 4))
    plt.plot(x, dist31_all, label="Label3 - Label1", linewidth=1.2)
    plt.plot(x, dist32_all, label="Label3 - Label2", linewidth=1.2)
    plt.title("Overall RGB L2 distance (all tiles/channels)")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_ratio_tiles(dist31_tiles, dist32_tiles, save_path, steps, eps=1e-8):
    """
    dist31_tiles, dist32_tiles: (steps,H,W)
    plots 3x3 grid of ratio = (Label3-Label1)/(Label3-Label2 + eps)
    """
    H, W = dist31_tiles.shape[1], dist31_tiles.shape[2]
    x = np.arange(steps)

    ratio = dist31_tiles / (dist32_tiles + eps)

    plt.figure(figsize=(14, 10))
    for i in range(H):
        for j in range(W):
            ax = plt.subplot(H, W, i * W + j + 1)
            ax.plot(x, ratio[:, i, j], linewidth=1.0)
            ax.set_title(f"tile({i},{j}) ratio 31/32")
            ax.grid(True, alpha=0.2)

            # optional: keep plot readable if ratios explode early
            # ax.set_ylim(0, 5)

    plt.suptitle("Per-tile ratio: (Label3-Label1) / (Label3-Label2)", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_ratio_overall(dist31_all, dist32_all, save_path, steps, eps=1e-8):
    """
    dist31_all, dist32_all: (steps,)
    plots ratio = (Label3-Label1)/(Label3-Label2 + eps)
    """
    x = np.arange(steps)
    ratio = dist31_all / (dist32_all + eps)

    plt.figure(figsize=(10, 4))
    plt.plot(x, ratio, linewidth=1.2)
    plt.title("Overall ratio: (Label3-Label1) / (Label3-Label2)")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def make_uniform_fixed_sv_init(
    batch: int,
    H: int,
    W: int,
    seed: int,
    fixed_s: float,
    fixed_v: float,
    device: str,
) -> torch.Tensor:
    """
    Create x_T where ALL pixels share the same hue, with fixed saturation/value.
    Output: (batch, 3, H, W) in [0,1]
    """
    dev = torch.device(device)
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)

    # one hue for the entire image (and batch)
    h0 = torch.rand((batch, 1, 1), generator=gen, device=dev)  # [0,1)
    h = h0.expand(batch, H, W)

    s = torch.full((batch, H, W), float(fixed_s), device=dev)
    v = torch.full((batch, H, W), float(fixed_v), device=dev)

    rgb = hsv_to_rgb_torch(h, s, v)                 # (batch, H, W, 3)
    rgb = rgb.permute(0, 3, 1, 2).contiguous()      # (batch, 3, H, W)
    return rgb

def main():
    labels = SAMPLE_LABELS 
    labels_tag = "__".join(labels)
    labels_tag = sanitize_for_path(labels_tag)
    out_dir = os.path.join(f"models/{MODEL}/{NUM_SNAPSHOTS}", f"{SEED}_{labels_tag}")
    os.makedirs(out_dir, exist_ok=True)

    ckpt, T, grid_size, cond_dim, vocab = load_checkpoint(CKPT_PATH, DEVICE)
    print(f"Loaded: {CKPT_PATH}")
    print(f"T={T}, grid_size={grid_size}, cond_dim={cond_dim}, vocab_size={len(vocab) if vocab else None}")

    # Force tile size
    grid_size = TILE_SIZE

    # Keep DDIM_STEPS reasonable (<=T is typical)
    steps = min(DDIM_STEPS, T)

    # 10 evenly spaced snapshot indices among [0..steps-1]
    save_steps = get_evenly_spaced_steps(NUM_SNAPSHOTS, steps)
    print("Saving snapshots at step indices:", save_steps)

    model = build_model(cond_dim, BASE_CH, TIME_DIM, DEVICE, ckpt["model_state"])
    sched = make_linear_schedule(T, device=DEVICE)

    cond = labels_to_cond(labels, vocab, cond_dim, DEVICE)

    x_T = make_fixed_sv_init(
        batch=1,
        H=grid_size,
        W=grid_size,
        seed=SEED,
        fixed_s=FIXED_SATURATION,
        fixed_v=FIXED_VALUE,
        device=DEVICE,
    )

    traj = ddim_sample(
        model=model,
        sched=sched,
        cond=cond,
        shape=(1, 3, grid_size, grid_size),
        seed=SEED,
        steps=steps,
        eta=ETA,
        device=DEVICE,
        x_T=x_T,
        save_steps=save_steps,  # <-- must use the "returns List[Tensor]" version
    )

    # Save each snapshot as models/{MODEL}/{step_idx}.png
    for step_idx, x in zip(save_steps, traj):
        save_path = os.path.join(out_dir, f"{step_idx:04d}.png")
        save_snapshot_image(x, save_path, clip01=CLIP_TO_01_FOR_DISPLAY)

    # Also save a contact sheet for convenience (wrap into rows)
    n = len(traj)
    per_row = 10  # max images per row
    rows = math.ceil(n / per_row)
    cols = min(n, per_row)

    # tweak figure size: ~2.2 inches per tile
    plt.figure(figsize=(2.2 * cols, 2.2 * rows))

    for i, (step_idx, x) in enumerate(zip(save_steps, traj)):
        img = tensor_to_image(x)
        if CLIP_TO_01_FOR_DISPLAY:
            img = np.clip(img, 0.0, 1.0)

        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(str(step_idx))

    # Hide leftover subplots if n not divisible by cols
    total_slots = rows * cols
    for j in range(n, total_slots):
        ax = plt.subplot(rows, cols, j + 1)
        ax.axis("off")

    plt.suptitle(f"{MODEL} | seed={SEED} | steps={steps}", y=1.02)
    plt.tight_layout()

    sheet_path = os.path.join(out_dir, "sheet.png")
    plt.savefig(sheet_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.show()
    plt.close()


def main2():
    LABEL1 = [
    "shape_color_purple",
    ]
    LABEL2 = [
    "shape_color_red",
    "shape_color_blue",
    ]
    LABEL3 = [
    "shape_color_red",
    "shape_color_blue",
    "shape_color_purple",
     ]

    label_sets = [
        ("Label1", LABEL1),
        ("Label2", LABEL2),
        ("Label3", LABEL3),
    ]

    compares = "compare_L1_L2_L3"
    base_out_dir = os.path.join(f"models/{MODEL}", f"{NUM_SNAPSHOTS}_{compares}_{SEED}")
    os.makedirs(base_out_dir, exist_ok=True)

    ckpt, T, grid_size, cond_dim, vocab = load_checkpoint(CKPT_PATH, DEVICE)
    print(f"Loaded: {CKPT_PATH}")
    print(f"T={T}, grid_size(ckpt)={grid_size}, cond_dim={cond_dim}, vocab_size={len(vocab) if vocab else None}")

    # Force tile size
    grid_size = TILE_SIZE
    steps = min(DDIM_STEPS, T)

    model = build_model(cond_dim, BASE_CH, TIME_DIM, DEVICE, ckpt["model_state"])
    sched = make_linear_schedule(T, device=DEVICE)

    # Shared initial x_T for ALL label conditions (same seed & fixed SV)
    x_T = make_fixed_sv_init(
        batch=1,
        H=grid_size,
        W=grid_size,
        seed=SEED,
        fixed_s=FIXED_SATURATION,
        fixed_v=FIXED_VALUE,
        device=DEVICE,
    )
    """
    x_T = make_uniform_fixed_sv_init(
        batch=1,
        H=grid_size,
        W=grid_size,
        seed=SEED,
        fixed_s=FIXED_SATURATION,
        fixed_v=FIXED_VALUE,
        device=DEVICE,
    )
    """
    # Snapshot indices (for saving PNGs)
    save_steps = get_evenly_spaced_steps(NUM_SNAPSHOTS, steps)
    print("Snapshot indices:", save_steps)

    # For charts, we need ALL steps (0..steps-1)
    all_steps = list(range(steps))

    # Run trajectories
    trajs_full = {}  # name -> list[tensor] length steps
    for name, labs in label_sets:
        labels_tag = sanitize_for_path("__".join(labs))
        out_dir = os.path.join(base_out_dir, labels_tag)
        os.makedirs(out_dir, exist_ok=True)

        cond = labels_to_cond(labs, vocab, cond_dim, DEVICE)

        # IMPORTANT: save_steps=all_steps -> returns every step
        traj_full = ddim_sample(
            model=model,
            sched=sched,
            cond=cond,
            shape=(1, 3, grid_size, grid_size),
            seed=SEED,
            steps=steps,
            eta=ETA,
            device=DEVICE,
            x_T=x_T,
            save_steps=all_steps,
        )
        trajs_full[name] = traj_full
        print(f"{name}: got {len(traj_full)} steps")

        # Save snapshot PNGs
        for step_idx in save_steps:
            x = traj_full[step_idx]
            save_path = os.path.join(out_dir, f"{step_idx:04d}.png")
            save_snapshot_image(x, save_path, clip01=CLIP_TO_01_FOR_DISPLAY)

        # Save a sheet for snapshots (only snapshots, not all 2000)
        snaps = [traj_full[k] for k in save_steps]
        n = len(snaps)
        per_row = 10
        rows = math.ceil(n / per_row)
        cols = min(n, per_row)

        plt.figure(figsize=(2.2 * cols, 2.2 * rows))
        for i, (step_idx, x) in enumerate(zip(save_steps, snaps)):
            img = tensor_to_image(x)
            if CLIP_TO_01_FOR_DISPLAY:
                img = np.clip(img, 0.0, 1.0)
            ax = plt.subplot(rows, cols, i + 1)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(str(step_idx))

        total_slots = rows * cols
        for j in range(n, total_slots):
            ax = plt.subplot(rows, cols, j + 1)
            ax.axis("off")

        plt.suptitle(f"{MODEL} | {name} | seed={SEED} | steps={steps}", y=1.02)
        plt.tight_layout()
        sheet_path = os.path.join(out_dir, "sheet.png")
        plt.savefig(sheet_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
        plt.close()

    # ---- Build hue plots (9 charts, each overlays 3 trajectories) ----
    hue1 = compute_hue_series(trajs_full["Label1"], device="cpu")
    hue2 = compute_hue_series(trajs_full["Label2"], device="cpu")
    hue3 = compute_hue_series(trajs_full["Label3"], device="cpu")

    hue_path = os.path.join(base_out_dir, "hue_tiles.png")
    plot_hue_tiles(hue1, hue2, hue3, hue_path, steps)
    print("Saved:", hue_path)

    # ---- Distance plots: Label3 vs Label1/Label2 ----
    dist31_tiles, dist31_all = compute_rgb_l2_tilewise(trajs_full["Label3"], trajs_full["Label1"])
    dist32_tiles, dist32_all = compute_rgb_l2_tilewise(trajs_full["Label3"], trajs_full["Label2"])
    # ---- Ratio charts: (Label3-Label1)/(Label3-Label2) ----
    ratio_tiles_path = os.path.join(base_out_dir, "ratio_tiles_31_over_32.png")
    plot_ratio_tiles(dist31_tiles, dist32_tiles, ratio_tiles_path, steps)

    ratio_all_path = os.path.join(base_out_dir, "ratio_overall_31_over_32.png")
    plot_ratio_overall(dist31_all, dist32_all, ratio_all_path, steps)
    
    dist_tiles_path = os.path.join(base_out_dir, "dist_tiles_label3_vs_12.png")
    plot_dist_tiles(dist31_tiles, dist32_tiles, dist_tiles_path, steps)
    print("Saved:", dist_tiles_path)

    dist_all_path = os.path.join(base_out_dir, "dist_overall_label3_vs_12.png")
    plot_overall_dist(dist31_all, dist32_all, dist_all_path, steps)
    print("Saved:", dist_all_path)

if __name__ == "__main__":
    main()   # your original
    #main2()    # new comparison runner