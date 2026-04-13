import csv
import json
import os
import random
from dataclasses import asdict
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Iterable, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Reuse your baseline diffusion utilities + model definition.
import x3pixel_DM_training as base


# ============================================================
# Repro helpers
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


# ============================================================
# LoRA (Conv2d) – minimal, matches your CondEpsModel trunk
# ============================================================
class LoRAConv2d(nn.Module):
    """LoRA wrapper for Conv2d.

    y = base(x) + scale * lora_up(lora_down(x))
      - lora_down: 1x1 conv (in_ch -> r)
      - lora_up:   same kernel/stride/pad as base (r -> out_ch)

    Initialization sets lora_up = 0 so LoRA starts as an exact no-op.
    """

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
            self.r,
            out_ch,
            kernel_size=k,
            stride=s,
            padding=p,
            dilation=d,
            groups=g,
            bias=False,
        )

        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)  # starts as NO-OP

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * self.lora_up(self.lora_down(x))


def inject_lora_into_selected_convs(
    model: nn.Module,
    r: int = 4,
    alpha: float = 4.0,
    target_names: Optional[Iterable[str]] = None,
):
    """Replace selected nn.Conv2d modules with LoRAConv2d.

    target_names are matched against the *full* names from model.named_modules().
    For your CondEpsModel, typical names are:
      - in_conv
      - block1.2, block2.2, block3.2
      - out_conv
    """
    if target_names is None:
        target_names = {"out_conv"}
    else:
        target_names = set(target_names)

    for full_name, module in list(model.named_modules()):
        if full_name in target_names and isinstance(module, nn.Conv2d):
            parent = model
            parts = full_name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], LoRAConv2d(module, r=r, alpha=alpha))


def freeze_base_only_train_lora(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, LoRAConv2d):
            for p in m.lora_down.parameters():
                p.requires_grad = True
            for p in m.lora_up.parameters():
                p.requires_grad = True


def lora_parameters(model: nn.Module):
    for m in model.modules():
        if isinstance(m, LoRAConv2d):
            yield from m.lora_down.parameters()
            yield from m.lora_up.parameters()


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    sd: Dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        if "lora_down" in k or "lora_up" in k:
            sd[k] = v.detach().cpu()
    return sd


# ============================================================
# Subset selection: build active label sets once
# ============================================================
def _vocab_to_index(vocab: Any) -> Dict[str, int]:
    if vocab is None:
        raise ValueError(
            "Dataset has no vocab. To sample by label strings, ColorGridDataset must expose ds.vocab (list or dict)."
        )
    if isinstance(vocab, dict):
        return {str(k): int(v) for k, v in vocab.items()}
    if isinstance(vocab, (list, tuple)):
        return {str(lbl): i for i, lbl in enumerate(vocab)}
    raise TypeError(f"Unsupported vocab type: {type(vocab)}")


def _active_label_set(cond: torch.Tensor, idx_to_label: List[str]) -> List[str]:
    active_idx = (cond > 0.5).nonzero(as_tuple=False).view(-1).tolist()
    return [idx_to_label[i] for i in active_idx if 0 <= i < len(idx_to_label)]


def _build_active_sets(ds) -> List[set]:
    vocab = getattr(ds, "vocab", None)
    label_to_idx = _vocab_to_index(vocab)

    idx_to_label = [None] * (max(label_to_idx.values()) + 1)
    for lbl, i in label_to_idx.items():
        if 0 <= i < len(idx_to_label):
            idx_to_label[i] = lbl
    for i in range(len(idx_to_label)):
        if idx_to_label[i] is None:
            idx_to_label[i] = f"<unk_{i}>"

    active_sets: List[set] = []
    for i in range(len(ds)):
        cond = ds[i][1]
        if not torch.is_tensor(cond):
            cond = torch.as_tensor(cond)
        labels = _active_label_set(cond, idx_to_label)
        active_sets.append(set(labels))
    return active_sets


def _sample_k(cands: List[int], k: int) -> List[int]:
    if k <= 0:
        return []
    if k >= len(cands):
        return list(cands)
    return random.sample(cands, k)


def _subset_random(pool: List[int], num: int) -> List[int]:
    return _sample_k(pool, num)


def _subset_single(active_sets: List[set], pool: List[int], num: int, label: str) -> Tuple[List[int], int]:
    cands = [i for i in pool if label in active_sets[i]]
    return _sample_k(cands, num), len(cands)


def _subset_any(active_sets: List[set], pool: List[int], num: int, labels: List[str]) -> Tuple[List[int], int]:
    labs = set(labels)
    cands = [i for i in pool if len(active_sets[i].intersection(labs)) > 0]
    return _sample_k(cands, num), len(cands)


def _subset_all(active_sets: List[set], pool: List[int], num: int, labels: List[str]) -> Tuple[List[int], int]:
    labs = set(labels)
    cands = [i for i in pool if labs.issubset(active_sets[i])]
    return _sample_k(cands, num), len(cands)


def select_subset_plan(ds_tmp, config: Dict[str, Any]) -> Tuple[List[int], List[Dict[str, Any]]]:
    """
    从多个子集合组件拼成一个大 subset。

    config 必须包含：
      - subset_size: int (最终想要的总大小)
      - subset_seed: int
      - subset_disjoint: bool  (各组件之间是否不重叠，推荐 True)
      - subset_fill_random: bool (不够时是否补随机，推荐 True)
      - subset_plan: List[dict]  (组件列表)

    subset_plan 每个 dict 支持：
      {"name":"A", "mode":"single", "num":35,  "label":"background_yellow"}
      {"name":"B", "mode":"any",    "num":35,  "labels":["shape_emptiness","background_yellow"]}
      {"name":"C", "mode":"all",    "num":315, "labels":["shape_c","background_yellow"]}
      {"name":"R", "mode":"random", "num":50}
    """
    subset_size = int(config.get("subset_size", 0))
    seed = int(config.get("subset_seed", 0))
    disjoint = bool(config.get("subset_disjoint", True))
    fill_random = bool(config.get("subset_fill_random", True))
    plan = config.get("subset_plan", None)

    if subset_size <= 0:
        raise ValueError("CONFIG['subset_size'] must be > 0")
    if not isinstance(plan, list) or len(plan) == 0:
        raise ValueError("CONFIG['subset_plan'] must be a non-empty list")

    set_seed(seed)

    # precompute active labels once
    active_sets = _build_active_sets(ds_tmp)

    remaining = list(range(len(ds_tmp)))
    selected: List[int] = []
    report: List[Dict[str, Any]] = []

    def _take_from_pool(picked: List[int]):
        if not disjoint:
            return
        picked_set = set(picked)
        nonlocal remaining
        remaining = [i for i in remaining if i not in picked_set]

    for idx, item in enumerate(plan):
        if not isinstance(item, dict):
            raise ValueError(f"subset_plan[{idx}] must be a dict")

        name = str(item.get("name", f"part{idx}"))
        mode = str(item.get("mode", "random")).lower()
        num = int(item.get("num", 0))
        if num <= 0:
            report.append(dict(name=name, mode=mode, requested=num, picked=0, note="requested<=0, skipped"))
            continue

        pool = remaining if disjoint else list(range(len(ds_tmp)))

        if mode == "random":
            picked = _subset_random(pool, num)
            cand_n = len(pool)

        elif mode == "single":
            label = item.get("label", None)
            if not label:
                raise ValueError(f"[{name}] mode='single' requires key 'label'")
            picked, cand_n = _subset_single(active_sets, pool, num, str(label))

        elif mode == "any":
            labels = item.get("labels", None)
            if not labels:
                raise ValueError(f"[{name}] mode='any' requires key 'labels' (list of strings)")
            picked, cand_n = _subset_any(active_sets, pool, num, list(labels))

        elif mode == "all":
            labels = item.get("labels", None)
            if not labels:
                raise ValueError(f"[{name}] mode='all' requires key 'labels' (list of strings)")
            picked, cand_n = _subset_all(active_sets, pool, num, list(labels))

        else:
            raise ValueError(f"[{name}] Unknown mode={mode}. Use random/single/any/all")

        selected.extend(picked)
        _take_from_pool(picked)

        report.append(
            dict(
                name=name,
                mode=mode,
                requested=num,
                candidates_in_pool=cand_n,
                picked=len(picked),
                labels=item.get("labels", None) if mode in ("any", "all") else item.get("label", None),
                disjoint=disjoint,
            )
        )

    # dedup (if disjoint=False, different parts might overlap)
    if len(set(selected)) != len(selected):
        seen = set()
        deduped = []
        for i in selected:
            if i not in seen:
                seen.add(i)
                deduped.append(i)
        selected = deduped

    # fill to subset_size if needed
    if len(selected) < subset_size and fill_random:
        need = subset_size - len(selected)
        pool = remaining if disjoint else [i for i in range(len(ds_tmp)) if i not in set(selected)]
        filler = _sample_k(pool, need)
        selected.extend(filler)
        if disjoint:
            _take_from_pool(filler)
        report.append(
            dict(name="__fill_random__", mode="random", requested=need, candidates_in_pool=len(pool), picked=len(filler))
        )

    # truncate if too long
    if len(selected) > subset_size:
        selected = selected[:subset_size]
        report.append(dict(name="__truncate__", mode="info", note=f"truncated to subset_size={subset_size}"))

    # warning if still short
    if len(selected) < subset_size:
        report.append(
            dict(
                name="__warning__",
                mode="warning",
                requested=subset_size,
                picked=len(selected),
                note="Dataset/pool too small; cannot reach subset_size even after fill.",
            )
        )

    return selected, report


# ============================================================
# Checkpoint helpers
# ============================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def latest_checkpoint_in_dir(dir_path: str, suffix: str = ".pt") -> Optional[str]:
    if not os.path.isdir(dir_path):
        return None
    files = [f for f in os.listdir(dir_path) if f.endswith(suffix)]
    if not files:
        return None
    files.sort()
    return os.path.join(dir_path, files[-1])


def next_indexed_path(dir_path: str, prefix: str, ext: str = ".pt") -> str:
    ensure_dir(dir_path)
    i = 1
    while True:
        path = os.path.join(dir_path, f"{prefix}_{i:04d}{ext}")
        if not os.path.exists(path):
            return path
        i += 1


def save_subset_csv(path: str, indices: List[int]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx"])
        for i in indices:
            w.writerow([int(i)])


def save_report_json(path: str, report: List[Dict[str, Any]], extra: Optional[Dict[str, Any]] = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "report": report,
        "extra": extra or {},
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


# ============================================================
# Training: baseline and LoRA fine-tune
# ============================================================
def train_baseline(
    csv_path: str,
    grid_size: int,
    batch_size: int,
    epochs: int,
    lr: float,
    T: int,
    seed: int,
    sample_every: int,
    ddim_steps: int,
    device: str,
    base_ch: int,
    time_dim: int,
    out_dir: str,
    save_every_epochs: int,
):
    set_seed(seed)

    from dataset_loader import ColorGridDataset

    if "grid_size" in ColorGridDataset.__init__.__code__.co_varnames:
        ds = ColorGridDataset(csv_path, grid_size=grid_size)
    else:
        ds = ColorGridDataset(csv_path)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    x0, cond0 = ds[0][:2]
    C, H, W = x0.shape
    cond_dim = int(cond0.numel())
    print(f"Dataset: N={len(ds)}, image=(C,H,W)=({C},{H},{W}), cond_dim={cond_dim}")

    sched = base.make_linear_schedule(T, device=device)
    model = base.CondEpsModel(in_ch=C, cond_dim=cond_dim, base_ch=base_ch, time_dim=time_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    ensure_dir(out_dir)
    meta = {
        "csv_path": csv_path,
        "grid_size": grid_size,
        "dataset_size": len(ds),
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "T": T,
        "seed": seed,
        "base_ch": base_ch,
        "time_dim": time_dim,
        "device": device,
        "cond_dim": cond_dim,
        "vocab": getattr(ds, "vocab", None),
    }
    with open(os.path.join(out_dir, "baseline_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        for x0b, condb in loader:
            x0b = x0b.to(device)
            condb = condb.to(device)
            B = x0b.size(0)
            t = torch.randint(0, sched.T, (B,), device=device, dtype=torch.long)
            noise = torch.randn_like(x0b)
            xt = base.q_sample(x0b, t, noise, sched)

            eps_pred = model(xt, t, condb)
            loss = F.mse_loss(eps_pred, noise)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            losses.append(float(loss.item()))
            global_step += 1

        avg_loss = float(np.mean(losses)) if losses else float("nan")
        print(f"[baseline] epoch {epoch:03d}/{epochs} loss={avg_loss:.6f}")

        if (epoch % save_every_epochs) == 0 or epoch == epochs:
            ckpt = {
                "model_state": model.state_dict(),
                "T": T,
                "grid_size": grid_size,
                "cond_dim": cond_dim,
                "seed": seed,
                "vocab": getattr(ds, "vocab", None),
                "base_ch": base_ch,
                "time_dim": time_dim,
                "epoch": epoch,
                "global_step": global_step,
            }
            path = next_indexed_path(out_dir, prefix="baseline")
            torch.save(ckpt, path)
            print(f"  saved: {path}")

        if sample_every > 0 and (epoch % sample_every) == 0:
            model.eval()
            _, c_real = ds[random.randrange(len(ds))][:2]
            c_real = c_real.unsqueeze(0).to(device)
            x_gen = base.ddim_sample(
                model=model,
                sched=sched,
                cond=c_real,
                shape=(1, C, H, W),
                seed=seed,
                steps=ddim_steps,
                eta=0.0,
                device=device,
            )[-1]
            print(
                f"  sample stats: min={x_gen.min().item():.3f}, max={x_gen.max().item():.3f}, mean={x_gen.mean().item():.3f}"
            )

    return out_dir


def lora_update_on_subset(
    csv_path: str,
    grid_size: int,
    device: str,
    baseline_ckpt_path: str,
    out_dir: str,
    subset_indices: List[int],
    lora_r: int,
    lora_alpha: float,
    lora_targets: List[str],
    lr: float,
    steps: int,
    batch_size: int,
    save_every_steps: int,
    seed: int,
    init_lora_path: Optional[str] = None,
):
    set_seed(seed)
    ensure_dir(out_dir)

    from dataset_loader import ColorGridDataset

    if "grid_size" in ColorGridDataset.__init__.__code__.co_varnames:
        ds = ColorGridDataset(csv_path, grid_size=grid_size)
    else:
        ds = ColorGridDataset(csv_path)

    ckpt = torch.load(baseline_ckpt_path, map_location=device)

    x0, cond0 = ds[0][:2]
    C, H, W = x0.shape
    cond_dim = int(cond0.numel())

    base_ch = int(ckpt.get("base_ch", 64))
    time_dim = int(ckpt.get("time_dim", 128))
    T = int(ckpt["T"])
    sched = base.make_linear_schedule(T, device=device)

    model = base.CondEpsModel(in_ch=C, cond_dim=cond_dim, base_ch=base_ch, time_dim=time_dim).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)

    inject_lora_into_selected_convs(model, r=lora_r, alpha=lora_alpha, target_names=lora_targets)

    if init_lora_path is not None:
        payload = torch.load(init_lora_path, map_location="cpu")
        lora_sd = payload.get("lora_state", payload)
        model.load_state_dict(lora_sd, strict=False)

    freeze_base_only_train_lora(model)

    from torch.utils.data import SubsetRandomSampler

    sampler = SubsetRandomSampler(subset_indices)
    loader = DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=0, drop_last=True)

    opt = torch.optim.AdamW(list(lora_parameters(model)), lr=lr)

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "csv_path": csv_path,
        "grid_size": grid_size,
        "dataset_size": len(ds),
        "subset_size": len(subset_indices),
        "baseline_ckpt": baseline_ckpt_path,
        "init_lora_path": init_lora_path,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_targets": list(lora_targets),
        "lr": lr,
        "steps": steps,
        "batch_size": batch_size,
        "save_every_steps": save_every_steps,
        "seed": seed,
    }
    with open(os.path.join(out_dir, "lora_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    save_subset_csv(os.path.join(out_dir, "subset.csv"), subset_indices)

    model.train()
    it = iter(loader)
    for step in range(1, steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        x0b, condb = batch[:2]
        x0b = x0b.to(device)
        condb = condb.to(device)
        B = x0b.size(0)
        t = torch.randint(0, sched.T, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(x0b)
        xt = base.q_sample(x0b, t, noise, sched)

        eps_pred = model(xt, t, condb)
        loss = F.mse_loss(eps_pred, noise)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (step % 25) == 0 or step == 1:
            print(f"[lora] step {step:05d}/{steps} loss={float(loss.item()):.6f}")

        if save_every_steps > 0 and ((step % save_every_steps) == 0 or step == steps):
            payload = {
                "lora_state": lora_state_dict(model),
                "step": step,
                "baseline_ckpt": baseline_ckpt_path,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_targets": list(lora_targets),
            }
            path = next_indexed_path(out_dir, prefix="lora")
            torch.save(payload, path)
            print(f"  saved: {path}")

    return out_dir


# ============================================================
# FILE CONFIG (no argparse)
# ============================================================

#MODE = "baseline"  # "baseline" or "lora"
MODE = "baseline" 

CONFIG = dict(
    # data / runtime
    csv_path="generated_database/RBY/dataset_3_base_plus_red.csv",
    grid_size=3,
    device="cpu",
    seed=0,

    # baseline training
    batch_size=256,
    epochs=350,
    lr=2e-4,
    T=2000,
    base_ch=64,
    time_dim=128,
    sample_every=10,
    ddim_steps=50,
    save_every_epochs=1,

    # directory layout
    root_dir="models_checkpoints/r",
    model_name=None,  # None -> auto model_{dataset_size}

    # -------- subset plan (for lora) --------
    subset_size=10000,            # final combined size
    subset_seed=123,
    subset_disjoint=True,       # parts do not overlap (recommended)
    subset_fill_random=True,    # if parts sum < subset_size, auto fill random

    # Each part contributes num items.
    # mode: "single" | "any" | "all" | "random"
    subset_plan=[
        dict(name="part1_ab_any", mode="any", num=10000, labels=["shape_color_red","shape_color_blue",]),
        #dict(name="part2_cb_all", mode="all", num=315, labels=["shape_c", "background_yellow"]),
        # dict(name="extra_random", mode="random", num=10),
        # dict(name="only_shape_purple", mode="single", num=10000, label="shape_color_purple"),
    ],

    # lora update params
    baseline_ckpt=None,        # None -> auto pick latest in baseline_dir
    init_lora=None,            # None -> start from scratch
    lora_r=4,
    lora_alpha=4.0,
    lora_targets=["out_conv"],  # e.g. ["out_conv", "block3.2"]
    lora_lr=1e-4,
    lora_steps=8000,
    lora_batch_size=128,
    save_every_steps=50,
    lora_tag="N10000_bredorblue10000",  # folder tag; None -> auto
)


def build_ds_and_paths(CONFIG):
    from dataset_loader import ColorGridDataset
    csv_path = CONFIG["csv_path"]

    if "grid_size" in ColorGridDataset.__init__.__code__.co_varnames:
        ds_tmp = ColorGridDataset(csv_path, grid_size=CONFIG["grid_size"])
    else:
        ds_tmp = ColorGridDataset(csv_path)

    dataset_size = len(ds_tmp)
    model_folder = CONFIG["model_name"] or f"model_{dataset_size}"
    model_root = os.path.join(CONFIG["root_dir"], model_folder)
    baseline_dir = os.path.join(model_root, "baseline")
    return ds_tmp, model_root, baseline_dir


def run_baseline(CONFIG, baseline_dir: str):
    train_baseline(
        csv_path=CONFIG["csv_path"],
        grid_size=CONFIG["grid_size"],
        batch_size=CONFIG["batch_size"],
        epochs=CONFIG["epochs"],
        lr=CONFIG["lr"],
        T=CONFIG["T"],
        seed=CONFIG["seed"],
        sample_every=CONFIG["sample_every"],
        ddim_steps=CONFIG["ddim_steps"],
        device=CONFIG["device"],
        base_ch=CONFIG["base_ch"],
        time_dim=CONFIG["time_dim"],
        out_dir=baseline_dir,
        save_every_epochs=CONFIG["save_every_epochs"],
    )


def run_lora_update(CONFIG, ds_tmp, model_root: str, baseline_dir: str):
    # ---- subset selection (combined plan) ----
    selected, report = select_subset_plan(ds_tmp, CONFIG)

    print("\n[subset] summary")
    for r in report:
        print(r)
    print(f"[subset] final selected: {len(selected)}\n")

    # ---- choose baseline checkpoint ----
    baseline_ckpt = CONFIG["baseline_ckpt"]
    if baseline_ckpt is None:
        baseline_ckpt = latest_checkpoint_in_dir(baseline_dir)
    if baseline_ckpt is None or (not os.path.exists(baseline_ckpt)):
        raise FileNotFoundError(
            "No baseline checkpoint found. Run MODE='baseline' first, or set CONFIG['baseline_ckpt']."
        )

    # ---- tag + output dir ----
    tag = CONFIG.get("lora_tag", None)
    if tag is None:
        tag = f"N{CONFIG['subset_size']}_plan"

    lora_dir = os.path.join(model_root, f"lora_update_{tag}")
    ensure_dir(lora_dir)

    # ---- save report + subset indices ----
    save_report_json(
        os.path.join(lora_dir, "subset_report.json"),
        report,
        extra={
            "subset_size": CONFIG["subset_size"],
            "subset_seed": CONFIG["subset_seed"],
            "subset_disjoint": CONFIG["subset_disjoint"],
            "subset_fill_random": CONFIG["subset_fill_random"],
            "subset_plan": CONFIG["subset_plan"],
        },
    )
    save_subset_csv(os.path.join(lora_dir, "subset.csv"), selected)

    # ---- run LoRA update ----
    lora_update_on_subset(
        csv_path=CONFIG["csv_path"],
        grid_size=CONFIG["grid_size"],
        device=CONFIG["device"],
        baseline_ckpt_path=baseline_ckpt,
        out_dir=lora_dir,
        subset_indices=selected,
        lora_r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_targets=CONFIG["lora_targets"],
        lr=CONFIG["lora_lr"],
        steps=CONFIG["lora_steps"],
        batch_size=CONFIG["lora_batch_size"],
        save_every_steps=CONFIG["save_every_steps"],
        seed=CONFIG["seed"],
        init_lora_path=CONFIG["init_lora"],
    )


def main():
    ds_tmp, model_root, baseline_dir = build_ds_and_paths(CONFIG)

    if MODE == "baseline":
        run_baseline(CONFIG, baseline_dir)
    elif MODE == "lora":
        run_lora_update(CONFIG, ds_tmp, model_root, baseline_dir)
    else:
        raise ValueError(f"Unknown MODE={MODE}")


if __name__ == "__main__":
    main()
