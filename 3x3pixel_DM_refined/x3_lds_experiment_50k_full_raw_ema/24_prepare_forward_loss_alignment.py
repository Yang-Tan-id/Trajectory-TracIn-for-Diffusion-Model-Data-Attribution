"""Prepare exact training-event replay, full trajectories, and baseline losses."""

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import x3pixel_DM_training as base
from dataset_loader import ColorGridDataset
from forward_loss_alignment_config import *
from train_worker import set_seed


class IndexDataset(Dataset):
    """Dataset with the same length and deterministic fetch count as training."""

    def __len__(self):
        return N_TRAIN

    def __getitem__(self, index):
        return int(index)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu")


def make_model(ds, state, device):
    x, cond = ds[0]
    model = base.CondEpsModel(
        in_ch=int(x.shape[0]), cond_dim=int(cond.numel()),
        base_ch=BASE_CH, time_dim=TIME_DIM,
    )
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def dataset_arrays(ds):
    images = np.empty((len(ds), 3, 3, 3), dtype=np.float32)
    conds = np.empty((len(ds), len(ds.vocab)), dtype=np.float32)
    for index in range(len(ds)):
        image, cond = ds[index]
        images[index] = image.numpy()
        conds[index] = cond.numpy()
    return images, conds


def prepare_training_event_replay(device):
    FLA_REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    t_path = replay_t_path()
    noise_path = replay_noise_path()
    meta_path = FLA_REPLAY_DIR / "metadata.json"
    if t_path.is_file() and noise_path.is_file() and meta_path.is_file():
        print(f"[replay] existing cache found: {FLA_REPLAY_DIR}", flush=True)
        return

    # Match train_worker ordering exactly: seed -> DataLoader construction ->
    # CPU model initialization -> CUDA model/EMA/schedule/optimizer -> first iter.
    set_seed(TRAIN_SEED)
    ds = ColorGridDataset(str(BASE_CSV), grid_size=3)
    loader = DataLoader(
        IndexDataset(), batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        drop_last=False, pin_memory=torch.cuda.is_available(),
    )
    x, cond = ds[0]
    model = base.CondEpsModel(
        in_ch=int(x.shape[0]), cond_dim=int(cond.numel()),
        base_ch=BASE_CH, time_dim=TIME_DIM,
    ).to(device)
    ema = copy.deepcopy(model).to(device).eval()
    sched = base.make_linear_schedule(T, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=PEAK_LR, betas=(ADAM_B1, ADAM_B2),
        eps=ADAM_EPS, weight_decay=WEIGHT_DECAY,
    )
    del ema, sched, optimizer

    t_cache = np.lib.format.open_memmap(
        t_path, mode="w+", dtype=np.int16,
        shape=(len(FLA_CHECKPOINT_EPOCHS), N_TRAIN, FLA_EVENTS_PER_CHECKPOINT),
    )
    noise_cache = np.lib.format.open_memmap(
        noise_path, mode="w+", dtype=np.float32,
        shape=(len(FLA_CHECKPOINT_EPOCHS), N_TRAIN, FLA_EVENTS_PER_CHECKPOINT, 3, 3, 3),
    )
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        checkpoint_index = (epoch - 1) // FLA_EVENTS_PER_CHECKPOINT
        event_index = (epoch - 1) % FLA_EVENTS_PER_CHECKPOINT
        seen = 0
        for indices in loader:
            indices_np = indices.numpy().astype(np.int64, copy=False)
            batch_size = len(indices_np)
            # These are the same two CUDA RNG calls, in the same order and
            # batch sizes, as train_worker.py.
            t = torch.randint(0, T, (batch_size,), device=device, dtype=torch.long)
            noise = torch.randn((batch_size, 3, 3, 3), device=device)
            t_cache[checkpoint_index, indices_np, event_index] = t.cpu().numpy().astype(np.int16)
            noise_cache[checkpoint_index, indices_np, event_index] = noise.cpu().numpy()
            seen += batch_size
        if seen != N_TRAIN:
            raise RuntimeError(f"epoch {epoch}: replay saw {seen}, expected {N_TRAIN}")
        if epoch % FLA_EVENTS_PER_CHECKPOINT == 0:
            t_cache.flush()
            noise_cache.flush()
            elapsed = time.perf_counter() - started
            print(
                f"[replay] checkpoint={checkpoint_index + 1:02d}/{len(FLA_CHECKPOINT_EPOCHS)} "
                f"epochs={epoch - 3:03d}-{epoch:03d} elapsed={elapsed / 60:.1f}m",
                flush=True,
            )
    with open(meta_path, "w") as handle:
        json.dump(
            {
                "train_seed": TRAIN_SEED,
                "n_train": N_TRAIN,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "events_per_checkpoint": FLA_EVENTS_PER_CHECKPOINT,
                "checkpoint_epochs": list(FLA_CHECKPOINT_EPOCHS),
                "note": "Replays DataLoader shuffle and CUDA randint/randn calls from train_worker.py.",
            },
            handle,
            indent=2,
        )
    print(f"[replay] saved exact event cache -> {FLA_REPLAY_DIR}", flush=True)


@torch.no_grad()
def prepare_reference_queries(device):
    FLA_QUERY_DIR.mkdir(parents=True, exist_ok=True)
    ds = ColorGridDataset(str(BASE_CSV), grid_size=3)
    final = load_checkpoint(checkpoint_path(EPOCHS))
    reference_model = make_model(ds, final["ema_model_state"], device)
    schedule = base.make_linear_schedule(T, device=device)
    with open(QUERY_DIR / "manifest.json") as handle:
        records = {int(item["query_id"]): item for item in json.load(handle)}
    ts = torch.linspace(T - 1, 0, DDIM_STEPS, device=device).long()
    save_steps = list(range(DDIM_STEPS))
    sparse_steps = np.linspace(0, DDIM_STEPS - 1, TRAJ_SNAPSHOTS, dtype=np.int64)

    for position, qid in enumerate(FLA_QUERY_IDS, start=1):
        output_dir = FLA_QUERY_DIR / f"q{qid:02d}"
        xt_path = output_dir / "trajectory_xt_1000.npy"
        target_path = output_dir / "target_eps_ema_1000.npy"
        timestamp_path = output_dir / "trajectory_t_1000.npy"
        query_meta_path = output_dir / "metadata.json"
        if all(path.is_file() for path in (xt_path, target_path, timestamp_path, query_meta_path)):
            print(f"[query] q{qid:02d} existing ({position}/{len(FLA_QUERY_IDS)})", flush=True)
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        record = records[qid]
        cond = torch.zeros((1, len(ds.vocab)), dtype=torch.float32, device=device)
        for label in record["labels"]:
            cond[0, ds.vocab[label]] = 1.0
        sparse = np.load(QUERY_DIR / f"q{qid:02d}" / "trajectory_xt.npy")
        initial = torch.from_numpy(sparse[0]).to(device)
        trajectory = base.ddim_sample(
            model=reference_model, sched=schedule, cond=cond,
            shape=(1, 3, 3, 3), seed=int(record["initial_seed"]),
            steps=DDIM_STEPS, eta=0.0, device=str(device),
            x_T=initial, save_steps=save_steps,
        )
        xt = torch.stack(trajectory, dim=0)
        replay_sparse = xt[sparse_steps].cpu().numpy()
        max_error = float(np.max(np.abs(replay_sparse - sparse)))
        if max_error > 2e-5:
            raise RuntimeError(f"q{qid:02d}: full trajectory does not match cache; max error={max_error}")
        targets = []
        for start in range(0, DDIM_STEPS, FLA_QUERY_BATCH_SIZE):
            end = min(start + FLA_QUERY_BATCH_SIZE, DDIM_STEPS)
            xb = xt[start:end, 0]
            tb = ts[start:end]
            cb = cond.expand(end - start, -1)
            targets.append(reference_model(xb, tb, cb).cpu())
        np.save(xt_path, xt[:, 0].cpu().numpy())
        np.save(target_path, torch.cat(targets, dim=0).numpy())
        np.save(timestamp_path, ts.cpu().numpy())
        with open(query_meta_path, "w") as handle:
            json.dump(
                {
                    "query_id": qid,
                    "family": record["family"],
                    "labels": record["labels"],
                    "initial_seed_metadata": record["initial_seed"],
                    "initial_state_source": str(QUERY_DIR / f"q{qid:02d}" / "trajectory_xt.npy"),
                    "reference_parameter_source": FLA_REFERENCE_PARAM_SOURCE,
                    "sparse_replay_max_abs_error": max_error,
                },
                handle,
                indent=2,
            )
        print(
            f"[query] q{qid:02d} saved full trajectory + targets "
            f"({position}/{len(FLA_QUERY_IDS)}) max_sparse_error={max_error:.3e}",
            flush=True,
        )


@torch.no_grad()
def prepare_baseline_losses(device):
    FLA_BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = baseline_path()
    done_path = FLA_BASELINE_DIR / "done.json"
    if output_path.is_file() and done_path.is_file():
        print(f"[baseline] existing cache found: {output_path}", flush=True)
        return
    if not replay_t_path().is_file() or not replay_noise_path().is_file():
        raise FileNotFoundError("Run training-event replay preparation first")
    ds = ColorGridDataset(str(BASE_CSV), grid_size=3)
    images, conds = dataset_arrays(ds)
    t_cache = np.load(replay_t_path(), mmap_mode="r")
    noise_cache = np.load(replay_noise_path(), mmap_mode="r")
    temporary_path = FLA_BASELINE_DIR / f"{FLA_FAMILY}_raw.tmp.npy"
    output = np.lib.format.open_memmap(
        temporary_path, mode="w+", dtype=np.float64,
        shape=(len(FLA_CHECKPOINT_EPOCHS), N_TRAIN),
    )
    schedule = base.make_linear_schedule(T, device=device)
    started = time.perf_counter()
    for checkpoint_index, epoch in enumerate(FLA_CHECKPOINT_EPOCHS):
        checkpoint = load_checkpoint(checkpoint_path(epoch))
        model = make_model(ds, checkpoint["model_state"], device)
        for start in range(0, N_TRAIN, FLA_DATAPOINT_BATCH_SIZE):
            end = min(start + FLA_DATAPOINT_BATCH_SIZE, N_TRAIN)
            count = end - start
            x0 = torch.from_numpy(images[start:end]).to(device)
            cond = torch.from_numpy(conds[start:end]).to(device)
            t = torch.from_numpy(np.asarray(t_cache[checkpoint_index, start:end], dtype=np.int64)).to(device)
            noise = torch.from_numpy(np.asarray(noise_cache[checkpoint_index, start:end])).to(device)
            x0 = x0[:, None].expand(-1, FLA_EVENTS_PER_CHECKPOINT, -1, -1, -1).reshape(-1, 3, 3, 3)
            cond = cond[:, None].expand(-1, FLA_EVENTS_PER_CHECKPOINT, -1).reshape(-1, cond.shape[-1])
            t = t.reshape(-1)
            noise = noise.reshape(-1, 3, 3, 3)
            xt = base.q_sample(x0, t, noise, schedule)
            pred = model(xt, t, cond)
            # Keep model arithmetic in its native float32, but do the loss
            # reduction in float64.  The one-step decrease can be ~1e-8.
            event_loss = (pred.double() - noise.double()).square().flatten(1).mean(1)
            output[checkpoint_index, start:end] = event_loss.reshape(count, FLA_EVENTS_PER_CHECKPOINT).mean(1).cpu().numpy()
        output.flush()
        elapsed = time.perf_counter() - started
        print(
            f"[baseline] checkpoint={checkpoint_index + 1:02d}/{len(FLA_CHECKPOINT_EPOCHS)} "
            f"epoch={epoch:03d} elapsed={elapsed / 60:.1f}m",
            flush=True,
        )
        del model
        torch.cuda.empty_cache()
    del output
    os.replace(temporary_path, output_path)
    with open(done_path, "w") as handle:
        json.dump(
            {
                "family": FLA_FAMILY,
                "parameter_source": FLA_CHECKPOINT_PARAM_SOURCE,
                "shape": [len(FLA_CHECKPOINT_EPOCHS), N_TRAIN],
                "dtype": "float64",
                "events_per_checkpoint": FLA_EVENTS_PER_CHECKPOINT,
            },
            handle,
            indent=2,
        )
    print(f"[baseline] saved -> {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip-replay", action="store_true")
    parser.add_argument("--skip-queries", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for exact replay of the original CUDA RNG stream")
    device = torch.device(f"cuda:{args.gpu}")
    print(f"[device] {device}", flush=True)
    if not args.skip_replay:
        prepare_training_event_replay(device)
    if not args.skip_queries:
        prepare_reference_queries(device)
    if not args.skip_baseline:
        prepare_baseline_losses(device)
    print("[done] forward-loss-alignment preparation complete", flush=True)


if __name__ == "__main__":
    main()
