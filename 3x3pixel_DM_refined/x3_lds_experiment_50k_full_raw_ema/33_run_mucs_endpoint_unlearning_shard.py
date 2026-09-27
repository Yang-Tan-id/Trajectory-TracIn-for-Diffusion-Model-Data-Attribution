"""Run endpoint-MC100 AdamW unlearning and MUCS scoring for a query shard."""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import x3pixel_DM_training as base
from dataset_loader import ColorGridDataset
from mucs_endpoint_unlearning_config import *


def atomic_torch_save(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def load_checkpoint(epoch):
    return torch.load(
        mucs_checkpoint_path(epoch),
        map_location="cpu",
        weights_only=False,
    )


def make_model(dataset, state, device):
    model = base.CondEpsModel(
        in_ch=3,
        cond_dim=len(dataset.vocab),
        base_ch=BASE_CH,
        time_dim=TIME_DIM,
    ).to(device)
    model.load_state_dict(state, strict=True)
    return model.eval()


def query_condition(query, dataset, device):
    condition = torch.zeros((1, len(dataset.vocab)), dtype=torch.float32, device=device)
    for label in query.get("labels", []):
        condition[0, dataset.vocab[label]] = 1.0
    return condition


def fixed_mc_bank(count, seed, device):
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    timesteps = torch.randint(
        0, T, (count,), generator=generator, device=device, dtype=torch.long
    )
    noises = torch.randn(
        (count, 3, 3, 3), generator=generator, device=device, dtype=torch.float32
    )
    return timesteps, noises


def endpoint_loss(model, endpoint, condition, timesteps, noises, schedule):
    x0 = endpoint.expand(len(timesteps), -1, -1, -1)
    xt = base.q_sample(x0, timesteps, noises, schedule)
    prediction = model(xt, timesteps, condition.expand(len(timesteps), -1))
    return F.mse_loss(prediction, noises)


def unlearn_endpoint(qid, query, dataset, device, schedule, work_dir):
    final_path = work_dir / "unlearned_model.pt"
    resume_path = work_dir / "unlearning_resume.pt"
    final_checkpoint = load_checkpoint(MUCS_INITIAL_EPOCH)
    null_checkpoint = load_checkpoint(MUCS_NULL_EPOCH)
    condition = query_condition(query, dataset, device)
    endpoint = torch.from_numpy(
        np.load(QUERY_DIR / f"q{qid:02d}" / "final_state.npy")
    ).to(device=device, dtype=torch.float32)
    endpoint_t, endpoint_noise = fixed_mc_bank(
        MUCS_ENDPOINT_MC,
        MUCS_ENDPOINT_MC_SEED_BASE + qid,
        device,
    )

    baseline_model = make_model(dataset, final_checkpoint["model_state"], device)
    null_model = make_model(dataset, null_checkpoint["model_state"], device)
    with torch.no_grad():
        initial_loss = float(
            endpoint_loss(
                baseline_model, endpoint, condition,
                endpoint_t, endpoint_noise, schedule,
            ).item()
        )
        null_loss = float(
            endpoint_loss(
                null_model, endpoint, condition,
                endpoint_t, endpoint_noise, schedule,
            ).item()
        )
    del null_model
    if not math.isfinite(initial_loss) or not math.isfinite(null_loss):
        raise RuntimeError(
            f"q{qid:02d}: non-finite endpoint loss final={initial_loss} null={null_loss}"
        )
    if null_loss <= initial_loss:
        raise RuntimeError(
            f"q{qid:02d}: null endpoint loss {null_loss:.8g} is not above "
            f"final loss {initial_loss:.8g}; ascent-to-null criterion is undefined"
        )
    target_loss = initial_loss + MUCS_NULL_GAP_FRACTION * (null_loss - initial_loss)

    if final_path.is_file():
        payload = torch.load(final_path, map_location=device, weights_only=False)
        model = make_model(dataset, payload["model_state"], device)
        print(
            f"[gpu {device.index} q{qid:02d}] reuse unlearned model "
            f"step={payload['steps']} loss={payload['endpoint_loss']:.8g}",
            flush=True,
        )
        return baseline_model, model, payload

    model = make_model(dataset, final_checkpoint["model_state"], device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=MUCS_UNLEARNING_LR,
        betas=(ADAM_B1, ADAM_B2),
        eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY,
    )
    optimizer.load_state_dict(final_checkpoint["optimizer_state"])
    for group in optimizer.param_groups:
        group["lr"] = MUCS_UNLEARNING_LR
        group["weight_decay"] = WEIGHT_DECAY

    start_step = 0
    history = []
    if resume_path.is_file():
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(resume["model_state"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state"])
        for group in optimizer.param_groups:
            group["lr"] = MUCS_UNLEARNING_LR
            group["weight_decay"] = WEIGHT_DECAY
        start_step = int(resume["step"])
        history = list(resume.get("history", []))
        print(f"[gpu {device.index} q{qid:02d}] resume step={start_step}", flush=True)

    started = time.perf_counter()
    reached = False
    current_loss = float("nan")
    for step in range(start_step, MUCS_MAX_UNLEARNING_STEPS + 1):
        model.eval()
        loss = endpoint_loss(
            model, endpoint, condition, endpoint_t, endpoint_noise, schedule
        )
        current_loss = float(loss.detach().item())
        if not math.isfinite(current_loss):
            raise RuntimeError(
                f"q{qid:02d}: endpoint loss became non-finite at step {step}"
            )
        if step == start_step or step % 10 == 0 or current_loss >= target_loss:
            gap_fraction = (current_loss - initial_loss) / (null_loss - initial_loss)
            history.append(
                {
                    "step": step,
                    "endpoint_loss": current_loss,
                    "null_gap_fraction": float(gap_fraction),
                }
            )
            print(
                f"[gpu {device.index} q{qid:02d}] step={step} "
                f"loss={current_loss:.8g} target={target_loss:.8g} "
                f"null={null_loss:.8g} gap={gap_fraction:.3%} "
                f"elapsed={(time.perf_counter()-started)/60:.1f}m",
                flush=True,
            )
        if current_loss >= target_loss:
            reached = True
            break
        if step == MUCS_MAX_UNLEARNING_STEPS:
            break

        optimizer.zero_grad(set_to_none=True)
        (-loss).backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), GRAD_CLIP
        )
        optimizer.step()

        completed_step = step + 1
        if completed_step % MUCS_RESUME_EVERY_STEPS == 0:
            atomic_torch_save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "step": completed_step,
                    "history": history,
                    "last_gradient_norm_before_clip": float(gradient_norm),
                },
                resume_path,
            )

    if not reached:
        raise RuntimeError(
            f"q{qid:02d}: target endpoint loss {target_loss:.8g} not reached "
            f"after {MUCS_MAX_UNLEARNING_STEPS} AdamW ascent steps; "
            f"last={current_loss:.8g}"
        )
    payload = {
        "model_state": model.state_dict(),
        "steps": int(step),
        "endpoint_loss_initial": initial_loss,
        "endpoint_loss_null": null_loss,
        "endpoint_loss_target": target_loss,
        "endpoint_loss": current_loss,
        "null_gap_fraction": float(
            (current_loss - initial_loss) / (null_loss - initial_loss)
        ),
        "unlearning_lr": MUCS_UNLEARNING_LR,
        "optimizer": "AdamW loaded from epoch-200 optimizer state; negative loss gradient",
        "grad_clip_norm": GRAD_CLIP,
        "history": history,
    }
    atomic_torch_save(payload, final_path)
    print(
        f"[gpu {device.index} q{qid:02d}] reached null-gap target at step={step}",
        flush=True,
    )
    return baseline_model, model, payload


@torch.no_grad()
def score_training_points(baseline_model, changed_model, dataset, device, schedule):
    scores = np.empty(N_TRAIN, dtype=np.float64)
    baseline_means = np.empty(N_TRAIN, dtype=np.float64)
    changed_means = np.empty(N_TRAIN, dtype=np.float64)
    generator = torch.Generator(device=device)
    generator.manual_seed(MUCS_TRAIN_MC_SEED)
    started = time.perf_counter()
    for start in range(0, N_TRAIN, MUCS_SCORE_DATAPOINT_BATCH):
        end = min(start + MUCS_SCORE_DATAPOINT_BATCH, N_TRAIN)
        count = end - start
        images = []
        conditions = []
        for index in range(start, end):
            image, condition = dataset[index]
            images.append(image)
            conditions.append(condition)
        x0 = torch.stack(images, dim=0).to(device)
        cond = torch.stack(conditions, dim=0).to(device)
        timesteps = torch.randint(
            0,
            T,
            (count, MUCS_TRAIN_MC),
            generator=generator,
            device=device,
            dtype=torch.long,
        )
        noises = torch.randn(
            (count, MUCS_TRAIN_MC, 3, 3, 3),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        flat_x0 = x0[:, None].expand(
            -1, MUCS_TRAIN_MC, -1, -1, -1
        ).reshape(-1, 3, 3, 3)
        flat_cond = cond[:, None].expand(
            -1, MUCS_TRAIN_MC, -1
        ).reshape(-1, cond.shape[-1])
        flat_t = timesteps.reshape(-1)
        flat_noise = noises.reshape(-1, 3, 3, 3)
        xt = base.q_sample(flat_x0, flat_t, flat_noise, schedule)
        baseline_prediction = baseline_model(xt, flat_t, flat_cond)
        changed_prediction = changed_model(xt, flat_t, flat_cond)
        baseline_loss = (
            baseline_prediction.double() - flat_noise.double()
        ).square().flatten(1).mean(1).reshape(count, MUCS_TRAIN_MC)
        changed_loss = (
            changed_prediction.double() - flat_noise.double()
        ).square().flatten(1).mean(1).reshape(count, MUCS_TRAIN_MC)
        ratio = (changed_loss - baseline_loss) / (
            changed_loss + baseline_loss + MUCS_EPS
        )
        scores[start:end] = ratio.mean(dim=1).cpu().numpy()
        baseline_means[start:end] = baseline_loss.mean(dim=1).cpu().numpy()
        changed_means[start:end] = changed_loss.mean(dim=1).cpu().numpy()
        if start == 0 or end == N_TRAIN or end % 5000 == 0:
            elapsed = time.perf_counter() - started
            eta = elapsed / end * (N_TRAIN - end)
            print(
                f"[gpu {device.index}] MUCS score {end}/{N_TRAIN} "
                f"({100*end/N_TRAIN:.1f}%) elapsed={elapsed/60:.1f}m "
                f"eta={eta/60:.1f}m",
                flush=True,
            )
    return scores, baseline_means, changed_means


def run_query(qid, query, dataset, device, schedule):
    output_dir = ATTR_DIR / MUCS_METHOD / f"q{qid:02d}"
    score_path = output_dir / "scores.npy"
    metadata_path = output_dir / "metadata.json"
    if score_path.is_file() and metadata_path.is_file():
        print(f"[gpu {device.index} q{qid:02d}] skip completed score", flush=True)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = MUCS_ROOT / f"q{qid:02d}"
    work_dir.mkdir(parents=True, exist_ok=True)
    baseline_model, changed_model, unlearning = unlearn_endpoint(
        qid, query, dataset, device, schedule, work_dir
    )
    scores, baseline_means, changed_means = score_training_points(
        baseline_model, changed_model, dataset, device, schedule
    )
    if not np.isfinite(scores).all():
        raise RuntimeError(f"q{qid:02d}: MUCS scores contain non-finite values")
    np.save(score_path, scores)
    np.save(output_dir / "baseline_loss_mc100_mean.npy", baseline_means)
    np.save(output_dir / "unlearned_loss_mc100_mean.npy", changed_means)
    metadata = {
        "method": MUCS_METHOD,
        "query_id": qid,
        "family": query["family"],
        "initial_parameter_source": "epoch-200 raw",
        "null_parameter_source": "epoch-4 raw",
        "endpoint_source": "saved final-EMA query endpoint",
        "endpoint_mc": MUCS_ENDPOINT_MC,
        "training_point_mc": MUCS_TRAIN_MC,
        "paired_mc_before_after": True,
        "score_formula": (
            "mean_m[(L_i,m(theta_prime)-L_i,m(theta))/"
            "(L_i,m(theta_prime)+L_i,m(theta)+eps)]"
        ),
        "unlearning": {
            key: value
            for key, value in unlearning.items()
            if key != "model_state"
        },
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "score_mean": float(scores.mean()),
    }
    atomic_json(metadata, metadata_path)
    print(
        f"[gpu {device.index} q{qid:02d}] saved MUCS score "
        f"range=[{scores.min():.6g},{scores.max():.6g}]",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    with open(QUERY_DIR / "manifest.json") as handle:
        manifest = json.load(handle)
    queries = {int(query["query_id"]): query for query in manifest}
    qids = list(MUCS_QUERY_IDS)[args.shard_index :: args.shard_count]
    dataset = ColorGridDataset(str(BASE_CSV), grid_size=3)
    schedule = base.make_linear_schedule(T, device=device)
    print(f"[gpu {args.gpu}] MUCS qids={qids}", flush=True)
    for qid in qids:
        query = queries[qid]
        if query["family"] != MUCS_FAMILY:
            raise ValueError(f"q{qid:02d} family={query['family']}, expected {MUCS_FAMILY}")
        run_query(qid, query, dataset, device, schedule)


if __name__ == "__main__":
    main()
