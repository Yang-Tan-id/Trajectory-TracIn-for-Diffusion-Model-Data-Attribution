import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import apply_index_subset, build_dataset
from .utils import default_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Compute projected per-example gradients for DAS-style scoring.")
    parser.add_argument("--model-dir", default="runs/smoke/ddpm")
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--dataset-kind", default="synthetic", choices=["synthetic", "cifar2", "cifar10"])
    parser.add_argument("--dataset-type", default="train", choices=["train", "val", "test", "gen"])
    parser.add_argument("--index-path", default=None)
    parser.add_argument("--output", default="runs/smoke/train_grads.npy")
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--center-crop", action="store_true")
    parser.add_argument("--synthetic-samples", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-timesteps", type=int, default=2)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--projector-chunk-size", type=int, default=2048, help="Rows of the random projector generated at a time")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def flat_grad(model):
    parts = []
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            parts.append(param.grad.detach().flatten())
    return torch.cat(parts)


def vectorize_per_sample_grads(grad_dict):
    values = list(grad_dict.values())
    batch_size = values[0].shape[0]
    rows = []
    for batch_id in range(batch_size):
        rows.append(torch.cat([value[batch_id].reshape(-1) for value in values]))
    return torch.stack(rows)


def project_vector(vector, projection_dim, chunk_size, seed=0):
    out = torch.zeros(projection_dim, dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    scale = 1.0 / np.sqrt(projection_dim)
    vector = vector.cpu().float()
    for start in range(0, vector.numel(), chunk_size):
        end = min(start + chunk_size, vector.numel())
        block = torch.randn(end - start, projection_dim, generator=generator, dtype=torch.float32)
        out += vector[start:end] @ block
    return out * scale


def build_projector(param_count, projection_dim, device):
    if device.type != "cuda":
        return None
    try:
        from trak.projectors import CudaProjector, ProjectionType
    except Exception:
        return None
    return CudaProjector(
        grad_dim=param_count,
        proj_dim=projection_dim,
        seed=0,
        proj_type=ProjectionType.normal,
        device=str(device),
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    device = default_device(args.device)
    split = "test" if args.dataset_type in {"val", "test"} else args.dataset_type
    dataset = build_dataset(args, split=split)
    dataset = apply_index_subset(dataset, args.index_path, args.max_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = UNet2DModel.from_pretrained(Path(args.model_dir) / "unet").to(device).eval()
    scheduler = DDPMScheduler.from_pretrained(args.model_dir, subfolder="scheduler")
    selected = np.linspace(0, scheduler.config.num_train_timesteps - 1, args.num_timesteps, dtype=int)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    projector = build_projector(param_count, args.projection_dim, device)
    projected = np.memmap(args.output, dtype=np.float32, mode="w+", shape=(len(dataset), args.projection_dim))

    params = {k: v.detach() for k, v in model.named_parameters() if v.requires_grad}
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    from torch.func import functional_call, grad, vmap

    def compute_loss(params_, buffers_, noisy_image, timestep, target):
        pred = functional_call(
            model,
            (params_, buffers_),
            (noisy_image.unsqueeze(0), timestep.unsqueeze(0)),
        ).sample
        return F.mse_loss(pred.float(), target.unsqueeze(0).float())

    sample_grad = vmap(grad(compute_loss), in_dims=(None, None, 0, 0, 0))

    offset = 0
    for batch in tqdm(dataloader, desc="grad"):
        clean = batch["input"].to(device)
        emb = None
        for t in selected:
            timesteps = torch.full((clean.shape[0],), int(t), device=device, dtype=torch.long)
            set_seed(args.seed * 1000 + int(t))
            noise = torch.randn_like(clean)
            noisy = scheduler.add_noise(clean, noise, timesteps)
            model.zero_grad(set_to_none=True)
            grad_dict = sample_grad(params, buffers, noisy, timesteps, noise)
            grad_rows = vectorize_per_sample_grads(grad_dict)
            grad_rows = grad_rows / (torch.linalg.norm(grad_rows, dim=1, keepdim=True) + 1e-8)
            emb = grad_rows if emb is None else emb + grad_rows
        emb = emb / len(selected)
        if projector is not None:
            projected_batch = projector.project(emb, model_id=0).detach().cpu().numpy()
        else:
            projected_batch = np.stack([
                project_vector(row, args.projection_dim, args.projector_chunk_size).numpy()
                for row in emb
            ])
        projected[offset:offset + clean.shape[0]] = projected_batch
        offset += clean.shape[0]
    projected.flush()
    print(f"saved projected gradients {(len(dataset), args.projection_dim)} to {args.output}")


if __name__ == "__main__":
    main()
