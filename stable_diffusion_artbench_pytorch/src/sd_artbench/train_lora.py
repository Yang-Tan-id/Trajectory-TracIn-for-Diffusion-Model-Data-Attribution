import argparse
import math
import pickle
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.optimization import get_scheduler
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer

from .config import ArtBenchConfig, label_to_prompt


class IndexedArtBenchDataset(Dataset):
    def __init__(
        self,
        metadata_csv: Path,
        image_root: Path,
        index_path: Path,
        tokenizer: CLIPTokenizer,
        resolution: int,
    ) -> None:
        frame = pd.read_csv(metadata_csv)
        with index_path.open("rb") as handle:
            indices = pickle.load(handle)
        self.frame = frame.loc[indices].copy()
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, item: int):
        row = self.frame.iloc[item]
        image_path = self.image_root / row["label"] / row["name"]
        with Image.open(image_path) as image:
            pixel_values = self.transform(image.convert("RGB"))
        tokenized = self.tokenizer(
            label_to_prompt(row["label"]),
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized.input_ids[0],
        }


def create_lora_layers(unet: UNet2DConditionModel, rank: int) -> AttnProcsLayers:
    processors = {}
    reversed_channels = list(reversed(unet.config.block_out_channels))
    for name in unet.attn_processors.keys():
        cross_attention_dim = (
            None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = reversed_channels[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            raise ValueError(f"Unexpected attention processor name: {name}")
        processors[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=rank,
        )
    unet.set_attn_processor(processors)
    return AttnProcsLayers(unet.attn_processors)


def parse_args() -> argparse.Namespace:
    defaults = ArtBenchConfig()
    parser = argparse.ArgumentParser(description="Fine-tune miniSD on ArtBench-2 with LoRA.")
    parser.add_argument("--model", default="lambdalabs/miniSD-diffusers")
    parser.add_argument("--metadata-csv", type=Path, default=defaults.metadata_csv)
    parser.add_argument("--image-root", type=Path, default=defaults.image_root)
    parser.add_argument(
        "--index-path", type=Path, default=defaults.index_root / "idx-train.pkl"
    )
    parser.add_argument("--output-dir", type=Path, default=defaults.index_root.parent.parent / "outputs" / "artbench2-lora")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument("--save-every-epochs", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    torch.manual_seed(args.seed)

    tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    scheduler = DDPMScheduler.from_pretrained(args.model, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(args.model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.model, subfolder="unet")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    lora_layers = create_lora_layers(unet, args.rank)

    dataset = IndexedArtBenchDataset(
        args.metadata_csv,
        args.image_root,
        args.index_path,
        tokenizer,
        args.resolution,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    optimizer = torch.optim.AdamW(
        lora_layers.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(len(dataloader) / accelerator.num_processes)
    total_steps = updates_per_epoch * args.epochs
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=math.ceil(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    lora_layers, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        lora_layers, optimizer, dataloader, lr_scheduler
    )
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0
    for epoch in range(args.epochs):
        unet.train()
        epoch_loss = 0.0
        for batch in dataloader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(
                    accelerator.device, dtype=weight_dtype
                )
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                ).long()
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                hidden_states = text_encoder(batch["input_ids"])[0]
                prediction = unet(noisy_latents, timesteps, hidden_states).sample
                if scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif scheduler.config.prediction_type == "v_prediction":
                    target = scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unsupported prediction type: {scheduler.config.prediction_type}"
                    )
                loss = F.mse_loss(prediction.float(), target.float(), reduction="mean")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_layers.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            epoch_loss += loss.detach().float().item()
            global_step += 1

        if accelerator.is_main_process:
            mean_loss = epoch_loss / max(len(dataloader), 1)
            print(f"epoch={epoch + 1}/{args.epochs} step={global_step} loss={mean_loss:.6f}")
            if (epoch + 1) % args.save_every_epochs == 0:
                unwrapped = accelerator.unwrap_model(unet)
                unwrapped.save_attn_procs(args.output_dir / f"epoch-{epoch + 1}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(unet).save_attn_procs(args.output_dir / "final")
        print(f"Saved final LoRA weights to {args.output_dir / 'final'}")


if __name__ == "__main__":
    main()

