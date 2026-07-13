# Torch DAS for CIFAR2 Diffusion

This folder is a cleaned-up, runnable torch/diffusers version of the DAS CIFAR2
pipeline, placed inside the Trajectory TracIn repository.

It uses the DAS torch dependency set:

- `torch==2.5.0`
- `torchvision==0.20.0`
- `diffusers==0.23.1`
- `accelerate==0.25.0`
- `datasets==2.16.1`

The default commands run on a tiny synthetic dataset, so you can verify the
code without downloading anything.

## Setup

From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-torch.txt
```

For CUDA, install the `torch==2.5.0` wheel matching your CUDA version from the
official PyTorch instructions, then install the remaining requirements.

## Smoke Test, No Data Download

```bash
cd diffusion_torch_das
bash scripts/smoke.sh
```

This trains a tiny DDPM for one epoch on synthetic 32x32 images, generates a few
images, computes losses, computes projected train/query gradients, and writes
ridge scores under `runs/smoke/`.

Equivalent manual commands:

```bash
python3 -m torch_das.train \
  --dataset synthetic \
  --config configs/tiny_unet.json \
  --output-dir runs/smoke/ddpm \
  --synthetic-samples 8 \
  --batch-size 4 \
  --num-epochs 1 \
  --num-train-timesteps 20 \
  --device auto

python3 -m torch_das.generate \
  --model-dir runs/smoke/ddpm \
  --output-dir runs/smoke/gen \
  --num-images 4 \
  --num-inference-steps 5

python3 -m torch_das.eval_loss \
  --model-dir runs/smoke/ddpm \
  --dataset synthetic \
  --synthetic-samples 8 \
  --max-samples 4 \
  --num-timesteps 4 \
  --output runs/smoke/losses.pkl

python3 -m torch_das.gradients \
  --model-dir runs/smoke/ddpm \
  --dataset synthetic \
  --synthetic-samples 8 \
  --max-samples 4 \
  --num-timesteps 2 \
  --projection-dim 32 \
  --output runs/smoke/train_grads.npy

python3 -m torch_das.gradients \
  --model-dir runs/smoke/ddpm \
  --dataset runs/smoke/gen \
  --dataset-type gen \
  --max-samples 4 \
  --num-timesteps 2 \
  --projection-dim 32 \
  --output runs/smoke/query_grads.npy

python3 -m torch_das.score \
  --train-grads runs/smoke/train_grads.npy \
  --query-grads runs/smoke/query_grads.npy \
  --train-shape 4,32 \
  --query-shape 4,32 \
  --output runs/smoke/scores.npy
```

## Dataset Folders

This torch version now mirrors the repo's dataset-first layout:

- `cifar2/`: CIFAR-10 filtered to automobile + horse.
- `cifar10/`: full CIFAR-10, all 10 classes.

The shared implementation lives in `torch_das/`; the dataset folders contain
the runnable scripts and short dataset-specific notes.

## Real CIFAR2 Run

You do not need to download CIFAR-10 if the repository already contains:

```text
diffusion_jax_refined/dataset/cifar2/cifar-10-batches-py/
```

From `diffusion_torch_das/`, train on the automobile/horse subset:

```bash
DATA="../diffusion_jax_refined/dataset/cifar2/cifar-10-batches-py"

python3 -m torch_das.train \
  --dataset "$DATA" \
  --dataset-kind cifar2 \
  --config configs/cifar2_unet_das.json \
  --output-dir runs/cifar2/ddpm \
  --center-crop \
  --random-flip \
  --batch-size 128 \
  --num-epochs 200 \
  --checkpointing-steps 500 \
  --device cuda
```

If you pass `--dataset cifar10`, HuggingFace `datasets` will download CIFAR-10
to its cache. If you pass the local raw CIFAR batch folder above, no download is
needed.

You can also use the dataset folder script:

```bash
bash cifar2/scripts/01_train.sh
```

## Real CIFAR10 Run

Use the local full CIFAR-10 folder:

```bash
DATA="../diffusion_jax_refined/dataset/cifar10/cifar-10-batches-py"

python3 -m torch_das.train \
  --dataset "$DATA" \
  --dataset-kind cifar10 \
  --config configs/cifar10_unet_das.json \
  --output-dir runs/cifar10/ddpm \
  --center-crop \
  --random-flip \
  --batch-size 128 \
  --num-epochs 200 \
  --checkpointing-steps 500 \
  --device cuda
```

Or:

```bash
bash cifar10/scripts/01_train.sh
```

## Outputs

- `runs/.../ddpm/`: diffusers DDPM pipeline with `unet/` and `scheduler/`.
- `runs/.../gen/`: generated PNG files.
- `*.pkl`: per-example denoising losses over selected timesteps.
- `*_grads.npy`: projected per-example gradients, saved as float32 memmaps.
- `scores.npy`: attribution scores for train points.

## Notes

This folder intentionally avoids WandB and Accelerate for the default path so it
is easier to run locally and push to GitHub. The full DAS-size model config is
kept in `configs/cifar2_unet_das.json`; use `configs/tiny_unet.json` for quick
checks.
