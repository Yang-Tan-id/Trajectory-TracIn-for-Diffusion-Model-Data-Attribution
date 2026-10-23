# stable_diffusion_artbench_pytorch

Independent PyTorch/Diffusers implementation for reproducing the Stable
Diffusion + ArtBench-2 branch of D-TRAK. It intentionally sits beside
`diffusion_jax_refined`; Stable Diffusion model code is not shared with the JAX
DDPM implementation.

## Experiment definition

The paper-faithful target dataset is sampled as follows:

- candidate pool: all 10,000 official training images from
  `post_impressionism` and `ukiyo_e`;
- target-model training set: 5,000 images selected with stratified
  `train_test_split`, `random_state=42` (2,500 per class);
- validation set: 1,000 official test images selected the same way (500 per
  class);
- LDS: 64 subsets, each containing 50% of the 5,000-image target training set.

The model defaults reproduce the D-TRAK setup: `lambdalabs/miniSD-diffusers`,
256 px, U-Net attention LoRA rank 128, batch size 64, 100 epochs, AdamW,
learning rate `3e-4`, weight decay `1e-6`, cosine schedule with 10% warmup,
center crop, random horizontal flip, and seed 42.

## Layout

```text
stable_diffusion_artbench_pytorch/
  dataset/
    metadata/ArtBench-10.csv
    images/artbench-10-imagefolder/<label>/<filename>
    indices/
  outputs/
  src/sd_artbench/
    prepare_data.py
    train_lora.py
    generate.py
    attribution/          # D-TRAK feature/score stages go here
  tests/
```

Large datasets, model weights, and generated outputs are ignored by Git.

## Setup

The pinned packages follow the official D-TRAK release and require an NVIDIA
CUDA environment for attribution. Python 3.8 is the safest choice.

```bash
conda create -n sd-artbench-dtrak python=3.8 -y
conda activate sd-artbench-dtrak
pip install -r requirements.txt
pip install -e .
accelerate config
```

Install `requirements-attribution.txt` only when the per-sample gradient and
random-projection stages are added/run; `fast-jl` requires CUDA.

## 1. Place ArtBench

See `dataset/README.md`. Do not copy data from `diffusion_jax_refined` unless it
has the original images and matching `ArtBench-10.csv`; its latent cache alone
is not sufficient for Stable Diffusion fine-tuning.

## 2. Prepare exact ArtBench-2 indices

```bash
python -m sd_artbench.prepare_data
```

This writes `idx-train.pkl`, `idx-val.pkl`, 64 LDS subset files, readable CSV
manifests, and `split_summary.json` under `dataset/indices/`.

## 3. Train LoRA

```bash
accelerate launch --mixed_precision fp16 -m sd_artbench.train_lora
```

For a smoke test:

```bash
accelerate launch --mixed_precision fp16 -m sd_artbench.train_lora \
  --epochs 1 --batch-size 2 --num-workers 0 \
  --output-dir outputs/smoke-lora
```

The pretrained model is downloaded from Hugging Face on first use. Subsequent
runs can use the local cache.

## 4. Generate with 50-step DDIM

```bash
python -m sd_artbench.generate \
  --lora-dir outputs/artbench2-lora/final \
  --seeds 0,1,2 \
  --images-per-class 1
```

The D-TRAK per-sample gradient, random projection, and score-combination code
will live in `src/sd_artbench/attribution/` so the PyTorch model path stays
independent of the JAX implementation.

