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
attention dropout 0.1, conditioning dropout 0.1, center crop, random horizontal
flip, and seed 42.
Resumable training state is saved at 25 evenly spaced epochs (with the default
100 epochs: 4, 8, ..., 100), matching the 25-checkpoint attribution setup.
Every checkpoint contains the LoRA parameters, AdamW moments, scheduler,
mixed-precision scaler, and RNG state. Resume the newest saved state with
`--resume-from-checkpoint latest`.
Final inference weights are written to
`outputs/artbench2-lora/final/`.

On a 24 GiB GPU, use `--batch-size 8 --gradient-accumulation-steps 8` to
preserve the effective batch size of 64 without placing all 64 images in GPU
memory simultaneously.

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

Use the checked-in Conda environment; do not install project dependencies into
`base`:

```bash
conda env create -f environment.yml
conda activate sd_artbench
export PYTHONPATH="$PWD/src"
accelerate config
```

The environment pins the paper-compatible PyTorch 2.0.1, Diffusers 0.16.1,
and CUDA 11.8 stack. Confirm that the server's NVIDIA driver supports this
runtime with `nvidia-smi`. `requirements.txt` remains only as a reference for
the official D-TRAK Python versions.

`fast-jl` and `traker` are intentionally not part of the base environment;
they are needed only by the later attribution stage and require CUDA-specific
installation/testing.

## 1. Place ArtBench

See `dataset/README.md`. Do not copy data from `diffusion_jax_refined` unless it
has the original images and matching `ArtBench-10.csv`; its latent cache alone
is not sufficient for Stable Diffusion fine-tuning.

The original Berkeley file endpoints currently return HTTP 404. The supported
fallback downloads the complete 256 px, 60,000-row Hugging Face mirror and
exports only the 12,000 ArtBench-2 train/test images without re-encoding them:

```bash
python -m sd_artbench.download_hf_artbench
```

This creates both `dataset/metadata/ArtBench-10.csv` and the two required class
folders under `dataset/images/artbench-10-imagefolder/`.

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

`trainable_parameters.json` records the exact LoRA parameter names and shapes
used for updates and later attribution. Official D-TRAK checkpoints infer this
set from the LoRA architecture rather than storing a separate Boolean mask;
the manifest is additional metadata and does not change training.

