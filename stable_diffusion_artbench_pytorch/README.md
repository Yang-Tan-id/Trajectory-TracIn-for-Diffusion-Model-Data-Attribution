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
The terminal progress bar advances once per optimizer update and reports the
current epoch, micro-batch-averaged loss, and learning rate.

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

Sampling performs 50 sequential DDIM updates.  Each update evaluates one
predicted-noise tensor and advances the latent to the next state on the exact
`999,979,...,19` schedule (a training-timestep stride of 20).

```bash
python -m sd_artbench.generate \
  --lora-dir outputs/artbench2-lora/final \
  --seeds 0,1,2 \
  --images-per-class 1 \
  --replace-flagged
```

Pass `--prompt` more than once to compare custom prompts with the same initial
noise seeds:

```bash
python -m sd_artbench.generate \
  --lora-dir outputs/artbench2-lora/final \
  --output-dir outputs/generated-custom \
  --seeds 0,1,2 \
  --prompt "a post impressionism painting" \
  --prompt "a ukiyo e painting" \
  --prompt "a bird flying through the sky, an ukiyo e painting"
```

For a paired base-model control, replay the exact prompt and actual seed of
every generated query while skipping LoRA loading:

```bash
python -m sd_artbench.generate \
  --no-lora \
  --replay-manifest outputs/artbench2-queries-60/manifest.json \
  --output-dir outputs/artbench2-queries-60-base \
  --steps 50 \
  --guidance-scale 7.5
```

The replay mode deliberately rejects `--replace-flagged`, because replacing a
seed would invalidate the pair. If a base-model result is safety-flagged, rerun
the exact replay with `--disable-safety-checker` rather than changing its seed.

After both sets exist, validate their pairing and create per-query plus overview
comparison images:

```bash
python -m sd_artbench.compare_generations \
  --base-manifest outputs/artbench2-queries-60-base/manifest.json \
  --lora-manifest outputs/artbench2-queries-60/manifest.json \
  --output-dir outputs/artbench2-base-vs-lora
```

To inspect how one fixed query changes across the 25 raw LoRA checkpoints,
render Base plus epochs 4, 8, ..., 100 and build both a timeline and GIF:

```bash
CUDA_VISIBLE_DEVICES=0 python -m sd_artbench.generate_checkpoint_trajectory \
  --query-manifest outputs/artbench2-queries-60/manifest.json \
  --query-indices 22 \
  --checkpoint-root outputs/artbench2-lora \
  --output-dir outputs/checkpoint-trajectories-q22 \
  --disable-safety-checker
```

The trajectory tool infers the LoRA rank from the first selected raw checkpoint
(rank 128 for the default experiment) and rejects an incompatible override.

## 5. Raw-checkpoint attribution

The attribution implementation lives in `src/sd_artbench/attribution/` and is
independent of the JAX model. It reads the 25 raw LoRA/Accelerate checkpoints;
it does not use or reconstruct EMA weights.

Each run writes both training transforms:

- `raw_gradient`: the ordinary per-example LoRA loss gradient;
- `adamw_full`: the hypothetical next AdamW parameter direction reconstructed
  from that checkpoint's first moment, second moment, step, epsilon, betas, and
  decoupled weight decay, after the training run's global-norm clipping at 1.0.
  The learning rate is factored out of the direction and applied exactly once
  during scoring.

Endpoint methods additionally write the four score normalizations `raw`,
`query_l2`, `train_l2`, and `query_train_l2`. Endpoint-Pollute writes linear,
termwise-squared, timestamp-sum-squared, and total-sum-squared reductions as
separate files so their square order cannot be confused.

Endpoint-Pollute evaluates query and training examples with batched VJPs. The
default query batch, training batch, and per-example VJP chunk are all 8;
reduce `--pollute-query-batch-size`, `--pollute-train-batch-size`, or
`--per-example-chunk-size` only if the smoke run runs out of GPU memory.
PyTorch 2.0's legacy `is_grads_batched` implementation falls back to slow
operator-by-operator loops for several Stable Diffusion backward operators.
Pass `--per-example-backend torch_func` to use the public
`torch.func.vmap(torch.func.grad(...))` implementation. The default remains
`legacy` until the two backends have been numerically compared on the target
server/GPU stack.

First run a one-image, one-query, one-timestep smoke test. The query manifest is
the `manifest.json` produced by `sd_artbench.generate`:

```bash
CUDA_VISIBLE_DEVICES=0 python -m sd_artbench.attribution.run_endpoint \
  --query-manifest outputs/sample-preview/manifest.json \
  --checkpoint-indices 0 \
  --query-indices 0 \
  --timestep-count 1 \
  --proj-dim 512 \
  --train-start 0 \
  --train-end 1 \
  --output-dir outputs/attribution-smoke/endpoint
```

Then smoke-test exact-event ReTrac. Its batch size and accumulation must equal
the values used for training. New checkpoints record and validate every replay
argument; older checkpoints are at least checked for consistency against each
saved `global_step`:

```bash
CUDA_VISIBLE_DEVICES=0 python -m sd_artbench.attribution.run_retrac \
  --query-manifest outputs/sample-preview/manifest.json \
  --checkpoint-indices 0 \
  --query-indices 0 \
  --timestep-count 1 \
  --proj-dim 512 \
  --batch-size 8 \
  --gradient-accumulation-steps 8 \
  --num-workers 0 \
  --train-start 0 \
  --train-end 1 \
  --max-replay-batches 1 \
  --output-dir outputs/attribution-smoke/retrac
```

The smoke run is deliberately saved as `retrac_debug.npz`, never as the formal
`retrac.npz`. For the full jobs, remove all smoke limits:

```bash
CUDA_VISIBLE_DEVICES=0 python -m sd_artbench.attribution.run_endpoint \
  --query-manifest outputs/sample-preview/manifest.json \
  --timestep-count 50 \
  --proj-dim 32768 \
  --output-dir outputs/attribution/endpoint

CUDA_VISIBLE_DEVICES=0 python -m sd_artbench.attribution.run_retrac \
  --query-manifest outputs/sample-preview/manifest.json \
  --timestep-count 50 \
  --proj-dim 32768 \
  --batch-size 8 \
  --gradient-accumulation-steps 8 \
  --output-dir outputs/attribution/retrac
```

`run_retrac` defaults to `exact_batch`, which recovers the shuffled training
batch, random flip, VAE sample, noise, timestep, conditioning dropout, and
attention-dropout graph from the saved RNG state. If exact per-example
full-batch differentiation runs out of memory, `--replay-mode batch1` is the
lower-memory approximation; this choice is recorded in the result manifest.

Both runners checkpoint their accumulated scores after each model checkpoint
and resume automatically. For multiple GPUs, launch separate processes with
non-overlapping `--train-start/--train-end` ranges and distinct output
directories. Merge matching `.npz` score files afterward:

```bash
python -m sd_artbench.attribution.merge_shards \
  --inputs outputs/shard0/endpoint_tracin.npz \
           outputs/shard1/endpoint_tracin.npz \
           outputs/shard2/endpoint_tracin.npz \
           outputs/shard3/endpoint_tracin.npz \
  --output outputs/attribution/endpoint_tracin.npz
```

The full attribution run is much more expensive than image generation. Always
finish both smoke tests before starting all 5,000 training images, 25
checkpoints, and the same 50 timestep positions used by DDIM-50 sampling
(`999, 979, ..., 19`).

### Four-GPU optimized launch

The exact ReTrac path uses PyTorch's vmap-backed batched VJP, so raw and
AdamW-full reuse the same per-example gradient. `--per-example-chunk-size`
controls how many per-example VJPs run together (default 2; try 4 after the
smoke test if memory permits). Query/train timestep gradients are also evaluated
in bounded batches instead of 50 independent forward/backward calls.

Use the launcher to restrict work to exactly the GPU IDs assigned to you. For
ReTrac it divides the 24 checkpoint intervals across the four GPUs (six per
GPU), avoiding duplicated replay, then sums the four full-training-set score
files. It writes one log per GPU and resumes each shard independently:

```bash
python -m sd_artbench.attribution.launch_shards \
  --method retrac \
  --gpu-ids 0,1,2,3 \
  --output-dir outputs/attribution/retrac-4gpu \
  -- \
  --query-manifest outputs/sample-preview/manifest.json \
  --timestep-count 50 \
  --timestep-batch-size 2 \
  --proj-dim 32768 \
  --batch-size 8 \
  --gradient-accumulation-steps 8 \
  --num-workers 8 \
  --per-example-chunk-size 2
```

`--gpu-ids` is never inferred from all GPUs on the host. Only the comma-separated
IDs supplied there are placed in child-process `CUDA_VISIBLE_DEVICES` values.
Use `tail -f outputs/attribution/retrac-4gpu/shard-00-gpu-0.log` to follow one
worker. If vmap-backed VJP is unsupported by an operation on the installed
PyTorch/Diffusers build, the worker fails explicitly instead of silently
switching to a different attribution definition.

`trainable_parameters.json` records the exact LoRA parameter names and shapes
used for updates and later attribution. Official D-TRAK checkpoints infer this
set from the LoRA architecture rather than storing a separate Boolean mask;
the manifest is additional metadata and does not change training.

## 6. CLIP nearest-neighbor baseline

Rank the exact 5,000 training images against every generated endpoint using
normalized CLIP image embeddings and cosine similarity:

```bash
CUDA_VISIBLE_DEVICES=0 python -m sd_artbench.clip_similarity \
  --query-manifest outputs/artbench2-queries-60/manifest.json \
  --output-dir outputs/clip-similarity-q60 \
  --batch-size 64 \
  --top-k 20 \
  --contact-sheet-top-k 10
```

The command writes the complete `60 x 5000` matrix to `similarities.npy`, a
long-form `topk.csv`, reusable CLIP embeddings, and one visual contact sheet per
query under `contact_sheets/`. On a rerun, add `--reuse-embeddings` to skip CLIP
encoding when the model and image paths still match the cache.

