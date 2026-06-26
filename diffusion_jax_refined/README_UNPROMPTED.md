# Unprompted JAX Track

The unprompted comparison now uses the same JAX UNet and the same five
attribution engines as the prompted track. The only model-level difference is
that `class_cond=False`: training, sampling, query gradients, and training
gradients never use labels or prompts.

The unconditional defaults are kept close to the previous Diffusers run:

- 1,000 linear diffusion steps with epsilon prediction
- 200 epochs (100 for ArtBench latent)
- batch size 128
- AdamW with learning rate `1e-4`, beta `(0.95, 0.999)`, epsilon `1e-8`,
  and weight decay `1e-6`
- dropout `0.1`, EMA `0.999`, gradient clipping `1.0`, and fp32

The JAX and Diffusers UNets are not layer-for-layer identical, so their weights
cannot be compared directly. The data, schedule, optimizer defaults, sample
seed, trajectory format, and attribution definitions are aligned.

## Run

From a dataset directory (`cifar10`, `cifar2`, or `artbench`):

```bash
TRAIN_SEED=42 CUDA_VISIBLE_DEVICES=0 bash scripts/00_train_unprompted.sh
TRAIN_SEED=42 CUDA_VISIBLE_DEVICES=0 bash scripts/00_sample_unprompted.sh
CUDA_VISIBLE_DEVICES=0 ALGORITHMS="das traj_tracin dtrak end_tracin journey_trak" \
  bash scripts/01_data_attribution_unprompted.sh
```

Unprompted checkpoints are written to:

```text
result/<experiment>/model/unprompted_jax/
```

`00_sample_unprompted.sh` generates the unconditional sample and saved
trajectory with the same JAX sampler used by the prompted path.
`01_data_attribution_unprompted.sh` only reads that saved query; it never
starts sampling implicitly.

Attribution outputs retain the existing evaluation contract:

```text
result/<experiment>/attribution_score/<algorithm>_unprompted/
```

The following engines are supported:

- `das`
- `dtrak`
- `end_tracin`
- `traj_tracin`
- `journey_trak`

They use the prompted implementation and hyperparameters. The derived
unprompted config changes only conditioning, checkpoint paths, sample paths,
and output paths.

The old `common/unprompted_diffusers_attribution.py` proxy is no longer called
by dataset scripts. It scored each training image with its own denoising MSE
and therefore was not sample-specific or directly comparable to prompted
attribution.
