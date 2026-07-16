# TACC launchers

The old dataset-local `cifar2/tacc` scripts were removed. New launchers live by
machine family:

- `h100/script_0.sh`
- `vista/script_0.sh`

Both wrappers select a dataset and one of the training modes, then delegate to
that dataset's refined `scripts/script_0.sh`. The mode names are launcher
conventions: `prompted` means class-conditioned training, `unprompted` means
unconditional training, and `solo`/`multi` select the GPU launch style. Training
does not read `QUERY` or create prompt-specific model folders.

Examples:

```bash
DATASET=cifar2 TRAIN_MODE=prompted_solo TRAIN_SEED=42 GPU_IDS=0 \
  bash diffusion_jax_refined/tacc/h100/script_0.sh

DATASET=cifar2 TRAIN_MODES="prompted_solo unprompted_solo" TRAIN_SEED=42 GPU_IDS=0 \
  bash diffusion_jax_refined/tacc/h100/script_0.sh

DATASET=cifar10 TRAIN_MODE=prompted_multi TRAIN_SEED=42 GPU_IDS=0,1,2,3 \
  bash diffusion_jax_refined/tacc/vista/script_0.sh

DATASET=artbench TRAIN_MODE=unprompted_multi TRAIN_SEED=42 GPU_IDS=0,1,2,3 \
  bash diffusion_jax_refined/tacc/h100/script_0.sh
```

Checkpoints are grouped by dataset/experiment/model type, with the training seed
in the file name:

```text
<dataset>/result/<experiment>/model/prompted_jax/seed_<TRAIN_SEED>_epoch_<epoch>.ckpt
<dataset>/result/<experiment>/model/unprompted_jax/seed_<TRAIN_SEED>_epoch_<epoch>.ckpt
```
