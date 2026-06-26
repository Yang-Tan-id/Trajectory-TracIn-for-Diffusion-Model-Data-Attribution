# Dataset Storage

This folder is the single data entrypoint for `diffusion_jax_refined`.

Put real datasets or symlinks here:

```text
dataset/
  cifar2/
    cifar-10-batches-py/      # JAX CIFAR batch files: batches.meta, data_batch_*
    hf_cifar10/train/         # optional legacy Diffusers dataset
    indices/lds-val/          # sub-idx-<index>.pkl
  cifar10/
    cifar-10-batches-py/
    hf_cifar10/train/
    indices/lds-val/
  artbench/
    latents/artbench256/      # train_latents.npz, test_latents.npz
    hf_artbench/train/        # optional legacy Diffusers dataset
    indices/
    raw/
```

The actual data files are git-ignored by default; README and `.gitkeep` files keep the directory shape.

Both prompted and unprompted JAX training use the native CIFAR batch files or
ArtBench raw/latent data. The `hf_*` folders are retained only for the legacy
Diffusers utilities and are not required by the maintained unprompted JAX path.
