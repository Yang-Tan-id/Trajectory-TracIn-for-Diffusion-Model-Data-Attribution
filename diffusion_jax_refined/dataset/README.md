# Dataset Storage

This folder is the single data entrypoint for `diffusion das refine`.

Put real datasets or symlinks here:

```text
dataset/
  cifar2/
    cifar-10-batches-py/      # JAX CIFAR batch files: batches.meta, data_batch_*
    hf_cifar10/train/         # HuggingFace load_from_disk format used by diffusers training
    indices/lds-val/          # sub-idx-<index>.pkl
  cifar10/
    cifar-10-batches-py/
    hf_cifar10/train/
    indices/lds-val/
  artbench/
    latents/artbench256/      # train_latents.npz, test_latents.npz
    hf_artbench/train/        # optional diffusers training dataset
    indices/
    raw/
```

The actual data files are git-ignored by default; README and `.gitkeep` files keep the directory shape.

