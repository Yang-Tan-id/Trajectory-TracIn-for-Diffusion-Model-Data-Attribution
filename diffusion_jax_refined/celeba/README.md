# CelebA 10k unconditional DDPM

This framework mirrors the 3D Shapes pixel-DDPM training contract, while using
CelebA's official training partition and no class conditioning.

## Fixed experiment contract

- Read the official `list_eval_partition.txt`; only split `0` is eligible.
- Select 10,000 distinct filenames with NumPy `default_rng(42)`.
- Center-crop each aligned image to 140x140, then bicubic-resize to 64x64.
- Select and persist a seed-42 random 5,000-point attribution universe as both
  integer indices and a boolean mask.
- Train the same epsilon-prediction pixel UNet used by 3D Shapes for 200 epochs,
  with 1,000 DDPM steps and cosine learning-rate decay after 10% warmup.
- The initial runner stops after data preparation and base training; LDS has a
  separate four-GPU launcher for the later stage.

Prepare data and the attribution mask locally:

```bash
cd diffusion_jax_refined/celeba
python script/prepare_celeba.py \
  --images-zip /path/to/img_align_celeba.zip \
  --partition /path/to/list_eval_partition.txt
```

Train after preparation (GPU environment recommended):

```bash
python training/run_training.py
```

The combined runner prints its plan by default and executes only with
`--execute`:

```bash
python script/run_celeba_experiment.py \
  --images-zip /path/to/img_align_celeba.zip \
  --partition /path/to/list_eval_partition.txt \
  --execute
```

On TACC RTX-small, place the raw archive and partition file under
`diffusion_jax_refined/dataset/celeba/` (or set `CELEBA_IMAGES_ZIP` and
`CELEBA_PARTITION`) and submit:

```bash
sbatch diffusion_jax_refined/celeba/tacc/rtx_small/train_base_rtx_small.sh
```

## Four-GPU LDS training

After the epoch-200 base checkpoint exists, train the Shape3D-compatible LDS
bank with three subset seeds. Each seed owns 64 independently trained random
2,500-of-5,000 subset models; four GPUs receive 16 models each. Only epoch 200
is retained for each subset model:

```bash
cd diffusion_jax_refined/celeba
bash script/run_lds_4gpu_all_seeds.sh
```

The three subset seeds run sequentially because all four GPUs participate in
each seed. The launcher is resumable: completed final checkpoints are skipped.
