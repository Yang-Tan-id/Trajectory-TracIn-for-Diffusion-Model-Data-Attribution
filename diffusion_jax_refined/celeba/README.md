# CelebA 10k unconditional DDPM

This framework mirrors the 3D Shapes pixel-DDPM training contract, while using
CelebA's official training partition and no class conditioning.

The optional Attr40 runner creates a separate conditionally trained model from
the official `list_attr_celeba.txt`; the original unconditional experiment is
kept intact.

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

## Conditional 40-attribute training

```bash
python script/run_celeba_attr40_experiment.py \
  --images-zip /path/to/img_align_celeba.zip \
  --partition /path/to/list_eval_partition.txt \
  --attributes /path/to/list_attr_celeba.txt \
  --execute
```

This writes a separate prepared dataset at `dataset/celeba/10000_attr40/` and
checkpoints below `result/attr40_experiment1/model/prompted_jax/`.

After the conditional epoch-200 checkpoint exists, its separate four-GPU LDS
bank uses the same three-seed, 64-model, 2,500-of-5,000 contract:

```bash
bash script/run_lds_attr40_4gpu_all_seeds.sh
```

## Attr40 validation-label query100 and LDS targets

Create 25 deterministic labels from the official validation split (excluding
`Blurry`, with 4--14 positive attributes), then pair every label with initial
seeds 0--3 to obtain 100 queries:

```bash
python script/build_attr40_query100.py \
  --partition ../dataset/celeba/list_eval_partition.txt \
  --attributes ../dataset/celeba/list_attr_celeba.txt
```

After all 192 conditional LDS checkpoints exist, GPUs 4--7 can sample the 100
queries with 50-step DDIM and cache the four reusable true-f targets. Endpoint
and trajectory counterfactual are computed together from one subset-model
trajectory. The run is resumable and exports one long CSV plus one CSV per
target:

```bash
python script/run_attr40_query100_sample_lds.py \
  --execute \
  --query-file queries_attr40_val25_x4.json \
  --gpus 4,5,6,7
```

Outputs are written below `result/attr40_experiment1/sample_ddim_eta0_50/`,
the per-query LDS target cache below `result/attr40_experiment1/eval/`, and
the CSV files below `result/attr40_experiment1/query100_lds_csv_ddim50/`.
