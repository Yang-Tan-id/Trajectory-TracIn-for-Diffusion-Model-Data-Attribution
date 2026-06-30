# Trajectory-TracIn-for-Diffusion-Model-Data-Attribution

This repository studies how training points influence diffusion model generations by
accumulating attribution signals along the reverse denoising trajectory, rather than
only at the endpoint.

The maintained experiment scaffold is:

```text
diffusion_jax_refined/
```

The old raw snapshot folders are kept only as historical/reference material. New
runs should be launched from `diffusion_jax_refined/<dataset>/scripts/`.

## Current Main Workflow

The default/mainline workflow is prompted JAX training plus legacy JAX attribution
engines:

1. Select a dataset folder: `cifar2`, `cifar10`, or `artbench`.
2. Train one model for one experiment/training seed.
3. Generate one or many samples from that model.
4. Run attribution algorithms on the generated sample trajectory.
5. Run counterfactual and LDS analysis from the attribution score files.

Example:

```bash
cd diffusion_jax_refined/cifar10

EXPERIMENT_TAG=experiment1 TRAIN_SEED=42 CUDA_VISIBLE_DEVICES=0 bash scripts/00_train.sh
EXPERIMENT_TAG=experiment1 SAMPLE_SEEDS=0,1,2 CUDA_VISIBLE_DEVICES=0 bash scripts/00_sample.sh
EXPERIMENT_TAG=experiment1 ALGORITHMS="das traj_tracin" CUDA_VISIBLE_DEVICES=0 bash scripts/01_data_attribution.sh
EXPERIMENT_TAG=experiment1 ALGORITHMS="das traj_tracin" TOPK=5000 bash scripts/02_metric_counterfactual.sh
EXPERIMENT_TAG=experiment1 LDS_M=100 LDS_K=5000 LDS_SAMPLE_RANDOM_SEED=0 bash scripts/03_lds_training.sh
EXPERIMENT_TAG=experiment1 ALGORITHMS="das traj_tracin" \
LDS_MODEL_DIRS="result/experiment1/lds_model/m_100_k_5000_seed_0" \
bash scripts/04_lds_eval.sh
```

Important convention: one `EXPERIMENT_TAG` should normally have one
`TRAIN_SEED`, therefore one prompted model checkpoint directory:

```text
diffusion_jax_refined/<dataset>/result/<experiment>/model/prompted_jax/
```

The same model can generate many attribution samples. Change sample count/identity
with:

```bash
SAMPLE_SEEDS=0,1,2,3,4
SAMPLE_BATCH_SIZE=1
SAMPLE_TRAJECTORY_STEPS=100
bash scripts/00_sample.sh
```

The sample output path is controlled by each dataset's `sampling/CONFIG.py` and
lands under:

```text
diffusion_jax_refined/<dataset>/result/<experiment>/eval/sampling/
```

Attribution currently reads `attribution_sample_seed=0` and
`attribution_sample_index=0` from `<dataset>/dataset_config.py`. If you generate
many samples and want to attribute a different one, edit those two fields in the
target algorithm config inside `<dataset>/dataset_config.py`.

## Prompted JAX vs Unprompted

Prompted JAX is one primary path. It uses the JAX models in
`diffusion_jax_refined/legacy_jax/` and writes checkpoints to the shared
`model/prompted_jax` folder for the experiment. All prompted attribution
algorithms read that same model.

The unprompted path uses the same JAX UNet family and the same attribution
engines, with `class_cond=False`. Training, sampling, and attribution remain
three separate stages:

```bash
cd diffusion_jax_refined/cifar10
EXPERIMENT_TAG=experiment1 TRAIN_SEED=42 CUDA_VISIBLE_DEVICES=0 bash scripts/00_train_unprompted.sh
EXPERIMENT_TAG=experiment1 TRAIN_SEED=42 SAMPLE_SEEDS=0 CUDA_VISIBLE_DEVICES=0 bash scripts/00_sample_unprompted.sh
EXPERIMENT_TAG=experiment1 TRAIN_SEED=42 ALGORITHMS="das traj_tracin dtrak end_tracin journey_trak" CUDA_VISIBLE_DEVICES=0 bash scripts/01_data_attribution_unprompted.sh
```

The checkpoint and generated query trajectory are stored under:

```text
diffusion_jax_refined/<dataset>/result/<experiment>/model/unprompted_jax/
diffusion_jax_refined/<dataset>/result/<experiment>/eval/sampling/
```

Attribution never samples implicitly. `00_sample_unprompted.sh` must run first,
and `01_data_attribution_unprompted.sh` only consumes the saved sample and
trajectory. Evaluation remains separate:

```bash
EXPERIMENT_TAG=experiment1 ALGORITHMS="das" TOPK=5000 bash scripts/02_metric_counterfactual_unprompted.sh
EXPERIMENT_TAG=experiment1 LDS_M=100 LDS_K=5000 LDS_SAMPLE_RANDOM_SEED=0 bash scripts/03_lds_training_unprompted.sh
EXPERIMENT_TAG=experiment1 ALGORITHMS="das" \
LDS_MODEL_DIRS="result/experiment1/lds_model/unprompted/m_100_k_5000_seed_0" \
bash scripts/04_lds_eval_unprompted.sh
```

See `diffusion_jax_refined/README_UNPROMPTED.md` for the unprompted details.

## Trajectory TracIn Range Splitting

`traj_tracin` is expensive because it scores training examples along saved reverse
denoising trajectories. Split the training-score index universe with
`ATTRIBUTION_RANGES` / `SCORE_INDEX_RANGES`:

```bash
cd diffusion_jax_refined/cifar10

EXPERIMENT_TAG=experiment1 ALGORITHMS="traj_tracin" \
ATTRIBUTION_RANGES="1-2500,2501-5000,5001-7500,7501-10000" \
CUDA_VISIBLE_DEVICES=0 bash scripts/01_data_attribution.sh
```

For multiple GPUs, run disjoint ranges in different terminals:

```bash
CUDA_VISIBLE_DEVICES=0 ALGORITHMS="traj_tracin" ATTRIBUTION_RANGES="1-2500,2501-5000" bash scripts/01_data_attribution.sh
CUDA_VISIBLE_DEVICES=1 ALGORITHMS="traj_tracin" ATTRIBUTION_RANGES="5001-7500,7501-10000" bash scripts/01_data_attribution.sh
```

Counterfactual and LDS can then combine those range outputs by passing the same
`ATTRIBUTION_RANGES`:

```bash
ALGORITHMS="traj_tracin" ATTRIBUTION_RANGES="1-2500,2501-5000,5001-7500,7501-10000" TOPK=5000 bash scripts/02_metric_counterfactual.sh
LDS_M=100 LDS_K=5000 LDS_SAMPLE_RANDOM_SEED=0 bash scripts/03_lds_training.sh
ALGORITHMS="traj_tracin" ATTRIBUTION_RANGES="1-2500,2501-5000,5001-7500,7501-10000" \
LDS_MODEL_DIRS="result/experiment1/lds_model/m_100_k_5000_seed_0" \
bash scripts/04_lds_eval.sh
```

If you already know the score folders exactly, use:

```bash
ATTRIBUTION_RESULT_DIRS="/path/to/range_1,/path/to/range_2"
```

## Current Metrics

Counterfactual analysis currently takes one or more attribution score folders,
selects the top `TOPK` influential training indices, retrains a CIFAR JAX model
with those rows removed, and compares the retrained model against the base
checkpoint for the selected query/prompt. The prompted entrypoint is:

```bash
TOPK=5000 ALGORITHMS="das" bash scripts/02_metric_counterfactual.sh
```

LDS training and evaluation can be run independently. For a CIFAR dataset,
training inherits the normal model checkpoint config and only needs `m`, `k`,
and the subset sampling seed:

```bash
cd diffusion_jax_refined/cifar10
LDS_M=100 LDS_K=5000 LDS_SAMPLE_RANDOM_SEED=0 bash scripts/03_lds_training.sh
```

Reusable subset models are stored under
`result/<experiment>/lds_model/m_<m>_k_<k>_seed_<seed>/`. Evaluation accepts
one or more comma-separated model folders and never retrains them:

```bash
LDS_MODEL_DIRS="\
  result/experiment1/lds_model/m_50_k_5000_seed_0,
  result/experiment1/lds_model/m_50_k_5000_seed_1" \
ALGORITHMS="das traj_tracin" bash scripts/04_lds_eval.sh
```

The combined scatter/CSV/summary outputs are written below
`result/<experiment>/eval/lds/<algorithm>/`.

Unprompted LDS follows the same split workflow and inherits the
`class_cond=False` checkpoint config:

```bash
LDS_M=100 LDS_K=5000 LDS_SAMPLE_RANDOM_SEED=0 \
bash scripts/03_lds_training_unprompted.sh

LDS_MODEL_DIRS="result/experiment1/lds_model/unprompted/m_100_k_5000_seed_0" \
ALGORITHMS="das traj_tracin" bash scripts/04_lds_eval_unprompted.sh
```

Unprompted models live under `result/<experiment>/lds_model/unprompted/`;
their results are written under `result/<experiment>/eval/lds_unprompted/`.

Trajectory TracIn and LDS use the same trajectory-noise objective:
`sum_k w_k ||eps_theta(x_ref_k,k) - eps_theta_ref(x_ref_k,k)||^2`. LDS evaluates
the subset checkpoint against the same full/reference checkpoint on the same
saved sample trajectory. Attribution therefore needs training checkpoints other
than the reference checkpoint; the objective and its gradient are exactly zero
at `theta == theta_ref`.

If the loaded attribution contains any negative score, LDS also squares every
datapoint score and writes only two additional artifacts:
`lds_results_squared_scores.csv` and `lds_scatter_squared_scores.png`.

The legacy counterfactual and full LDS engines are still CIFAR-centered.
ArtBench has the same scaffold folders, but full ArtBench
counterfactual/LDS needs an ArtBench metric engine.

## ArtBench

ArtBench uses the latent model path. During training,
`DM__training_ARTBENCH_latent.py` trains or reuses an autoencoder, caches image
latents, and trains the diffusion model in latent space. This is already wired
into:

```bash
cd diffusion_jax_refined/artbench
EXPERIMENT_TAG=experiment1 TRAIN_SEED=42 CUDA_VISIBLE_DEVICES=0 bash scripts/00_train.sh
```

Important ArtBench knobs:

```bash
ARTBENCH_REUSE_AUTOENCODER=1
ARTBENCH_IMAGE_SIZE=256
ARTBENCH_AE_DOWNSAMPLE_FACTOR=4
ARTBENCH_LATENT_CHANNELS=4
JAX_EPOCHS=100
```

Raw ArtBench images should live in:

```text
diffusion_jax_refined/dataset/artbench/raw/
```

Latent caches are written/read from:

```text
diffusion_jax_refined/dataset/artbench/latents/artbench256/
```
