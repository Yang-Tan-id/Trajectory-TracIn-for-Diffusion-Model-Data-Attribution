# ArtBench refine runs

Default setup:

- Dataset: ArtBench image-folder / latent pipeline
- Prompt/query: `baroque`
- Default training seed: `TRAIN_SEED=42`
- Default experiment: `EXPERIMENT_TAG=experiment1`

Set `QUERY` once (or change its default in `dataset_config.py`). The sampling
prompt and attribution sample folder are derived from it automatically; do not
edit `ATTRIBUTION_SAMPLE_DIR` separately.

## Prompted JAX Latent Flow

Prompted ArtBench training calls `legacy_jax/DM__training_ARTBENCH_latent.py`.
In one training run it trains or reuses the autoencoder, caches image latents
under `dataset/artbench/latents/artbench256/`, and trains the latent diffusion
model under `result/<experiment>/model/prompted_jax/`.

```bash
cd diffusion_jax_refined/artbench

EXPERIMENT_TAG=experiment1 TRAIN_SEED=42 CUDA_VISIBLE_DEVICES=0 bash scripts/00_train.sh
EXPERIMENT_TAG=experiment1 SAMPLE_SEEDS=0,1,2 CUDA_VISIBLE_DEVICES=0 bash scripts/00_sample.sh
EXPERIMENT_TAG=experiment1 ALGORITHMS="das end_tracin dtrak journey_trak" CUDA_VISIBLE_DEVICES=0 bash scripts/01_data_attribution.sh
```

Prompted checkpoint path:

```text
result/experiment1/model/prompted_jax/seed_42_epoch_0100.ckpt
```

Generated sample trajectories:

```text
result/experiment1/eval/sampling/
```

The current legacy prompted counterfactual and full LDS engines are CIFAR-specific.
ArtBench metric folders are scaffolded, but a full ArtBench metric engine is
still needed for prompted counterfactual/LDS.

## Prompted Traj TracIn Range Split

Trajectory TracIn should usually be split by score-index ranges:

```bash
cd diffusion_jax_refined/artbench

EXPERIMENT_TAG=experiment1 ALGORITHMS="traj_tracin" \
ATTRIBUTION_RANGES="1-2500,2501-5000,5001-7500,7501-10000" \
CUDA_VISIBLE_DEVICES=0 bash scripts/01_data_attribution.sh
```

For multiple GPUs, run disjoint ranges in separate terminals:

```bash
EXPERIMENT_TAG=experiment1 ALGORITHMS="traj_tracin" ATTRIBUTION_RANGES="1-2500,2501-5000" bash scripts/01_data_attribution.sh
CUDA_VISIBLE_DEVICES=1 EXPERIMENT_TAG=experiment1 ALGORITHMS="traj_tracin" ATTRIBUTION_RANGES="5001-7500,7501-10000" bash scripts/01_data_attribution.sh
```

## Unprompted JAX Latent Flow

Unprompted uses the same JAX latent UNet, autoencoder, sample format, and five
attribution engines as prompted, with `class_cond=False`. Training, sampling,
attribution, and evaluation are separate commands.

```bash
cd diffusion_jax_refined/artbench

EXPERIMENT_TAG=experiment1 TRAIN_SEED=42 CUDA_VISIBLE_DEVICES=0 bash scripts/00_train_unprompted.sh
EXPERIMENT_TAG=experiment1 TRAIN_SEED=42 SAMPLE_SEEDS=0 CUDA_VISIBLE_DEVICES=0 bash scripts/00_sample_unprompted.sh
EXPERIMENT_TAG=experiment1 TRAIN_SEED=42 ALGORITHMS="das traj_tracin dtrak end_tracin journey_trak" CUDA_VISIBLE_DEVICES=0 bash scripts/01_data_attribution_unprompted.sh
EXPERIMENT_TAG=experiment1 ALGORITHMS="das" TOPK=5000 bash scripts/02_metric_counterfactual_unprompted.sh
EXPERIMENT_TAG=experiment1 ALGORITHMS="das" LDS_M=100 LDS_SUBSET_SIZE=5000 bash scripts/03_metric_lds_unprompted.sh
```

Default unprompted model path:

```text
result/experiment1/model/unprompted_jax/seed_42_epoch_0100.ckpt
```

`01_data_attribution_unprompted.sh` only reads the sample and trajectory created
by `00_sample_unprompted.sh`; it does not run sampling automatically.

## Unprompted Traj TracIn Range Split

```bash
cd diffusion_jax_refined/artbench

EXPERIMENT_TAG=experiment1 TRAIN_SEED=42 CUDA_VISIBLE_DEVICES=0 bash scripts/00_train_unprompted.sh
EXPERIMENT_TAG=experiment1 TRAIN_SEED=42 SAMPLE_SEEDS=0 CUDA_VISIBLE_DEVICES=0 bash scripts/00_sample_unprompted.sh

EXPERIMENT_TAG=experiment1 TRAIN_SEED=42 ALGORITHMS="traj_tracin" \
ATTRIBUTION_RANGES="1-2500,2501-5000,5001-7500,7501-10000" \
CUDA_VISIBLE_DEVICES=0 bash scripts/01_data_attribution_unprompted.sh

EXPERIMENT_TAG=experiment1 ALGORITHMS="traj_tracin" \
ATTRIBUTION_RANGES="1-2500,2501-5000,5001-7500,7501-10000" \
TOPK=5000 bash scripts/02_metric_counterfactual_unprompted.sh

EXPERIMENT_TAG=experiment1 ALGORITHMS="traj_tracin" \
ATTRIBUTION_RANGES="1-2500,2501-5000,5001-7500,7501-10000" \
LDS_M=100 LDS_SUBSET_SIZE=5000 bash scripts/03_metric_lds_unprompted.sh
```
