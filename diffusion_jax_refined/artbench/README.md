# ArtBench latent pipeline

Defaults: query `baroque`, train seed `42`, experiment `experiment1`.

```bash
cd diffusion_jax_refined/artbench

EXPERIMENT_TAG=experiment1 TRAIN_SEED=42 CUDA_VISIBLE_DEVICES=0 \
bash scripts/00_train.sh

EXPERIMENT_TAG=experiment1 QUERY=baroque SAMPLE_SEEDS=0,1,2 \
CUDA_VISIBLE_DEVICES=0 bash scripts/00_sample.sh

EXPERIMENT_TAG=experiment1 QUERY=baroque INITIAL_SEED=0 \
ALGORITHMS="das dtrak end_tracin traj_tracin" \
CUDA_VISIBLE_DEVICES=0 bash scripts/01_data_attribution.sh
```

These four engines read a saved sample. `QUERY` selects its default folder; the
prompt stored in sample metadata becomes the attribution condition. Select an
existing sample with `ATTRIBUTION_SAMPLE_DIR=/absolute/path`. `INITIAL_SEED`
selects the saved sample seed. Attribution scores are grouped under
`result/<experiment>/attribution_score/<SAMPLE_MODEL_MODE>/train_seed_<TRAIN_SEED>/query_<query>/initial_seed_<seed>/`.
Unprompted scores use `unprompted/initial_seed_<seed>/` instead of a query
folder.
LDS eval outputs are grouped under
`result/<experiment>/eval/<SAMPLE_MODEL_MODE>/query_<query>/initial_seed_<seed>/lds/`.
`journey_trak` constructs its trajectory internally.

Traj TracIn ranges:

```bash
QUERY=baroque INITIAL_SEED=0 ALGORITHMS="traj_tracin" \
ATTRIBUTION_RANGES="1-2500,2501-5000,5001-7500,7501-10000" \
CUDA_VISIBLE_DEVICES=0 bash scripts/01_data_attribution.sh
```

Unprompted:

```bash
TRAIN_SEED=42 CUDA_VISIBLE_DEVICES=0 bash scripts/00_train_unprompted.sh
SAMPLE_SEEDS=0 CUDA_VISIBLE_DEVICES=0 bash scripts/00_sample_unprompted.sh
INITIAL_SEED=0 ALGORITHMS="das traj_tracin dtrak end_tracin journey_trak" \
CUDA_VISIBLE_DEVICES=0 bash scripts/01_data_attribution_unprompted.sh
```

Raw images belong in `diffusion_jax_refined/dataset/artbench/raw/`; latent
caches use `dataset/artbench/latents/artbench256/`.

Full counterfactual and LDS evaluation is currently CIFAR-specific.
