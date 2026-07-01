# CIFAR2 (horse/automobile)

Defaults: classes `horse, automobile`, query `horse`, train seed `42`,
experiment `experiment1`.

## Prompted

```bash
cd diffusion_jax_refined/cifar2

EXPERIMENT_TAG=experiment1 TRAIN_SEED=42 CUDA_VISIBLE_DEVICES=0 \
bash scripts/00_train.sh

EXPERIMENT_TAG=experiment1 QUERY="horse,automobile" SAMPLE_SEEDS=0,1,2 \
CUDA_VISIBLE_DEVICES=0 bash scripts/00_sample.sh

EXPERIMENT_TAG=experiment1 QUERY="horse,automobile" \
ALGORITHMS="das dtrak end_tracin traj_tracin" \
CUDA_VISIBLE_DEVICES=0 bash scripts/01_data_attribution.sh
```

The four listed engines read a saved sample. `QUERY` selects its default folder;
the prompt stored in sample metadata becomes the actual attribution condition.
Choose another existing sample with `ATTRIBUTION_SAMPLE_DIR=/absolute/path`.
Sample seed/index defaults are in `dataset_config.py`. `journey_trak` constructs
its trajectory internally.

Traj TracIn range split:

```bash
QUERY="horse,automobile" ALGORITHMS="traj_tracin" \
ATTRIBUTION_RANGES="1-2500,2501-5000,5001-7500,7501-10000" \
CUDA_VISIBLE_DEVICES=0 bash scripts/01_data_attribution.sh
```

## LDS

```bash
LDS_M=50 LDS_K=5000 LDS_SAMPLE_RANDOM_SEED=0 \
bash scripts/03_lds_training.sh

LDS_M=50 LDS_K=5000 LDS_SAMPLE_RANDOM_SEED=1 \
bash scripts/03_lds_training.sh

ALGORITHMS="das traj_tracin" \
LDS_MODEL_DIRS="\
  result/experiment1/lds_model/m_50_k_5000_seed_0,
  result/experiment1/lds_model/m_50_k_5000_seed_1" \
bash scripts/04_lds_eval.sh
```

## Unprompted

```bash
TRAIN_SEED=42 CUDA_VISIBLE_DEVICES=0 bash scripts/00_train_unprompted.sh
SAMPLE_SEEDS=0 CUDA_VISIBLE_DEVICES=0 bash scripts/00_sample_unprompted.sh
ALGORITHMS="das traj_tracin dtrak end_tracin journey_trak" \
CUDA_VISIBLE_DEVICES=0 bash scripts/01_data_attribution_unprompted.sh

LDS_M=100 LDS_K=5000 LDS_SAMPLE_RANDOM_SEED=0 \
bash scripts/03_lds_training_unprompted.sh

ALGORITHMS="das traj_tracin" \
LDS_MODEL_DIRS="result/experiment1/lds_model/unprompted/m_100_k_5000_seed_0" \
bash scripts/04_lds_eval_unprompted.sh
```

Data: `diffusion_jax_refined/dataset/cifar2/cifar-10-batches-py/`.
Change defaults in `dataset_config.py`.
