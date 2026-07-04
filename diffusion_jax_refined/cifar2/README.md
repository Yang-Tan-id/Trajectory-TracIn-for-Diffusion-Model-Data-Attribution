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

EXPERIMENT_TAG=experiment1_42 QUERY="horse" INITIAL_SEED=0 \
ALGORITHMS="das dtrak end_tracin traj_tracin" \
CUDA_VISIBLE_DEVICES=0 bash scripts/01_data_attribution.sh

EXPERIMENT_TAG=experiment1_42 QUERY="horse" INITIAL_SEED=1 \
ALGORITHMS="das" \
CUDA_VISIBLE_DEVICES=2 bash scripts/01_data_attribution.sh

```

The four listed engines read a saved sample. `QUERY` selects its default folder;
the prompt stored in sample metadata becomes the actual attribution condition.
Choose another existing sample with `ATTRIBUTION_SAMPLE_DIR=/absolute/path`.
`INITIAL_SEED` selects the saved sample seed. Attribution scores are grouped
under `result/<experiment>/attribution_score/query_<query>/initial_seed_<seed>/`.
`journey_trak` constructs its trajectory internally.

Traj TracIn range split:

```bash
QUERY="horse" INITIAL_SEED=0 EXPERIMENT_TAG=experiment1_42 ALGORITHMS="traj_tracin" \
ATTRIBUTION_RANGES="1-2500" \
CUDA_VISIBLE_DEVICES=0 bash scripts/01_data_attribution.sh

QUERY="horse" INITIAL_SEED=0 EXPERIMENT_TAG=experiment1_42 ALGORITHMS="traj_tracin" \
ATTRIBUTION_RANGES="2501-5000" \
CUDA_VISIBLE_DEVICES=1 bash scripts/01_data_attribution.sh

QUERY="horse" INITIAL_SEED=0 EXPERIMENT_TAG=experiment1_42 ALGORITHMS="traj_tracin" \
ATTRIBUTION_RANGES="5001-7500" \
CUDA_VISIBLE_DEVICES=2 bash scripts/01_data_attribution.sh

QUERY="horse" INITIAL_SEED=0 EXPERIMENT_TAG=experiment1_42 ALGORITHMS="traj_tracin" \
ATTRIBUTION_RANGES="7501-10000" \
CUDA_VISIBLE_DEVICES=3 bash scripts/01_data_attribution.sh


```

## LDS

```bash
CUDA_VISIBLE_DEVICES=1 EXPERIMENT_TAG=experiment1_42 LDS_M=50 LDS_K=5000 LDS_SAMPLE_RANDOM_SEED=4 \
bash scripts/03_lds_training.sh

LDS_M=50 LDS_K=5000 LDS_SAMPLE_RANDOM_SEED=1 \
bash scripts/03_lds_training.sh

CUDA_VISIBLE_DEVICES=3 \
EXPERIMENT_TAG="experiment1_42" \
QUERY="horse" INITIAL_SEED=0 \
ALGORITHMS="das" \
LDS_MODEL_DIRS="\
result/experiment1_42/lds_model/m_50_k_5000_seed_42, \
result/experiment1_42/lds_model/m_50_k_5000_seed_24" \
bash scripts/04_lds_eval.sh --target-function noise_trajectory

CUDA_VISIBLE_DEVICES=3 \
ALGORITHMS="traj_tracin" \
EXPERIMENT_TAG="experiment1_42" \
QUERY="horse" INITIAL_SEED=0 \
LDS_MODEL_DIRS="\
result/experiment1_42/lds_model/m_50_k_5000_seed_42,\
result/experiment1_42/lds_model/m_50_k_5000_seed_24" \
bash scripts/04_lds_eval.sh

CUDA_VISIBLE_DEVICES=3 \
ALGORITHMS="traj_tracin" \
EXPERIMENT_TAG="experiment1_42" \
QUERY="horse" INITIAL_SEED=0 \
LDS_MODEL_DIRS="result/experiment1_42/lds_model/m_50_k_5000_seed_42" \
bash scripts/04_lds_eval.sh --target-function simple_loss
```

When `ATTRIBUTION_RESULT_DIRS` is omitted, LDS automatically combines every
matching algorithm/range folder for the selected `QUERY` and `INITIAL_SEED`.

## Unprompted

```bash
TRAIN_SEED=42 CUDA_VISIBLE_DEVICES=0 bash scripts/00_train_unprompted.sh
SAMPLE_SEEDS=0 CUDA_VISIBLE_DEVICES=0 bash scripts/00_sample_unprompted.sh
INITIAL_SEED=0 ALGORITHMS="das traj_tracin dtrak end_tracin journey_trak" \
CUDA_VISIBLE_DEVICES=0 bash scripts/01_data_attribution_unprompted.sh

LDS_M=100 LDS_K=5000 LDS_SAMPLE_RANDOM_SEED=0 \
bash scripts/03_lds_training_unprompted.sh

INITIAL_SEED=0 ALGORITHMS="das traj_tracin" \
LDS_MODEL_DIRS="result/experiment1/lds_model/unprompted/m_100_k_5000_seed_0" \
bash scripts/04_lds_eval_unprompted.sh
```

Data: `diffusion_jax_refined/dataset/cifar2/cifar-10-batches-py/`.
Change defaults in `dataset_config.py`.
