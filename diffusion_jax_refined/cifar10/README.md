# CIFAR10 refine runs

Default setup:

- Dataset: full CIFAR-10
- Prompt/query: `truck`
- Default training seed: `TRAIN_SEED=42`
- Default experiment: `EXPERIMENT_TAG=experiment1`

Set `QUERY` once (or change its default in `dataset_config.py`). The sampling
prompt and attribution sample folder are derived from it automatically; do not
edit `ATTRIBUTION_SAMPLE_DIR` separately.

## Prompted JAX Flow

Prompted JAX is the conditional model path. One experiment should normally have
one training seed and one prompted checkpoint.

```bash
cd diffusion_jax_refined/cifar10

EXPERIMENT_TAG=experiment1 TRAIN_SEED=42 CUDA_VISIBLE_DEVICES=0 bash scripts/00_train.sh
EXPERIMENT_TAG=experiment1 SAMPLE_SEEDS=0,1,2 CUDA_VISIBLE_DEVICES=0 bash scripts/00_sample.sh
EXPERIMENT_TAG=experiment1 ALGORITHMS="das end_tracin dtrak journey_trak" CUDA_VISIBLE_DEVICES=0 bash scripts/01_data_attribution.sh
EXPERIMENT_TAG=experiment1 ALGORITHMS="das" TOPK=5000 CUDA_VISIBLE_DEVICES=0 bash scripts/02_metric_counterfactual.sh
EXPERIMENT_TAG=experiment1 LDS_M=50 LDS_K=5000 LDS_SAMPLE_RANDOM_SEED=0 CUDA_VISIBLE_DEVICES=0 bash scripts/03_lds_training.sh
EXPERIMENT_TAG=experiment1 LDS_M=50 LDS_K=5000 LDS_SAMPLE_RANDOM_SEED=1 CUDA_VISIBLE_DEVICES=0 bash scripts/03_lds_training.sh
EXPERIMENT_TAG=experiment1 ALGORITHMS="das" \
LDS_MODEL_DIRS="\
  result/experiment1/lds_model/m_50_k_5000_seed_0,
  result/experiment1/lds_model/m_50_k_5000_seed_1" \
CUDA_VISIBLE_DEVICES=0 bash scripts/04_lds_eval.sh
```

Prompted checkpoint path:

```text
result/experiment1/model/prompted_jax/seed_42_epoch_0200.ckpt
```

Generated sample trajectories:

```text
result/experiment1/eval/sampling/
```

## Prompted Traj TracIn Range Split

Trajectory TracIn should usually be split by score-index ranges:

```bash
cd diffusion_jax_refined/cifar10

EXPERIMENT_TAG=experiment1 ALGORITHMS="traj_tracin" \
ATTRIBUTION_RANGES="1-10000,10001-20000,20001-30000,30001-40000,40001-50000" \
CUDA_VISIBLE_DEVICES=0 bash scripts/01_data_attribution.sh
```

For multiple GPUs, run disjoint ranges in separate terminals:

```bash
EXPERIMENT_TAG=experiment1 ALGORITHMS="traj_tracin" ATTRIBUTION_RANGES="1-10000,10001-20000" bash scripts/01_data_attribution.sh
CUDA_VISIBLE_DEVICES=1 EXPERIMENT_TAG=experiment1 ALGORITHMS="traj_tracin" ATTRIBUTION_RANGES="20001-30000,30001-40000,40001-50000" bash scripts/01_data_attribution.sh
```

Use the same range list when evaluating:

```bash
EXPERIMENT_TAG=experiment1 ALGORITHMS="traj_tracin" ATTRIBUTION_RANGES="1-10000,10001-20000,20001-30000,30001-40000,40001-50000" TOPK=5000 bash scripts/02_metric_counterfactual.sh
EXPERIMENT_TAG=experiment1 LDS_M=50 LDS_K=5000 LDS_SAMPLE_RANDOM_SEED=0 bash scripts/03_lds_training.sh
EXPERIMENT_TAG=experiment1 LDS_M=50 LDS_K=5000 LDS_SAMPLE_RANDOM_SEED=1 bash scripts/03_lds_training.sh
EXPERIMENT_TAG=experiment1 ALGORITHMS="traj_tracin" \
ATTRIBUTION_RANGES="1-10000,10001-20000,20001-30000,30001-40000,40001-50000" \
LDS_MODEL_DIRS="\
  result/experiment1/lds_model/m_50_k_5000_seed_0,
  result/experiment1/lds_model/m_50_k_5000_seed_1" \
bash scripts/04_lds_eval.sh
```

## Unprompted JAX Flow

Unprompted uses the same JAX UNet, sample format, and five attribution engines
as prompted, with `class_cond=False`. Training, sampling, attribution, and
evaluation are separate commands.

```bash
cd diffusion_jax_refined/cifar10

EXPERIMENT_TAG=experiment1 TRAIN_SEED=42 CUDA_VISIBLE_DEVICES=0 bash scripts/00_train_unprompted.sh
EXPERIMENT_TAG=experiment1 TRAIN_SEED=42 SAMPLE_SEEDS=0 CUDA_VISIBLE_DEVICES=0 bash scripts/00_sample_unprompted.sh
EXPERIMENT_TAG=experiment1 TRAIN_SEED=42 ALGORITHMS="das traj_tracin dtrak end_tracin journey_trak" CUDA_VISIBLE_DEVICES=0 bash scripts/01_data_attribution_unprompted.sh
EXPERIMENT_TAG=experiment1 ALGORITHMS="das" TOPK=5000 bash scripts/02_metric_counterfactual_unprompted.sh
EXPERIMENT_TAG=experiment1 LDS_M=50 LDS_K=5000 LDS_SAMPLE_RANDOM_SEED=0 bash scripts/03_lds_training_unprompted.sh
EXPERIMENT_TAG=experiment1 LDS_M=50 LDS_K=5000 LDS_SAMPLE_RANDOM_SEED=1 bash scripts/03_lds_training_unprompted.sh
EXPERIMENT_TAG=experiment1 ALGORITHMS="das" \
LDS_MODEL_DIRS="\
  result/experiment1/lds_model/unprompted/m_50_k_5000_seed_0,
  result/experiment1/lds_model/unprompted/m_50_k_5000_seed_1" \
bash scripts/04_lds_eval_unprompted.sh
```

Default unprompted model path:

```text
result/experiment1/model/unprompted_jax/seed_42_epoch_0200.ckpt
```

`01_data_attribution_unprompted.sh` only reads the sample and trajectory created
by `00_sample_unprompted.sh`; it does not run sampling automatically.

## Unprompted Traj TracIn Range Split

```bash
cd diffusion_jax_refined/cifar10

EXPERIMENT_TAG=experiment1 TRAIN_SEED=42 CUDA_VISIBLE_DEVICES=0 bash scripts/00_train_unprompted.sh
EXPERIMENT_TAG=experiment1 TRAIN_SEED=42 SAMPLE_SEEDS=0 CUDA_VISIBLE_DEVICES=0 bash scripts/00_sample_unprompted.sh

EXPERIMENT_TAG=experiment1 TRAIN_SEED=42 ALGORITHMS="traj_tracin" \
ATTRIBUTION_RANGES="1-10000,10001-20000,20001-30000,30001-40000,40001-50000" \
CUDA_VISIBLE_DEVICES=0 bash scripts/01_data_attribution_unprompted.sh

EXPERIMENT_TAG=experiment1 ALGORITHMS="traj_tracin" \
ATTRIBUTION_RANGES="1-10000,10001-20000,20001-30000,30001-40000,40001-50000" \
TOPK=5000 bash scripts/02_metric_counterfactual_unprompted.sh

EXPERIMENT_TAG=experiment1 LDS_M=50 LDS_K=5000 LDS_SAMPLE_RANDOM_SEED=0 \
bash scripts/03_lds_training_unprompted.sh
EXPERIMENT_TAG=experiment1 LDS_M=50 LDS_K=5000 LDS_SAMPLE_RANDOM_SEED=1 \
bash scripts/03_lds_training_unprompted.sh

EXPERIMENT_TAG=experiment1 ALGORITHMS="traj_tracin" \
ATTRIBUTION_RANGES="1-10000,10001-20000,20001-30000,30001-40000,40001-50000" \
LDS_MODEL_DIRS="\
  result/experiment1/lds_model/unprompted/m_50_k_5000_seed_0,
  result/experiment1/lds_model/unprompted/m_50_k_5000_seed_1" \
bash scripts/04_lds_eval_unprompted.sh
```
