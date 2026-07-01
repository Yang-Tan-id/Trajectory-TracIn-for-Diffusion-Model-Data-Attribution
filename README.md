# Trajectory TracIn for Diffusion Data Attribution

Maintained code lives in `diffusion_jax_refined/`. Run experiments from a
dataset directory: `cifar10`, `cifar2`, or `artbench`.

## Prompted workflow

```bash
cd diffusion_jax_refined/cifar10

EXPERIMENT_TAG=experiment1 TRAIN_SEED=42 CUDA_VISIBLE_DEVICES=0 \
bash scripts/00_train.sh

EXPERIMENT_TAG=experiment1 QUERY=truck SAMPLE_SEEDS=0,1,2 \
CUDA_VISIBLE_DEVICES=0 bash scripts/00_sample.sh

EXPERIMENT_TAG=experiment1 QUERY=truck ALGORITHMS="das traj_tracin" \
CUDA_VISIBLE_DEVICES=0 bash scripts/01_data_attribution.sh
```

For `das`, `dtrak`, `end_tracin`, and `traj_tracin`, attribution consumes a
saved sample; it never samples implicitly. `QUERY` selects the default sample
folder. After loading, the prompt in `seed_info.json`/`manifest.json` is
authoritative and becomes the attribution condition. Select an existing sample
directly with:

```bash
ATTRIBUTION_SAMPLE_DIR="/absolute/path/to/model_..."
```

`journey_trak` is the exception: it constructs its query trajectory internally.
Sample seed/index defaults are in each dataset's `dataset_config.py`.

## Traj TracIn ranges

Split expensive scoring with `ATTRIBUTION_RANGES`:

```bash
QUERY=truck ALGORITHMS="traj_tracin" \
ATTRIBUTION_RANGES="1-10000,10001-20000,20001-30000,30001-40000,40001-50000" \
CUDA_VISIBLE_DEVICES=0 bash scripts/01_data_attribution.sh
```

Different terminals/GPUs may process disjoint ranges. Pass the complete range
list to evaluation so the score files are combined.

## Counterfactual and LDS

```bash
ALGORITHMS="das" TOPK=5000 bash scripts/02_metric_counterfactual.sh
```

LDS model training is independent of evaluation. It inherits the reference
checkpoint's training config and only requires `m`, `k`, and a subset seed:

```bash
LDS_M=100 LDS_K=5000 LDS_SAMPLE_RANDOM_SEED=0 \
bash scripts/03_lds_training.sh
```

Evaluate one or more reusable LDS model folders:

```bash
ALGORITHMS="das traj_tracin" \
LDS_MODEL_DIRS="\
  result/experiment1/lds_model/m_50_k_5000_seed_0,
  result/experiment1/lds_model/m_50_k_5000_seed_1" \
bash scripts/04_lds_eval.sh
```

Prompted paths:

```text
result/<experiment>/model/prompted_jax/
result/<experiment>/lds_model/m_<m>_k_<k>_seed_<seed>/
result/<experiment>/eval/lds/<algorithm>/
```

Traj TracIn and LDS use the same saved trajectory and objective:

```text
mean_k ||eps_theta(x_k,t_k,q) - eps_ref(x_k,t_k,q)||^2
```

## Unprompted workflow

```bash
EXPERIMENT_TAG=experiment1 TRAIN_SEED=42 CUDA_VISIBLE_DEVICES=0 \
bash scripts/00_train_unprompted.sh

EXPERIMENT_TAG=experiment1 SAMPLE_SEEDS=0 CUDA_VISIBLE_DEVICES=0 \
bash scripts/00_sample_unprompted.sh

EXPERIMENT_TAG=experiment1 \
ALGORITHMS="das traj_tracin dtrak end_tracin journey_trak" \
CUDA_VISIBLE_DEVICES=0 bash scripts/01_data_attribution_unprompted.sh

LDS_M=100 LDS_K=5000 LDS_SAMPLE_RANDOM_SEED=0 \
bash scripts/03_lds_training_unprompted.sh

ALGORITHMS="das traj_tracin" \
LDS_MODEL_DIRS="result/experiment1/lds_model/unprompted/m_100_k_5000_seed_0" \
bash scripts/04_lds_eval_unprompted.sh
```

See `diffusion_jax_refined/README_UNPROMPTED.md` for details.

## Outputs and dataset notes

```text
diffusion_jax_refined/<dataset>/result/<experiment>/
├── model/
├── attribution_score/
├── lds_model/
└── eval/
```

- One `EXPERIMENT_TAG` should normally use one `TRAIN_SEED`.
- CIFAR10 and CIFAR2 support full counterfactual/LDS evaluation.
- ArtBench uses the latent pipeline; full counterfactual/LDS remains
  CIFAR-specific.
- Dataset-specific defaults and commands are documented in each dataset README.
