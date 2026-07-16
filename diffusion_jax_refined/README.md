# diffusion_jax_refined

Independent experiment scaffold for diffusion data-attribution runs. This is the
maintained framework for new experiments.

The tree is organized by dataset first, then by data-attribution algorithm:

- `cifar2`
- `cifar10`
- `artbench`

Each dataset owns:

- `dataset_config.py`: shared defaults, paths, experiment tags, and per-algorithm config dictionaries.
- `data_attribution/<algorithm>/CONFIG.py`: algorithm-local config entrypoint.
- `data_attribution/<algorithm>/01_train_datapoint_gradient.py`: train-dataset gradient/features stage.
- `data_attribution/<algorithm>/02_query_gradient.py`: query/sample gradient/features stage.
- `data_attribution/<algorithm>/03_score.py`: score-combination stage.
- `sampling/`, `counterfactual/`, `lds/`: separate metric or helper entrypoints.
- `scripts/`: convenience shell commands for training, attribution, counterfactual, and LDS.
- `result/<experiment_tag>/model`, `result/<experiment_tag>/attribution_score`, `result/<experiment_tag>/eval`: output layout.

All dataset file links are centralized under `dataset/`:

- `dataset/cifar2/cifar-10-batches-py`
- `dataset/cifar2/hf_cifar10`
- `dataset/cifar2/indices/lds-val`
- `dataset/cifar10/cifar-10-batches-py`
- `dataset/cifar10/hf_cifar10`
- `dataset/cifar10/indices/lds-val`
- `dataset/artbench/latents/artbench256`
- `dataset/artbench/hf_artbench`
- `dataset/artbench/indices`

Move or symlink real data into these directories. The configs point here by default.

Supported attribution folders:

- `das`
- `traj_tracin`
- `dtrak`
- `end_tracin`
- `journey_trak`

The legacy JAX engines are vendored under `legacy_jax/` so this refine folder can run the attribution engines locally. Large assets such as checkpoints, attribution samples, and latent caches are linked or stored separately to avoid duplicating multi-GB artifacts.

Prompted JAX training is the default training path and matches the legacy
attribution engines. It writes checkpoints to
`result/<experiment>/model/prompted_jax/`, and attribution, sampling,
counterfactual, and LDS read the checkpoint from that same experiment folder.
Use one `TRAIN_SEED` per `EXPERIMENT_TAG` unless you intentionally create a new
experiment tag. The unprompted JAX track uses the same engines with
`class_cond=False`; see
`README_UNPROMPTED.md`.

The `dataset/.../hf_*/train/` folders are input datasets in HuggingFace `load_from_disk` format. They are not training outputs. All generated artifacts should live under each dataset's `result/<experiment>/` folder.

## Typical use

```bash
cd diffusion_jax_refined/cifar2

# Pick a GPU with CUDA=0, CUDA=1, or CUDA_VISIBLE_DEVICES=0.

# Training framework: experiment tags still write under result/<experiment>/.
# The selector accepts prompted_solo, prompted_multi, unprompted_solo, or
# unprompted_multi. Solo/multi means single-GPU vs multi-GPU launch style;
# query/prompt labels are for sampling later, not for training.
EXPERIMENT_TAG=experiment1 TRAIN_SEED=42 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/script_0.sh prompted_solo

EXPERIMENT_TAG=experiment1 TRAIN_SEED=43 CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/script_0.sh prompted_multi

EXPERIMENT_TAG=experiment1 TRAIN_SEED=42 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/script_0.sh unprompted_solo

EXPERIMENT_TAG=experiment1 TRAIN_SEED=43 CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/script_0.sh unprompted_multi

# Backward-compatible aliases still route into script_0/new train modes.
CUDA_VISIBLE_DEVICES=0,1 bash scripts/00_train.sh
CUDA_VISIBLE_DEVICES=0,1 bash scripts/00_train_prompted_jax.sh

# Generate attribution sample trajectories from result/<experiment>/model/prompted_jax.
# One model can produce many samples; attribution_sample_seed/index in dataset_config.py
# selects which sample a given attribution run reads.
CUDA_VISIBLE_DEVICES=0 SAMPLE_SEEDS=0,1,2 SAMPLE_TRAJECTORY_STEPS=100 bash scripts/00_sample.sh

# Run attribution for one saved query/initial-seed pair.
QUERY=horse INITIAL_SEED=0 CUDA_VISIBLE_DEVICES=0 bash scripts/01_data_attribution.sh

# Run only DAS and trajectory TracIn.
CUDA_VISIBLE_DEVICES=1 ALGORITHMS="das traj_tracin" bash scripts/01_data_attribution.sh

# Optional training + datapoint-gradient/feature computation entrypoint.
# TRAIN_MODES is optional; omit it to reuse existing checkpoints.
# ALGORITHMS/ALGO is optional and accepts spaces or commas.
TRAIN_MODES="prompted_solo unprompted_solo" \
DATAPOINT_GRADIENT_MODES="both" \
ALGORITHMS="das,dtrak" \
CUDA_VISIBLE_DEVICES=0 bash scripts/01_datapoint_gradients.sh

This writes stage-1 outputs next to the model identity, for example:

```text
<dataset>/result/<experiment>/model/prompted_solo/seed_<TRAIN_SEED>_train_gradient/das/
<dataset>/result/<experiment>/model/unprompted_solo/seed_<TRAIN_SEED>_train_gradient/das/
```

# Split trajectory TracIn scoring into multiple index ranges.
CUDA_VISIBLE_DEVICES=0 ALGORITHMS="traj_tracin" ATTRIBUTION_RANGES="1-2500,2501-5000,5001-7500,7501-10000" bash scripts/01_data_attribution.sh

# Run counterfactual metric for DAS.
QUERY=horse INITIAL_SEED=0 CUDA_VISIBLE_DEVICES=0 ALGORITHMS="das" TOPK=5000 bash scripts/02_metric_counterfactual.sh

# Counterfactual automatically combines trajectory TracIn range outputs when ATTRIBUTION_RANGES is set.
CUDA_VISIBLE_DEVICES=0 ALGORITHMS="traj_tracin" ATTRIBUTION_RANGES="1-2500,2501-5000,5001-7500,7501-10000" TOPK=5000 bash scripts/02_metric_counterfactual.sh

# Run LDS for DAS.
QUERY=horse INITIAL_SEED=0 CUDA_VISIBLE_DEVICES=0 ALGORITHMS="das" LDS_M=100 LDS_SUBSET_SIZE=5000 bash scripts/03_metric_lds.sh

# LDS also combines trajectory TracIn range outputs with the same ATTRIBUTION_RANGES value.
CUDA_VISIBLE_DEVICES=0 ALGORITHMS="traj_tracin" ATTRIBUTION_RANGES="1-2500,2501-5000,5001-7500,7501-10000" LDS_M=100 LDS_SUBSET_SIZE=5000 bash scripts/03_metric_lds.sh

# Run train, attribution, counterfactual, and LDS in sequence.
CUDA_VISIBLE_DEVICES=0 ALGORITHMS="das" bash scripts/04_all.sh 0 0
```

## Unprompted Comparison Model

Unprompted is a full no-prompt JAX comparison path. It uses the same UNet
implementation, sampler, trajectory files, and five attribution engines as
prompted; only conditioning and artifact paths differ.

```bash
cd diffusion_jax_refined/cifar2

EXPERIMENT_TAG=experiment1 TRAIN_SEED=42 CUDA_VISIBLE_DEVICES=0 bash scripts/00_train_unprompted.sh
EXPERIMENT_TAG=experiment1 TRAIN_SEED=42 SAMPLE_SEEDS=0 CUDA_VISIBLE_DEVICES=0 bash scripts/00_sample_unprompted.sh
EXPERIMENT_TAG=experiment1 TRAIN_SEED=42 ALGORITHMS="das traj_tracin dtrak end_tracin journey_trak" CUDA_VISIBLE_DEVICES=0 bash scripts/01_data_attribution_unprompted.sh
EXPERIMENT_TAG=experiment1 ALGORITHMS="das" TOPK=5000 bash scripts/02_metric_counterfactual_unprompted.sh
EXPERIMENT_TAG=experiment1 ALGORITHMS="das" LDS_M=100 LDS_SUBSET_SIZE=5000 bash scripts/03_metric_lds_unprompted.sh
```

The default unprompted comparison path is:

```text
<dataset>/result/<experiment>/model/unprompted_jax/
```

Training does not create prompt-specific model folders. The dataset split is
defined by the dataset config, and checkpoint files are distinguished by
`TRAIN_SEED`:

```text
<dataset>/result/<experiment>/model/prompted_jax/seed_<TRAIN_SEED>_epoch_<epoch>.ckpt
<dataset>/result/<experiment>/model/unprompted_jax/seed_<TRAIN_SEED>_epoch_<epoch>.ckpt
```

Machine-family launch wrappers live under:

```bash
DATASET=cifar2 TRAIN_MODE=prompted_solo TRAIN_SEED=42 GPU_IDS=0 \
  bash ../tacc/h100/script_0.sh

DATASET=cifar2 TRAIN_MODES="prompted_solo unprompted_solo" TRAIN_SEED=42 GPU_IDS=0 \
  bash ../tacc/h100/script_0.sh

DATASET=cifar10 TRAIN_MODE=prompted_multi TRAIN_SEED=42 GPU_IDS=0,1,2,3 \
  bash ../tacc/vista/script_0.sh
```

Sampling is deliberately separate. `01_data_attribution_unprompted.sh` never
generates a query; it expects the sample and trajectory produced by
`00_sample_unprompted.sh`.

To spread trajectory TracIn across multiple GPUs, launch different ranges in separate terminals:

```bash
CUDA_VISIBLE_DEVICES=0 ALGORITHMS="traj_tracin" ATTRIBUTION_RANGES="1-2500,2501-5000" bash scripts/01_data_attribution.sh
CUDA_VISIBLE_DEVICES=1 ALGORITHMS="traj_tracin" ATTRIBUTION_RANGES="5001-7500,7501-10000" bash scripts/01_data_attribution.sh
```

Then evaluate all four range outputs:

```bash
CUDA_VISIBLE_DEVICES=0 ALGORITHMS="traj_tracin" ATTRIBUTION_RANGES="1-2500,2501-5000,5001-7500,7501-10000" TOPK=5000 bash scripts/02_metric_counterfactual.sh
CUDA_VISIBLE_DEVICES=0 ALGORITHMS="traj_tracin" ATTRIBUTION_RANGES="1-2500,2501-5000,5001-7500,7501-10000" LDS_M=100 LDS_SUBSET_SIZE=5000 bash scripts/03_metric_lds.sh
```

If you already know the exact output folders, bypass range inference with comma-separated `ATTRIBUTION_RESULT_DIRS`.

Set `EXPERIMENT_TAG=experiment2` or `experiment3` to route outputs into a different experiment folder.

## Where To Change Common Settings

- Training seed/model identity: `TRAIN_SEED=42` at launch time.
- Experiment output folder: `EXPERIMENT_TAG=experiment1`.
- Sampling/attribution query: `QUERY` in `<dataset>/dataset_config.py`.
- Training checkpoint path: `REFERENCE_CKPT` and `CHECKPOINT_DIR` in `<dataset>/dataset_config.py`.
- Number of generated samples: `SAMPLE_SEEDS=0,1,2,...` in `scripts/00_sample.sh`.
- Which generated sample attribution reads: `attribution_sample_seed` and `attribution_sample_index` in `<dataset>/dataset_config.py`.
- Traj TracIn score ranges: `ATTRIBUTION_RANGES` or `SCORE_INDEX_RANGES`.
- Counterfactual removal size: `TOPK`.
- LDS subset sweep: `LDS_M`, `LDS_SUBSET_SIZE`, and `LDS_SUBSET_SEED`.
- Unprompted sample seeds: `SAMPLE_SEEDS=0,1,2,...` in `scripts/00_sample_unprompted.sh`.
- Unprompted checkpoint identity: `TRAIN_SEED` and `JAX_EPOCHS`; use the same values for training, sampling, and attribution.

## Current Counterfactual And LDS Behavior

Prompted counterfactual is implemented by the legacy CIFAR retraining engine:
`legacy_jax/DM_counterfactual_retrain_from_attribution.py`. It loads the selected
attribution score folder(s), removes the top `TOPK` training indices, retrains a
CIFAR JAX model without those rows, and writes evaluation artifacts under
`result/<experiment>/eval/counterfactual/<algorithm>/`.

Prompted LDS is implemented by `legacy_jax/LDS/DM_cifar_lds.py`. It builds
`LDS_M` random subsets of size `LDS_SUBSET_SIZE`, compares attribution-predicted
subset influence against the retrained/evaluated subset target, and writes CSV,
summary JSON, and scatter plots under `result/<experiment>/eval/lds/<algorithm>/`.

Trajectory TracIn and LDS use the matched `noise_trajectory` objective by
default: `sum_k w_k ||eps_theta(x_ref_k,k) -
eps_theta_ref(x_ref_k,k)||^2`. The evaluator reuses the attribution sample
seed/index and trajectory timesteps, and compares every subset checkpoint with
the same full/reference checkpoint. The reference checkpoint itself contributes
zero query gradient, so the attribution checkpoint directory must also contain
earlier training checkpoints.

Both engines can combine split `traj_tracin` outputs when the same
`ATTRIBUTION_RANGES` are passed to the metric scripts.

When any loaded attribution score is negative, LDS additionally evaluates the
same subsets after squaring every datapoint score. That variant writes only
`lds_results_squared_scores.csv` and `lds_scatter_squared_scores.png`.

ArtBench currently uses the latent model training path. Training calls
`legacy_jax/DM__training_ARTBENCH_latent.py`, which trains/reuses an autoencoder,
caches ArtBench image latents, and trains latent diffusion in the same training
run. Full prompted counterfactual/LDS remains CIFAR-specific; ArtBench metric
folders are scaffolded for compatibility until the ArtBench metric engine is
added.
