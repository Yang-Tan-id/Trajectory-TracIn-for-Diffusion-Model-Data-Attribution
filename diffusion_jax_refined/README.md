# diffusion das refine

Independent experiment scaffold for diffusion data-attribution runs.

The tree is organized by dataset first, then by data-attribution algorithm:

- `cifar2`
- `cifar10`
- `artbench`

Each dataset owns:

- `dataset_config.py`: shared defaults, paths, experiment tags, and per-algorithm config dictionaries.
- `data_attribution/<algorithm>/CONFIG.py`: algorithm-local config entrypoint.
- `data_attribution/<algorithm>/run_attribution.py`: calls the legacy JAX attribution engine with that config.
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

Prompted JAX training is the default training path and matches the legacy attribution engines. It writes checkpoints to `result/<experiment>/model/prompted_jax/`, and attribution, sampling, counterfactual, and LDS read the checkpoint from that same experiment folder. For the separate unprompted diffusers training track, see `README_UNPROMPTED.md`.

The `dataset/.../hf_*/train/` folders are input datasets in HuggingFace `load_from_disk` format. They are not training outputs. All generated artifacts should live under each dataset's `result/<experiment>/` folder.

## Typical use

```bash
cd "diffusion das refine/cifar2"

# Pick a GPU with CUDA=0, CUDA=1, or CUDA_VISIBLE_DEVICES=0.

# Run the original prompted JAX training.
CUDA_VISIBLE_DEVICES=0 bash scripts/00_train.sh 0 0

# Same explicit entrypoint for prompted JAX training.
CUDA_VISIBLE_DEVICES=0 bash scripts/00_train_prompted_jax.sh

# Generate attribution sample trajectories from result/<experiment>/model/prompted_jax.
CUDA_VISIBLE_DEVICES=0 bash scripts/00_sample.sh

# Or run from an individual attribution folder.
(cd data_attribution/traj_tracin && CUDA_VISIBLE_DEVICES=0 bash script.sh train)

# Run attribution for all algorithms.
CUDA_VISIBLE_DEVICES=0 bash scripts/01_data_attribution.sh

# Run only DAS and trajectory TracIn.
CUDA_VISIBLE_DEVICES=1 ALGORITHMS="das traj_tracin" bash scripts/01_data_attribution.sh

# Split trajectory TracIn scoring into multiple index ranges.
CUDA_VISIBLE_DEVICES=0 ALGORITHMS="traj_tracin" ATTRIBUTION_RANGES="1-2500,2501-5000,5001-7500,7501-10000" bash scripts/01_data_attribution.sh

# Run counterfactual metric for DAS.
CUDA_VISIBLE_DEVICES=0 ALGORITHMS="das" TOPK=5000 bash scripts/02_metric_counterfactual.sh

# Counterfactual automatically combines trajectory TracIn range outputs when ATTRIBUTION_RANGES is set.
CUDA_VISIBLE_DEVICES=0 ALGORITHMS="traj_tracin" ATTRIBUTION_RANGES="1-2500,2501-5000,5001-7500,7501-10000" TOPK=5000 bash scripts/02_metric_counterfactual.sh

# Run LDS for DAS.
CUDA_VISIBLE_DEVICES=0 ALGORITHMS="das" LDS_M=100 LDS_SUBSET_SIZE=5000 bash scripts/03_metric_lds.sh

# LDS also combines trajectory TracIn range outputs with the same ATTRIBUTION_RANGES value.
CUDA_VISIBLE_DEVICES=0 ALGORITHMS="traj_tracin" ATTRIBUTION_RANGES="1-2500,2501-5000,5001-7500,7501-10000" LDS_M=100 LDS_SUBSET_SIZE=5000 bash scripts/03_metric_lds.sh

# Run train, attribution, counterfactual, and LDS in sequence.
CUDA_VISIBLE_DEVICES=0 ALGORITHMS="das" bash scripts/04_all.sh 0 0
```

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
