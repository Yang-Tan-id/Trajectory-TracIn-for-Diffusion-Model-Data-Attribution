# legacy_jax

This folder contains only the legacy JAX files still used by
`diffusion_jax_refined`.

Kept files:

- `DM__training_CIFAR10_pixel.py`: conditional and unconditional CIFAR/CIFAR2 JAX training.
- `DM__training_ARTBENCH_latent.py`: conditional and unconditional ArtBench latent training.
- `dataset_loader_cifar10.py`: CIFAR loader used by the CIFAR training/metric engines.
- `dataset_loader_artbench_latent.py`: ArtBench image/latent loader.
- `DM___sampler.py` and `DM___data_attribution_sampler.py`: prompted and unprompted sample/trajectory generation.
- `das/`, `dtrak/`, `end_tracin/`, `traj_tracin/`, and `journey_trak/`: shared data-attribution engines. Each folder contains:
  - `algorithm.py`: the maintained legacy implementation.
  - `stages.py`: the three-stage attribution facade: training datapoint gradients/features, query gradients/features, and score calculation.
- `DM_counterfactual_retrain_from_attribution.py`: prompted CIFAR counterfactual retraining metric.
- `LDS/DM_cifar_lds.py`: prompted CIFAR LDS metric.

Removed from the refined copy:

- x3 pixel toy experiment scripts and loaders.
- old manual shell launchers.
- unused Torch/indent prototype file.
- Python bytecode caches and `.DS_Store` files.

Conda environment used historically: `experiment_dm`.

The maintained dataset wrappers live one directory above. Unprompted training
sets `class_cond=False`; the engines otherwise use the same query-gradient and
training-gradient implementations as prompted attribution. Data-attribution
Monte Carlo and expectation sample counts are pinned to 10 by default across
DAS, endpoint TracIn, trajectory TracIn, D-TRAK, and JourneyTRAK.
