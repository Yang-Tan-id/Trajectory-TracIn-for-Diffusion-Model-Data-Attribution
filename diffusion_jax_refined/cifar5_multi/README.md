# CIFAR5 Multi

Generated 64x64 CIFAR composites with three objects in three of four quadrants.
Labels are unordered 5-way multi-hot vectors over:

`bird, horse, automobile, dog, cat`

## Data

Generate the default 10k split:

```bash
cd diffusion_jax_refined/cifar5_multi
python script/generate_cifar5_multi.py --size 10000 --seed 0
```

This writes:

`diffusion_jax_refined/dataset/cifar5_multi/10000/dataset.npz`

Set another size with `--size` or at runtime with `CIFAR5_MULTI_SIZE`.

## End-To-End Driver

Print the full experiment plan:

```bash
python script/run_cifar5_multi_experiment.py
```

Run it:

```bash
python script/run_cifar5_multi_experiment.py --execute
```

The driver runs:

- dataset generation
- prompted and unprompted JAX training
- two random prompted 3-label queries plus one unprompted query
- DAS, Traj TracIn, and End TracIn attribution
- 25% LDS smoke test with three subset seeds and 64 subsets each
- LDS eval for `noise_trajectory`, `endpoint_counterfactual`, `traj_counterfactual`, and `simple_loss`

Defaults match the requested setup: `DAS_PROJ_DIM=4096`, `DTRAK_PROJ_DIM=4096`,
and shared train-gradient Traj TracIn is requested through
`TRACIN_USE_SHARED_TRAIN_GRADIENT=1`.

Results are under:

`diffusion_jax_refined/cifar5_multi/result/<EXPERIMENT_TAG>/`
