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

It also writes 5 preview samples by default:

`diffusion_jax_refined/dataset/cifar5_multi/10000/samples/`

Change the number of previews with `--preview-count 3`, or disable them with
`--preview-count 0`.

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

Use four GPUs as independent workers:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python script/run_cifar5_multi_experiment.py \
  --execute \
  --experiment cifar5_multi_exp1 \
  --size 10000 \
  --data-seed 0 \
  --train-seed 42 \
  --epochs 200 \
  --lds-epochs 200 \
  --skip-generate \
  --gpus 0,1,2,3
```

With multiple GPUs, the driver runs prompted/unprompted base training on separate
GPUs, splits the three sample jobs across GPUs, splits DAS and Traj TracIn
attribution jobs by query across GPUs, and splits each LDS `m=64` run by subset
id across the four GPUs. Parallel task logs are written under:

`diffusion_jax_refined/cifar5_multi/result/<EXPERIMENT_TAG>/logs/`

Use `--no-parallel` to force the old sequential schedule. Add End TracIn with
`--attribution-algorithms das,traj_tracin,end_tracin`.

## TACC Jobs

Submit one 48-hour H100 job, using 4 nodes and 16 independent GPU workers:

```bash
sbatch diffusion_jax_refined/cifar5_multi/tacc/h100/run_full_2day_h100.sh
```

Submit one 48-hour RTX-small job, using 1 node and 2 independent GPU workers:

```bash
sbatch diffusion_jax_refined/cifar5_multi/tacc/rtx_small/run_full_2day_rtx_small.sh
```

Both jobs run the full pipeline in one allocation: dataset generation, base
models, sampling, 25% LDS subset models with `3 x 64` subsets, DAS and
Traj TracIn attribution for the three queries, then LDS eval. Override common
settings at submit time, for example:

```bash
EXPERIMENT_TAG=cifar5_multi_exp2 JAX_BATCH_SIZE=16 sbatch diffusion_jax_refined/cifar5_multi/tacc/rtx_small/run_full_2day_rtx_small.sh
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
