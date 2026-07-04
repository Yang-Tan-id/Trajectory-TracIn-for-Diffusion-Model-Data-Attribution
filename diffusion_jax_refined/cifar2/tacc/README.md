# CIFAR2 on Stampede3 H100

These scripts target Stampede3's `h100` partition and use at most two Slurm
jobs. Each requests the queue maximum of four nodes; every Stampede3 H100 node
has four H100 GPUs.

The first job trains LDS random seeds 1–16 concurrently, one seed per GPU. Each
seed contains 50 subsets of size 5000 by default:

```bash
lds_job=$(sbatch --parsable -A <allocation> \
  diffusion_jax_refined/cifar2/tacc/01_train_lds_h100.sh)
```

The second job samples seed 42 for `horse`, `automobile`, and the multi-label
query `horse,automobile`; runs five parallel Traj TracIn ranges plus DAS,
D-TRAK, and End-TracIn for every query; then evaluates all outputs against LDS
seeds 1–16:

```bash
sbatch -A <allocation> --dependency=afterok:${lds_job} \
  diffusion_jax_refined/cifar2/tacc/02_sample_attribution_eval_h100.sh
```

Both jobs default to `EXPERIMENT_TAG=experiment1_42`. Activate the Python
environment before submission, or pass an activation file:

```bash
ENV_SETUP=$HOME/envs/trajectory-tracin.sh sbatch -A <allocation> ...
```

Run `qlimits` before submission because TACC may adjust queue limits. These
scripts currently match the documented Stampede3 `h100` limits: four nodes per
job, 48 hours, four nodes running per user, and two running jobs per user.
