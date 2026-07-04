# CIFAR2 on Stampede3 H100

These scripts target Stampede3's `h100` partition. Each requests the queue
maximum of four nodes; every Stampede3 H100 node has four H100 GPUs. Keep at
most two jobs submitted at once using the staged submission below.

The first job trains LDS random seeds 1–16 concurrently, one seed per GPU. Each
seed contains 50 subsets of size 5000 by default:

```bash
lds_job=$(sbatch --parsable -A <allocation> \
  diffusion_jax_refined/cifar2/tacc/01_train_lds_h100.sh)
```

Attribution is split across two fresh 48-hour jobs because each single-GPU task
can take 20–25 hours. Part 1 samples all queries and runs the first 16 tasks;
part 2 runs the remaining 8 and then evaluates all outputs against LDS seeds
1–16 in the same allocation:

```bash
attr1_job=$(ATTR_SHARD=1 sbatch --parsable -A <allocation> \
  --dependency=afterok:${lds_job} \
  diffusion_jax_refined/cifar2/tacc/02_sample_attribution_h100.sh)
```

After the LDS job leaves the queue, submit part 2 + evaluation:

```bash
attr2_job=$(ATTR_SHARD=2 sbatch --parsable -A <allocation> \
  --dependency=afterok:${attr1_job} \
  diffusion_jax_refined/cifar2/tacc/02_sample_attribution_h100.sh)
```

`03_lds_eval_h100.sh` remains a standalone/recovery entrypoint, but part 2
invokes it automatically after all eight attribution tasks succeed.

All jobs default to `EXPERIMENT_TAG=experiment1_42`. Activate the Python
environment before submission, or pass an activation file:

```bash
ENV_SETUP=$HOME/envs/trajectory-tracin.sh sbatch -A <allocation> ...
```

The scripts refuse to overwrite an existing LDS, sample, attribution, or eval
output by default. Prefer a new experiment tag or archive old outputs. Set
`ALLOW_OVERWRITE=1` only when replacing an existing run is intentional.

Run `qlimits` before submission because TACC may adjust queue limits. These
scripts currently match the documented Stampede3 `h100` limits: four nodes per
job, 48 hours, four nodes running per user, and two running jobs per user.
