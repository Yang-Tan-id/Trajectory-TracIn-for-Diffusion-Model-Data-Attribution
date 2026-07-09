# CIFAR2 Vista gh Pipeline

Submit from the repository root on Vista:

```bash
bash diffusion_jax_refined/cifar2/vista/submit_vista_pipeline.sh
```

This submits three jobs:

```bash
sbatch diffusion_jax_refined/cifar2/vista/00_train_lds_50pct_vista.sh
sbatch diffusion_jax_refined/cifar2/vista/01_sample_and_attribute_vista.sh
sbatch --dependency=afterok:<train_job_id>:<attr_job_id> diffusion_jax_refined/cifar2/vista/02_eval_and_aggregate_vista.sh
```

Progress commands:

```bash
squeue -u "$USER"
sacct -j <job_id> --format=JobID,JobName%28,State,Elapsed,Timelimit,NodeList%24
tail -f cifar2-lds-50pct-<job_id>.out
tail -f cifar2-attr-norm-vista-<job_id>.out
tail -f cifar2-eval-vista-<job_id>.out
find diffusion_jax_refined/cifar2/result/experiment1_42/vista_logs -maxdepth 2 -type f -name '*.log' | sort
```

Useful checks after completion:

```bash
find diffusion_jax_refined/cifar2/result/experiment1_42/lds_model -maxdepth 1 -name 'm_50_k_5000_seed_*' | wc -l
find diffusion_jax_refined/cifar2/result/experiment1_42/attribution_score -path '*initial_seed_24*' -name scores.npy | wc -l
find diffusion_jax_refined/cifar2/result/experiment1_42/eval -path '*initial_seed_24*' -name lds_summary.json | wc -l
find diffusion_jax_refined/cifar2/result/experiment1_42/eval -path '*aggregate_m_50_k_5000_*normalized_traj_seed_24*' -name per_seed_summary.json
```

Defaults:

- Partition: `gh`
- Account: `CCR25021`
- LDS: `M=50`, `K=5000`, seeds `1..16`, 12 hours, 16 nodes
- Sample seed: `24`
- Queries: `horse`, `automobile`, `horse,automobile`
- Traj objective: `trajectory_noise_squared_deviation_normalized`
- Traj score ranges: `1-2000`, `2001-4000`, `4001-6000`, `6001-8000`, `8001-10000`
- Loss Monte Carlo for simple-loss LDS eval: `LDS_SIMPLE_LOSS_NUM_MC=10`
