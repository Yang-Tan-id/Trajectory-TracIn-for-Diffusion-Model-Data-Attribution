#!/bin/bash
#SBATCH -J cifar5-das-eval-rtx
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00
#SBATCH -o cifar5-das-eval-rtx-%j.out
#SBATCH -e cifar5-das-eval-rtx-%j.err

set -euo pipefail

cd /work2/11447/yangtan7447/stampede3/Trajectory-TracIn-for-Diffusion-Model-Data-Attribution
source env_setup_trajectory_tracin.sh

python diffusion_jax_refined/cifar5_multi/script/run_cifar5_multi_attribution_distributed.py \
  --execute \
  --experiment cifar5_multi_exp1 \
  --size 10000 \
  --train-seed 42 \
  --epochs 200 \
  --only-lds-eval \
  --eval-algorithms das
