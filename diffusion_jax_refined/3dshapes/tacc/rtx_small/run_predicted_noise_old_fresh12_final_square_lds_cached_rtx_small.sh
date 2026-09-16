#!/usr/bin/env bash
#SBATCH -J 3d-pn12-2fsq
#SBATCH -o 3d-pn12-2fsq-%j.out
#SBATCH -e 3d-pn12-2fsq-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -t 04:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/run_traj_tracin_lds_cached.py" ]]; do
  candidate="$(dirname "${candidate}")"
done
REPO_ROOT="${candidate}"
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"
LDS_DRIVER="${SHAPES_ROOT}/script/run_traj_tracin_lds_cached.py"
PRINTER="${SHAPES_ROOT}/script/print_predicted_noise_probe4_final_post_square_lds.py"
[[ -f "${LDS_DRIVER}" ]] || { echo "Could not locate repository" >&2; exit 1; }

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHON_BIN="${PYTHON_BIN:-python}"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export PYTHONUNBUFFERED=1

SCHEMES="predicted_noise_jvp_final_square_then_mean_probe12,predicted_noise_jvp_final_mean_then_square_probe12,predicted_noise_jvp_final_square_then_mean_probe12_fresh_seed20260915,predicted_noise_jvp_final_mean_then_square_probe12_fresh_seed20260915"

echo "[phase 1/2] cached LDS: complete trajectory per probe, then square"
for sign in 1 -1; do
  JAX_PLATFORMS=cpu "${PYTHON_BIN}" "${LDS_DRIVER}" \
    --execute --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
    --score-schemes "${SCHEMES}" --prediction-sign="${sign}"
done

echo "[phase 2/2] print old12 and fresh12; four normalization variants"
for suffix in "" fresh_seed20260915; do
  label="old12"
  suffix_args=()
  if [[ -n "${suffix}" ]]; then
    label="fresh12"
    suffix_args+=(--namespace-suffix "${suffix}")
  fi
  for sign in p1 m1; do
    echo "[bank=${label} sign=${sign}]"
    "${PYTHON_BIN}" "${PRINTER}" \
      --experiment "${EXPERIMENT_TAG}" --num-probes 12 \
      --method square --prediction-sign "${sign}" \
      "${suffix_args[@]}"
  done
done

echo "[done] old12 and fresh12 final trajectory-square LDS complete"
