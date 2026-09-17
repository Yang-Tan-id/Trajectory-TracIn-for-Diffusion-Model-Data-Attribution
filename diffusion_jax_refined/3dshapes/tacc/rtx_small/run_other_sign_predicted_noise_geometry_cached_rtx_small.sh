#!/usr/bin/env bash
#SBATCH -J 3d-sign-geom
#SBATCH -o 3d-sign-geom-%j.out
#SBATCH -e 3d-sign-geom-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH -t 01:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/analyze_q8_checkpoint_sign_predicted_noise_geometry.py" ]]; do
  candidate="$(dirname "${candidate}")"
done
REPO_ROOT="${candidate}"
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"
ANALYZER="${SHAPES_ROOT}/script/analyze_q8_checkpoint_sign_predicted_noise_geometry.py"
[[ -f "${ANALYZER}" ]] || { echo "Could not locate repository" >&2; exit 1; }

source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1

EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"
TRAIN_SEED="${TRAIN_SEED:-42}"
BASE_OUT="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/eval/other_sign_predicted_noise_geometry/run_${SLURM_JOB_ID}"
CKPT_13="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/eval/original_f_checkpoint_sign_crossfit_bad_queries/run_3507876"
TIME_13="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/eval/original_f_timestamp_sign_crossfit_bad_queries/run_3507651"
TIME_OTHERS="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}/eval/original_f_timestamp_sign_crossfit_bad_queries/run_3507842"

run_case() {
  local tag="$1"
  shift
  echo "[case] ${tag}"
  python "${ANALYZER}" \
    --experiment "${EXPERIMENT_TAG}" \
    --train-seed "${TRAIN_SEED}" \
    --out-dir "${BASE_OUT}/${tag}" \
    "$@"
}

run_case q1_checkpoint_query_l2_single_change \
  --query-id 1 --variant query_l2 --method single_change_point \
  --sign-axis checkpoint --checkpoint-crossfit-dir "${CKPT_13}"

run_case q3_checkpoint_raw_ten_bins \
  --query-id 3 --variant raw --method ten_bins \
  --sign-axis checkpoint --checkpoint-crossfit-dir "${CKPT_13}"

run_case q6_timestamp_query_l2 \
  --query-id 6 --variant query_l2 --sign-axis timestamp \
  --checkpoint-crossfit-dir "${TIME_OTHERS}"

run_case q9_timestamp_raw \
  --query-id 9 --variant raw --sign-axis timestamp \
  --checkpoint-crossfit-dir "${TIME_OTHERS}"

run_case q3_timestamp_raw \
  --query-id 3 --variant raw --sign-axis timestamp \
  --checkpoint-crossfit-dir "${TIME_13}"

echo "[done] other sign/noise geometry comparisons: ${BASE_OUT}"
