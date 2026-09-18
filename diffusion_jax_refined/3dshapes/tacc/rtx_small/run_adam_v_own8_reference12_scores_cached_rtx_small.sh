#!/usr/bin/env bash
#SBATCH -J 3d-adamv-p20
#SBATCH -o 3d-adamv-p20-%j.out
#SBATCH -e 3d-adamv-p20-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --cpus-per-task=8
#SBATCH -t 08:00:00

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
candidate="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
while [[ "${candidate}" != "/" && ! -f "${candidate}/diffusion_jax_refined/3dshapes/script/run_predicted_noise_jvp_l2_squared.py" ]]; do candidate="$(dirname "${candidate}")"; done
REPO_ROOT="${candidate}"
SHAPES_ROOT="${REPO_ROOT}/diffusion_jax_refined/3dshapes"
DRIVER="${SHAPES_ROOT}/script/run_predicted_noise_jvp_l2_squared.py"
source /scratch/11447/yangtan7447/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/11447/yangtan7447/conda-envs/trajectory-tracin
export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
EXPERIMENT_TAG="${EXPERIMENT_TAG:-experiment1}"; TRAIN_SEED="${TRAIN_SEED:-42}"
TRAIN_NAMESPACE=traj_tracin_adam_v
RESULT_ROOT="${SHAPES_ROOT}/result/${EXPERIMENT_TAG}"
LOG_ROOT="${RESULT_ROOT}/logs/adam_v_own8_reference12/${SLURM_JOB_ID}"
mkdir -p "${LOG_ROOT}"

run_bank() {
  local bank="$1" probes="$2" patterns="$3" seeds="$4" suffix="$5"
  for spec in 'squared square_mean' 'absolute absolute_mean' 'probe_l2 term_root' 'timestamp_probe_l2 timestamp_root'; do
    read -r contraction label <<<"${spec}"
    run_id="${SLURM_JOB_ID}_${bank}_${label}"
    pids=(); failed=0
    for shard in 0 1; do
      ( CUDA_VISIBLE_DEVICES="${shard}" JAX_NUM_DEVICES=1 JAX_PLATFORMS=cuda python "${DRIVER}" score-shard \
          --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" --train-namespace "${TRAIN_NAMESPACE}" \
          --run-id "${run_id}" --shard-index "${shard}" --shard-count 2 --num-probes "${probes}" \
          --contraction "${contraction}" --namespace-suffix "${suffix}" \
          --query-namespace-patterns "${patterns}" --expected-query-probe-mode timestamp_shared_gaussian \
          --expected-query-probe-seeds "${seeds}" ) >"${LOG_ROOT}/${bank}_${label}_${shard}.log" 2>&1 &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
    (( failed == 0 )) || { echo "${bank}/${label} failed" >&2; exit 1; }
    JAX_PLATFORMS=cpu python "${DRIVER}" merge --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
      --train-namespace "${TRAIN_NAMESPACE}" --run-id "${run_id}" --shard-count 2 --num-probes "${probes}" \
      --contraction "${contraction}" --namespace-suffix "${suffix}" --query-namespace-patterns "${patterns}" \
      --expected-query-probe-mode timestamp_shared_gaussian --expected-query-probe-seeds "${seeds}"
  done
}

OWN_SEEDS='20260917,20260918,73194261,418507293,90216487,563809241,247196803,816430927'
OWN_PATTERNS='loss_direction_predicted_noise_probe1_timestamp_shared_checkpoint_own_trajectory_r0,loss_direction_predicted_noise_probe1_timestamp_shared_seed20260918_checkpoint_own_trajectory_r0,loss_direction_predicted_noise_probe1_timestamp_shared_seed73194261_checkpoint_own_trajectory_r0,loss_direction_predicted_noise_probe1_timestamp_shared_seed418507293_checkpoint_own_trajectory_r0,loss_direction_predicted_noise_probe1_timestamp_shared_seed90216487_checkpoint_own_trajectory_r0,loss_direction_predicted_noise_probe1_timestamp_shared_seed563809241_checkpoint_own_trajectory_r0,loss_direction_predicted_noise_probe1_timestamp_shared_seed247196803_checkpoint_own_trajectory_r0,loss_direction_predicted_noise_probe1_timestamp_shared_seed816430927_checkpoint_own_trajectory_r0'
REF_SEEDS='314159265,271828183,161803399,141421357,538102947,794615203,126937481,682450719,905173624,417286953,263590817,849031576'
REF_PATTERNS=''
IFS=',' read -ra ref_seed_array <<<"${REF_SEEDS}"
for seed in "${ref_seed_array[@]}"; do [[ -z "${REF_PATTERNS}" ]] || REF_PATTERNS+=','; REF_PATTERNS+="loss_direction_predicted_noise_probe1_timestamp_shared_reference_seed${seed}_r0"; done

run_bank own8 8 "${OWN_PATTERNS}" "${OWN_SEEDS}" adam_v_own8
run_bank reference12 12 "${REF_PATTERNS}" "${REF_SEEDS}" adam_v_reference12
SCHEMES='predicted_noise_jvp_l2_squared_probe8_adam_v_own8,predicted_noise_jvp_absolute_probe8_adam_v_own8,predicted_noise_jvp_probe_l2_probe8_adam_v_own8,predicted_noise_jvp_timestamp_probe_l2_probe8_adam_v_own8,predicted_noise_jvp_l2_squared_probe12_adam_v_reference12,predicted_noise_jvp_absolute_probe12_adam_v_reference12,predicted_noise_jvp_probe_l2_probe12_adam_v_reference12,predicted_noise_jvp_timestamp_probe_l2_probe12_adam_v_reference12'
for sign in 1 -1; do
  JAX_PLATFORMS=cpu python "${SHAPES_ROOT}/script/run_traj_tracin_lds_cached.py" \
    --execute --experiment "${EXPERIMENT_TAG}" --train-seed "${TRAIN_SEED}" \
    --score-schemes "${SCHEMES}" --prediction-sign="${sign}"
done
echo "[done] Adam-v own8 and reference12 score banks"
