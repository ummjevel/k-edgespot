#!/usr/bin/env bash
#SBATCH --job-name=edgespot-ko-bothdevaug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --gres=gpu:6
#SBATCH --mem=192G
#SBATCH --time=36:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/users/voice/zoey/todak/impl/EdgeSpot}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${PROJECT_DIR}/data/manifests/splits/train_with_hard_negatives_aihub_reviewed_conservative_noise50far.jsonl}"
VALID_MANIFEST="${VALID_MANIFEST:-${PROJECT_DIR}/data/manifests/splits/val_aihub_reviewed_conservative_noise50far.jsonl}"
DEVICE_LIKE_PROFILE="${DEVICE_LIKE_PROFILE:-${PROJECT_DIR}/configs/device_like_profile_top_level.json}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs}"
GPU_IDS_CSV="${GPU_IDS:-2,3,4,5,6,7}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-3}"
UV_BIN="${UV_BIN:-uv}"

cd "${PROJECT_DIR}"
mkdir -p logs "${RUN_ROOT}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${PROJECT_DIR}/.uv-cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

IFS=',' read -r -a GPU_IDS_ARR <<< "${GPU_IDS_CSV}"
if [[ "${#GPU_IDS_ARR[@]}" -lt 6 ]]; then
  echo "GPU_IDS must contain at least six GPU ids" >&2
  exit 2
fi

echo "PROJECT_DIR=${PROJECT_DIR}"
echo "TRAIN_MANIFEST=${TRAIN_MANIFEST}"
echo "VALID_MANIFEST=${VALID_MANIFEST}"
echo "DEVICE_LIKE_PROFILE=${DEVICE_LIKE_PROFILE}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "GPU_IDS=${GPU_IDS_CSV}"
echo "EPOCHS=${EPOCHS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "NUM_WORKERS=${NUM_WORKERS}"

run_train() {
  local gpu_id="$1"
  local out_dir="$2"
  local log_file="$3"
  local prob="$4"
  local coloration="$5"
  local noise="$6"
  local gain="$7"

  echo "starting gpu=${gpu_id} out=${out_dir} prob=${prob} coloration=${coloration} noise=${noise} gain=${gain}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    "${UV_BIN}" run python -m edgespot.train \
      --manifest "${TRAIN_MANIFEST}" \
      --valid-manifest "${VALID_MANIFEST}" \
      --out-dir "${out_dir}" \
      --tau 4 \
      --objective proxy \
      --loss subcenter_arcface \
      --epochs "${EPOCHS}" \
      --batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --lr "${LR}" \
      --balanced-sampler \
      --validation-metric balanced_accuracy \
      --confusable-pairs "${PROJECT_DIR}/configs/confusable_command_pairs.tsv" \
      --confusable-margin-weight 0.05 \
      --confusable-margin 0.25 \
      --negative-prototype-margin-weight 0.1 \
      --negative-prototype-margin 0.35 \
      --device-like-profile "${DEVICE_LIKE_PROFILE}" \
      --device-like-augment-prob "${prob}" \
      --device-like-coloration-scale "${coloration}" \
      --device-like-noise-scale "${noise}" \
      --device-like-gain-scale "${gain}" \
      --device cuda
  ) > "${log_file}" 2>&1
}

pids=()
run_train \
  "${GPU_IDS_ARR[0]}" \
  "${RUN_ROOT}/edgespot-ko-proxy-noise50far-margin-confusable-bothdevaug-p20-tau4" \
  "logs/train-ko-proxy-noise50far-margin-confusable-bothdevaug-p20-tau4-${SLURM_JOB_ID:-manual}.log" \
  0.20 0.32 0.03 0.12 &
pids+=("$!")

run_train \
  "${GPU_IDS_ARR[1]}" \
  "${RUN_ROOT}/edgespot-ko-proxy-noise50far-margin-confusable-bothdevaug-p35-tau4" \
  "logs/train-ko-proxy-noise50far-margin-confusable-bothdevaug-p35-tau4-${SLURM_JOB_ID:-manual}.log" \
  0.35 0.42 0.04 0.16 &
pids+=("$!")

run_train \
  "${GPU_IDS_ARR[2]}" \
  "${RUN_ROOT}/edgespot-ko-proxy-noise50far-margin-confusable-bothdevaug-p50-tau4" \
  "logs/train-ko-proxy-noise50far-margin-confusable-bothdevaug-p50-tau4-${SLURM_JOB_ID:-manual}.log" \
  0.50 0.50 0.05 0.20 &
pids+=("$!")

run_train \
  "${GPU_IDS_ARR[3]}" \
  "${RUN_ROOT}/edgespot-ko-proxy-noise50far-margin-confusable-bothdevaug-p65-tau4" \
  "logs/train-ko-proxy-noise50far-margin-confusable-bothdevaug-p65-tau4-${SLURM_JOB_ID:-manual}.log" \
  0.65 0.58 0.06 0.23 &
pids+=("$!")

run_train \
  "${GPU_IDS_ARR[4]}" \
  "${RUN_ROOT}/edgespot-ko-proxy-noise50far-margin-confusable-bothdevaug-p75-tau4" \
  "logs/train-ko-proxy-noise50far-margin-confusable-bothdevaug-p75-tau4-${SLURM_JOB_ID:-manual}.log" \
  0.75 0.65 0.07 0.25 &
pids+=("$!")

run_train \
  "${GPU_IDS_ARR[5]}" \
  "${RUN_ROOT}/edgespot-ko-proxy-noise50far-margin-confusable-bothdevaug-p90-tau4" \
  "logs/train-ko-proxy-noise50far-margin-confusable-bothdevaug-p90-tau4-${SLURM_JOB_ID:-manual}.log" \
  0.90 0.75 0.08 0.30 &
pids+=("$!")

for pid in "${pids[@]}"; do
  wait "${pid}"
done
