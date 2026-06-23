#!/usr/bin/env bash
#SBATCH --job-name=edgespot-ko-aihub537-inv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/users/voice/zoey/todak/impl/EdgeSpot}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${PROJECT_DIR}/data/manifests/aihub_537_invocation/train.jsonl}"
VALID_MANIFEST="${VALID_MANIFEST:-${PROJECT_DIR}/data/manifests/aihub_537_invocation/val.jsonl}"
DEVICE_LIKE_PROFILE="${DEVICE_LIKE_PROFILE:-${PROJECT_DIR}/configs/device_like_profile_top_level.json}"
GPU_IDS_CSV="${GPU_IDS:-2,3}"
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
if [[ "${#GPU_IDS_ARR[@]}" -lt 2 ]]; then
  echo "GPU_IDS must contain at least two GPU ids" >&2
  exit 2
fi

echo "TRAIN_MANIFEST=${TRAIN_MANIFEST}"
echo "VALID_MANIFEST=${VALID_MANIFEST}"
echo "GPU_IDS=${GPU_IDS_CSV}"
echo "EPOCHS=${EPOCHS}"

run_train() {
  local gpu_id="$1"
  local out_dir="$2"
  local log_file="$3"
  shift 3

  echo "starting gpu=${gpu_id} out=${out_dir}"
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
      "$@" \
      --device cuda
  ) > "${log_file}" 2>&1
}

pids=()
run_train \
  "${GPU_IDS_ARR[0]}" \
  "${RUN_ROOT}/edgespot-ko-aihub537-invocation-only-tau4" \
  "logs/train-ko-aihub537-invocation-only-tau4-${SLURM_JOB_ID:-manual}.log" &
pids+=("$!")

run_train \
  "${GPU_IDS_ARR[1]}" \
  "${RUN_ROOT}/edgespot-ko-aihub537-invocation-only-deviceaug-tau4" \
  "logs/train-ko-aihub537-invocation-only-deviceaug-tau4-${SLURM_JOB_ID:-manual}.log" \
  --device-like-profile "${DEVICE_LIKE_PROFILE}" \
  --device-like-augment-prob 0.5 \
  --device-like-coloration-scale 0.5 \
  --device-like-noise-scale 0.05 \
  --device-like-gain-scale 0.2 &
pids+=("$!")

for pid in "${pids[@]}"; do
  wait "${pid}"
done
