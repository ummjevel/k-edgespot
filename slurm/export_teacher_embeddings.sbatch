#!/usr/bin/env bash
#SBATCH --job-name=edgespot-teacher-export
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/users/voice/zoey/todak/impl/EdgeSpot}"
MANIFEST="${MANIFEST:-${PROJECT_DIR}/data/manifests/tts_commands_and_negatives.done.jsonl}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-${PROJECT_DIR}/runs/teacher-xls-r-300m-scaf/best_teacher.pt}"
OUT="${OUT:-${PROJECT_DIR}/data/teacher/xls-r-300m_teacher64_all.npz}"
MODEL_ID="${MODEL_ID:-facebook/wav2vec2-xls-r-300m}"
CACHE_DIR="${CACHE_DIR:-${PROJECT_DIR}/models/huggingface}"
ENCODER_LAYER="${ENCODER_LAYER:-16}"
BATCH_SIZE="${BATCH_SIZE:-32}"
UV_BIN="${UV_BIN:-uv}"
GPU_ID="${GPU_ID:-4}"

cd "${PROJECT_DIR}"
mkdir -p logs "$(dirname "${OUT}")" "${CACHE_DIR}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${PROJECT_DIR}/.uv-cache}"
export HF_HOME="${HF_HOME:-${CACHE_DIR}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${CACHE_DIR}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

echo "PROJECT_DIR=${PROJECT_DIR}"
echo "MANIFEST=${MANIFEST}"
echo "TEACHER_CHECKPOINT=${TEACHER_CHECKPOINT}"
echo "OUT=${OUT}"
echo "MODEL_ID=${MODEL_ID}"
echo "CACHE_DIR=${CACHE_DIR}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "UV_BIN=${UV_BIN}"
echo "GPU_ID=${GPU_ID}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
"${UV_BIN}" run python -m edgespot.teacher \
  --manifest "${MANIFEST}" \
  --out "${OUT}" \
  --teacher-checkpoint "${TEACHER_CHECKPOINT}" \
  --model-id "${MODEL_ID}" \
  --cache-dir "${CACHE_DIR}" \
  --encoder-layer "${ENCODER_LAYER}" \
  --batch-size "${BATCH_SIZE}" \
  --device cuda
