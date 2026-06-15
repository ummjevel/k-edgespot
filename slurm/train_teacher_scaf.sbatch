#!/usr/bin/env bash
#SBATCH --job-name=edgespot-teacher
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/users/voice/zoey/todak/impl/EdgeSpot}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${PROJECT_DIR}/data/manifests/splits/train.jsonl}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/runs/teacher-xls-r-300m-scaf}"
MODEL_ID="${MODEL_ID:-facebook/wav2vec2-xls-r-300m}"
CACHE_DIR="${CACHE_DIR:-${PROJECT_DIR}/models/huggingface}"
ENCODER_LAYER="${ENCODER_LAYER:-16}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LR="${LR:-1e-3}"
UV_BIN="${UV_BIN:-uv}"
GPU_ID="${GPU_ID:-4}"

cd "${PROJECT_DIR}"
mkdir -p logs "${OUT_DIR}" "${CACHE_DIR}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${PROJECT_DIR}/.uv-cache}"
export HF_HOME="${HF_HOME:-${CACHE_DIR}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${CACHE_DIR}/hub}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

echo "PROJECT_DIR=${PROJECT_DIR}"
echo "TRAIN_MANIFEST=${TRAIN_MANIFEST}"
echo "OUT_DIR=${OUT_DIR}"
echo "MODEL_ID=${MODEL_ID}"
echo "CACHE_DIR=${CACHE_DIR}"
echo "ENCODER_LAYER=${ENCODER_LAYER}"
echo "EPOCHS=${EPOCHS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "UV_BIN=${UV_BIN}"
echo "GPU_ID=${GPU_ID}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
"${UV_BIN}" run python -m edgespot.train_teacher \
  --manifest "${TRAIN_MANIFEST}" \
  --out-dir "${OUT_DIR}" \
  --model-id "${MODEL_ID}" \
  --cache-dir "${CACHE_DIR}" \
  --encoder-layer "${ENCODER_LAYER}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --device cuda
