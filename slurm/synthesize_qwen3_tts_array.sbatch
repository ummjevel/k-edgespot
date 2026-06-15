#!/usr/bin/env bash
#SBATCH --job-name=edgespot-tts-array
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/users/voice/zoey/todak/impl/EdgeSpot}"
MANIFEST="${MANIFEST:-${PROJECT_DIR}/data/manifests/tts_commands_and_negatives.jsonl}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/data/audio/tts_ko}"
DONE_PREFIX="${DONE_PREFIX:-${PROJECT_DIR}/data/manifests/tts_commands_and_negatives}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign}"
CACHE_DIR="${CACHE_DIR:-${PROJECT_DIR}/models/huggingface}"
QWEN_SOURCE="${QWEN_SOURCE:-/data/users/voice/zoey/Qwen3-TTS}"
PYTHON_BIN="${PYTHON_BIN:-/data/users/voice/.conda/envs/qwen3-tts/bin/python}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_SHARDS="${NUM_SHARDS:-4}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID}"

cd "${PROJECT_DIR}"
mkdir -p logs "${OUT_DIR}" "$(dirname "${DONE_PREFIX}")" "${CACHE_DIR}"
mkdir -p /tmp/numba-cache /tmp/torch-cache

export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba-cache}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch-cache}"
export HF_HOME="${HF_HOME:-${CACHE_DIR}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${CACHE_DIR}/hub}"
unset TRANSFORMERS_CACHE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export PYTHONPATH="${PROJECT_DIR}/src:${QWEN_SOURCE}:${PYTHONPATH:-}"
"${PYTHON_BIN}" scripts/synthesize_qwen3_tts.py \
  --manifest "${MANIFEST}" \
  --out-dir "${OUT_DIR}" \
  --done-manifest "${DONE_PREFIX}.shard${SHARD_INDEX}.done.jsonl" \
  --model-id "${MODEL_ID}" \
  --cache-dir "${CACHE_DIR}" \
  --qwen-source "${QWEN_SOURCE}" \
  --device cuda \
  --batch-size "${BATCH_SIZE}" \
  --num-shards "${NUM_SHARDS}" \
  --shard-index "${SHARD_INDEX}"
