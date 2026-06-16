#!/usr/bin/env bash
#SBATCH --job-name=edgespot-eval-one
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/users/voice/zoey/todak/impl/EdgeSpot}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT}"
SUPPORT_MANIFEST="${SUPPORT_MANIFEST:-${PROJECT_DIR}/data/manifests/splits/val.jsonl}"
QUERY_MANIFEST="${QUERY_MANIFEST:?set QUERY_MANIFEST}"
OUT="${OUT:?set OUT}"
K_SHOT="${K_SHOT:-10}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-2}"
GPU_ID="${GPU_ID:-0}"
UV_BIN="${UV_BIN:-uv}"

cd "${PROJECT_DIR}"
mkdir -p logs "$(dirname "${OUT}")"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${PROJECT_DIR}/.uv-cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "PROJECT_DIR=${PROJECT_DIR}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "SUPPORT_MANIFEST=${SUPPORT_MANIFEST}"
echo "QUERY_MANIFEST=${QUERY_MANIFEST}"
echo "OUT=${OUT}"
echo "K_SHOT=${K_SHOT}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "GPU_ID=${GPU_ID}"
echo "UV_BIN=${UV_BIN}"

"${UV_BIN}" run python -m edgespot.eval \
  --checkpoint "${CHECKPOINT}" \
  --support-manifest "${SUPPORT_MANIFEST}" \
  --query-manifest "${QUERY_MANIFEST}" \
  --out "${OUT}" \
  --k-shot "${K_SHOT}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --device cuda
