#!/usr/bin/env bash
#SBATCH --job-name=aihub537-prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/users/voice/zoey/todak/impl/EdgeSpot}"
DATA_ROOT="${DATA_ROOT:-/data/datasets/voice/aihub_537/full}"
EXTRACTED_ROOT="${EXTRACTED_ROOT:-/data/datasets/voice/aihub_537/full_extracted}"
MANIFEST_DIR="${MANIFEST_DIR:-${PROJECT_DIR}/data/manifests/aihub_537_invocation}"
MIN_LABEL_COUNT="${MIN_LABEL_COUNT:-20}"
MAX_LABELS="${MAX_LABELS:-2048}"
MAX_ROWS_PER_LABEL="${MAX_ROWS_PER_LABEL:-500}"
EXTRACT_WORKERS="${EXTRACT_WORKERS:-16}"
MANIFEST_WORKERS="${MANIFEST_WORKERS:-16}"
UV_BIN="${UV_BIN:-uv}"

cd "${PROJECT_DIR}"
mkdir -p logs "${MANIFEST_DIR}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${PROJECT_DIR}/.uv-cache}"

echo "DATA_ROOT=${DATA_ROOT}"
echo "EXTRACTED_ROOT=${EXTRACTED_ROOT}"
echo "MANIFEST_DIR=${MANIFEST_DIR}"
echo "MIN_LABEL_COUNT=${MIN_LABEL_COUNT}"
echo "MAX_LABELS=${MAX_LABELS}"
echo "MAX_ROWS_PER_LABEL=${MAX_ROWS_PER_LABEL}"
echo "EXTRACT_WORKERS=${EXTRACT_WORKERS}"
echo "MANIFEST_WORKERS=${MANIFEST_WORKERS}"

"${UV_BIN}" run python scripts/extract_aihub_archives.py \
  --root "${DATA_ROOT}" \
  --out-root "${EXTRACTED_ROOT}" \
  --workers "${EXTRACT_WORKERS}"

"${UV_BIN}" run python scripts/build_aihub_537_invocation_manifest.py \
  --root "${DATA_ROOT}" \
  --extracted-root "${EXTRACTED_ROOT}" \
  --manifest-dir "${MANIFEST_DIR}" \
  --min-label-count "${MIN_LABEL_COUNT}" \
  --max-labels "${MAX_LABELS}" \
  --max-rows-per-label "${MAX_ROWS_PER_LABEL}" \
  --workers "${MANIFEST_WORKERS}"

wc -l "${MANIFEST_DIR}/all.jsonl" "${MANIFEST_DIR}/train.jsonl" "${MANIFEST_DIR}/val.jsonl"
