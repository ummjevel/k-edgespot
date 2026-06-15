#!/usr/bin/env bash
#SBATCH --job-name=edgespot-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/users/voice/zoey/todak/impl/EdgeSpot}"
DATA_ROOT="${DATA_ROOT:-/data/datasets/voice}"
CACHE_ROOT="${CACHE_ROOT:-/data/datasets/voice/.cache}"
PRESET="${PRESET:-default}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${PROJECT_DIR}"
mkdir -p logs "${DATA_ROOT}" "${CACHE_ROOT}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${PROJECT_DIR}/.uv-cache}"
export XDG_CACHE_HOME="${CACHE_ROOT}"

echo "PROJECT_DIR=${PROJECT_DIR}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "CACHE_ROOT=${CACHE_ROOT}"
echo "PRESET=${PRESET}"
echo "PYTHON_BIN=${PYTHON_BIN}"

"${PYTHON_BIN}" scripts/download_external_data.py \
  --data-root "${DATA_ROOT}" \
  --cache-root "${CACHE_ROOT}" \
  --preset "${PRESET}" \
  --no-clobber
