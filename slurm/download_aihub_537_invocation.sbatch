#!/usr/bin/env bash
#SBATCH --job-name=aihub537-download
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=96:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/users/voice/zoey/todak/impl/EdgeSpot}"
OUT_DIR="${OUT_DIR:-/data/datasets/voice/aihub_537/call_only}"
AIHUBSHELL="${AIHUBSHELL:-/data/datasets/voice/aihubshell}"
AIHUB_APIKEY_FILE="${AIHUB_APIKEY_FILE:-/data/datasets/voice/.secrets/aihub_api_key}"
FILE_LIST="${FILE_LIST:-${PROJECT_DIR}/configs/aihub_537_call_files.txt}"

cd "${PROJECT_DIR}"
mkdir -p logs "${OUT_DIR}"

if [[ -z "${AIHUB_APIKEY:-}" && -f "${AIHUB_APIKEY_FILE}" ]]; then
  AIHUB_APIKEY="$(tr -d '[:space:]' < "${AIHUB_APIKEY_FILE}")"
  export AIHUB_APIKEY
fi

if [[ -z "${AIHUB_APIKEY:-}" ]]; then
  echo "AIHUB_APIKEY is not set." >&2
  exit 1
fi

if [[ ! -x "${AIHUBSHELL}" ]]; then
  echo "AIHub shell is not executable: ${AIHUBSHELL}" >&2
  exit 1
fi

echo "DATASET_KEY=537"
echo "OUT_DIR=${OUT_DIR}"
echo "FILE_LIST=${FILE_LIST}"

FILE_KEYS="$(
  awk -F',' '
    /^[[:space:]]*#/ { next }
    NF >= 5 { gsub(/[[:space:]]/, "", $5); if ($5 != "") print $5 }
  ' "${FILE_LIST}" | paste -sd, -
)"

if [[ -z "${FILE_KEYS}" ]]; then
  echo "No file keys found in ${FILE_LIST}" >&2
  exit 1
fi

echo "FILE_KEYS=${FILE_KEYS}"

cd "${OUT_DIR}"
"${AIHUBSHELL}" -mode d -datasetkey 537 -filekey "${FILE_KEYS}"
