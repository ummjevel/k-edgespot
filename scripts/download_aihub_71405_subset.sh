#!/usr/bin/env bash
set -euo pipefail

DATASET_KEY="${DATASET_KEY:-71405}"
AIHUBSHELL="${AIHUBSHELL:-/data/datasets/voice/aihubshell}"
OUT_DIR="${OUT_DIR:-/data/datasets/voice/aihub_71405/validation_seed}"
FILE_LIST="${FILE_LIST:-configs/aihub_71405_validation_seed_files.txt}"
AIHUB_APIKEY_FILE="${AIHUB_APIKEY_FILE:-/data/datasets/voice/.secrets/aihub_api_key}"

if [[ -z "${AIHUB_APIKEY:-}" && -f "${AIHUB_APIKEY_FILE}" ]]; then
  AIHUB_APIKEY="$(tr -d '[:space:]' < "${AIHUB_APIKEY_FILE}")"
  export AIHUB_APIKEY
fi

if [[ -z "${AIHUB_APIKEY:-}" ]]; then
  echo "AIHUB_APIKEY is not set." >&2
  echo "Set AIHUB_APIKEY or put the key in ${AIHUB_APIKEY_FILE}." >&2
  exit 1
fi

if [[ ! -x "${AIHUBSHELL}" ]]; then
  echo "AIHub shell is not executable: ${AIHUBSHELL}" >&2
  exit 1
fi

file_keys="$(
  awk -F',' '
    /^[[:space:]]*#/ { next }
    NF >= 6 { gsub(/[[:space:]]/, "", $6); if ($6 != "") print $6 }
  ' "${FILE_LIST}" | paste -sd, -
)"

if [[ -z "${file_keys}" ]]; then
  echo "No file keys found in ${FILE_LIST}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
cd "${OUT_DIR}"

echo "DATASET_KEY=${DATASET_KEY}"
echo "OUT_DIR=${OUT_DIR}"
echo "FILE_LIST=${FILE_LIST}"
echo "FILE_KEYS=${file_keys}"

"${AIHUBSHELL}" -mode d -datasetkey "${DATASET_KEY}" -filekey "${file_keys}"
