#!/usr/bin/env bash
#SBATCH --job-name=edgespot-tts
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/users/voice/zoey/todak/impl/EdgeSpot}"
MANIFEST="${MANIFEST:-${PROJECT_DIR}/data/manifests/tts_commands_and_negatives.jsonl}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/data/audio/tts_ko}"
DONE_PREFIX="${DONE_PREFIX:-${PROJECT_DIR}/data/manifests/tts_commands_and_negatives}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign}"
CACHE_DIR="${CACHE_DIR:-${PROJECT_DIR}/models/huggingface}"
QWEN_SOURCE="${QWEN_SOURCE:-/data/users/voice/zoey/Qwen3-TTS}"
GPU_IDS_CSV="${GPU_IDS:-4,5,6,7}"
PYTHON_BIN="${PYTHON_BIN:-/data/users/voice/.conda/envs/qwen3-tts/bin/python}"
BATCH_SIZE="${BATCH_SIZE:-16}"

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

IFS=',' read -r -a GPU_IDS_ARR <<< "${GPU_IDS_CSV}"
NUM_SHARDS="${#GPU_IDS_ARR[@]}"
if [[ "${NUM_SHARDS}" -lt 1 ]]; then
  echo "GPU_IDS must contain at least one GPU id" >&2
  exit 2
fi

echo "PROJECT_DIR=${PROJECT_DIR}"
echo "MANIFEST=${MANIFEST}"
echo "OUT_DIR=${OUT_DIR}"
echo "MODEL_ID=${MODEL_ID}"
echo "CACHE_DIR=${CACHE_DIR}"
echo "GPU_IDS=${GPU_IDS_CSV}"
echo "NUM_SHARDS=${NUM_SHARDS}"

"${PYTHON_BIN}" - <<PY
import importlib.util
import sys

sys.path.insert(0, "${QWEN_SOURCE}")

if importlib.util.find_spec("qwen_tts") is None:
    raise SystemExit(
        "qwen_tts is not importable. Set QWEN_SOURCE to the local Qwen3-TTS checkout."
    )
PY

pids=()
for shard_idx in "${!GPU_IDS_ARR[@]}"; do
  gpu_id="${GPU_IDS_ARR[${shard_idx}]}"
  done_manifest="${DONE_PREFIX}.shard${shard_idx}.done.jsonl"
  shard_log="logs/tts-shard${shard_idx}-${SLURM_JOB_ID:-manual}.log"

  echo "starting shard=${shard_idx}/${NUM_SHARDS} gpu=${gpu_id}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    export PYTHONPATH="${PROJECT_DIR}/src:${QWEN_SOURCE}:${PYTHONPATH:-}"
    "${PYTHON_BIN}" scripts/synthesize_qwen3_tts.py \
      --manifest "${MANIFEST}" \
      --out-dir "${OUT_DIR}" \
      --done-manifest "${done_manifest}" \
      --model-id "${MODEL_ID}" \
      --cache-dir "${CACHE_DIR}" \
      --qwen-source "${QWEN_SOURCE}" \
      --device cuda \
      --batch-size "${BATCH_SIZE}" \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "${shard_idx}"
  ) > "${shard_log}" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

cat "${DONE_PREFIX}".shard*.done.jsonl > "${DONE_PREFIX}.done.jsonl"
echo "wrote ${DONE_PREFIX}.done.jsonl"
