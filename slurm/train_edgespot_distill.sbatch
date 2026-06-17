#!/usr/bin/env bash
#SBATCH --job-name=edgespot-distill
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/users/voice/zoey/todak/impl/EdgeSpot}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${PROJECT_DIR}/data/manifests/splits/train.jsonl}"
VALID_MANIFEST="${VALID_MANIFEST:-${PROJECT_DIR}/data/manifests/splits/val.jsonl}"
TEACHER_EMBEDDINGS="${TEACHER_EMBEDDINGS:-${PROJECT_DIR}/data/teacher/xls-r-300m_teacher64_all.npz}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs}"
RUN_PREFIX="${RUN_PREFIX:-edgespot-ko-distill-tau}"
GPU_IDS_CSV="${GPU_IDS:-4,5,6,7}"
TAUS_CSV="${TAUS:-1,2,3,4}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LR="${LR:-1e-3}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
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
IFS=',' read -r -a TAUS_ARR <<< "${TAUS_CSV}"

if [[ "${#GPU_IDS_ARR[@]}" -ne "${#TAUS_ARR[@]}" ]]; then
  echo "GPU_IDS and TAUS must have the same length" >&2
  exit 2
fi

echo "PROJECT_DIR=${PROJECT_DIR}"
echo "TRAIN_MANIFEST=${TRAIN_MANIFEST}"
echo "VALID_MANIFEST=${VALID_MANIFEST}"
echo "TEACHER_EMBEDDINGS=${TEACHER_EMBEDDINGS}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "RUN_PREFIX=${RUN_PREFIX}"
echo "GPU_IDS=${GPU_IDS_CSV}"
echo "TAUS=${TAUS_CSV}"
echo "EPOCHS=${EPOCHS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "EXTRA_TRAIN_ARGS=${EXTRA_TRAIN_ARGS}"
echo "UV_BIN=${UV_BIN}"

pids=()
for idx in "${!TAUS_ARR[@]}"; do
  tau="${TAUS_ARR[${idx}]}"
  gpu_id="${GPU_IDS_ARR[${idx}]}"
  out_dir="${RUN_ROOT}/${RUN_PREFIX}${tau}"
  train_log="logs/train-distill-tau${tau}-${SLURM_JOB_ID:-manual}.log"

  echo "starting distill tau=${tau} gpu=${gpu_id} out=${out_dir}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    "${UV_BIN}" run python -m edgespot.train \
      --manifest "${TRAIN_MANIFEST}" \
      --valid-manifest "${VALID_MANIFEST}" \
      --out-dir "${out_dir}" \
      --tau "${tau}" \
      --objective paper_distill \
      --teacher-embeddings "${TEACHER_EMBEDDINGS}" \
      --epochs "${EPOCHS}" \
      --batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --lr "${LR}" \
      ${EXTRA_TRAIN_ARGS} \
      --device cuda
  ) > "${train_log}" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done
