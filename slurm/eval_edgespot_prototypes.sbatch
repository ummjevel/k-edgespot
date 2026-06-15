#!/usr/bin/env bash
#SBATCH --job-name=edgespot-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/users/voice/zoey/todak/impl/EdgeSpot}"
SUPPORT_MANIFEST="${SUPPORT_MANIFEST:-${PROJECT_DIR}/data/manifests/splits/val.jsonl}"
QUERY_MANIFEST="${QUERY_MANIFEST:-${PROJECT_DIR}/data/manifests/splits/test.jsonl}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs}"
RUN_PREFIX="${RUN_PREFIX:-edgespot-ko-scaf-tau}"
TAUS_CSV="${TAUS:-1,2,3,4}"
K_SHOTS_CSV="${K_SHOTS:-1,5,10}"
GPU_IDS_CSV="${GPU_IDS:-2,3}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-2}"
UV_BIN="${UV_BIN:-uv}"

cd "${PROJECT_DIR}"
mkdir -p logs "${RUN_ROOT}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${PROJECT_DIR}/.uv-cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

IFS=',' read -r -a TAUS_ARR <<< "${TAUS_CSV}"
IFS=',' read -r -a K_SHOTS_ARR <<< "${K_SHOTS_CSV}"
IFS=',' read -r -a GPU_IDS_ARR <<< "${GPU_IDS_CSV}"

echo "PROJECT_DIR=${PROJECT_DIR}"
echo "SUPPORT_MANIFEST=${SUPPORT_MANIFEST}"
echo "QUERY_MANIFEST=${QUERY_MANIFEST}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "RUN_PREFIX=${RUN_PREFIX}"
echo "TAUS=${TAUS_CSV}"
echo "K_SHOTS=${K_SHOTS_CSV}"
echo "GPU_IDS=${GPU_IDS_CSV}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "UV_BIN=${UV_BIN}"

for tau in "${TAUS_ARR[@]}"; do
  checkpoint="${RUN_ROOT}/${RUN_PREFIX}${tau}/best.pt"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "missing checkpoint: ${checkpoint}" >&2
    exit 1
  fi
done

task_idx=0
pids=()
for tau in "${TAUS_ARR[@]}"; do
  checkpoint="${RUN_ROOT}/${RUN_PREFIX}${tau}/best.pt"
  for k_shot in "${K_SHOTS_ARR[@]}"; do
    gpu_id="${GPU_IDS_ARR[$((task_idx % ${#GPU_IDS_ARR[@]}))]}"
    out="${RUN_ROOT}/${RUN_PREFIX}${tau}/prototype_eval_k${k_shot}.json"
    eval_log="logs/eval-prototype-tau${tau}-k${k_shot}-${SLURM_JOB_ID:-manual}.log"

    echo "evaluating tau=${tau} k_shot=${k_shot} gpu=${gpu_id} checkpoint=${checkpoint} out=${out}"
    (
      export CUDA_VISIBLE_DEVICES="${gpu_id}"
      "${UV_BIN}" run python -m edgespot.eval \
        --checkpoint "${checkpoint}" \
        --support-manifest "${SUPPORT_MANIFEST}" \
        --query-manifest "${QUERY_MANIFEST}" \
        --out "${out}" \
        --k-shot "${k_shot}" \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" \
        --device cuda
    ) > "${eval_log}" 2>&1 &
    pids+=("$!")
    task_idx=$((task_idx + 1))

    if [[ "${#pids[@]}" -ge "${#GPU_IDS_ARR[@]}" ]]; then
      for pid in "${pids[@]}"; do
        wait "${pid}"
      done
      pids=()
    fi
  done
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done
