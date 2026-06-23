#!/usr/bin/env bash
#SBATCH --job-name=edgespot-ko-bothdevaug-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:6
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/users/voice/zoey/todak/impl/EdgeSpot}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs}"
GPU_IDS_CSV="${GPU_IDS:-2,3,4,5,6,7}"
UV_BIN="${UV_BIN:-uv}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-2}"

cd "${PROJECT_DIR}"
mkdir -p logs

export UV_CACHE_DIR="${UV_CACHE_DIR:-${PROJECT_DIR}/.uv-cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

IFS=',' read -r -a GPU_IDS_ARR <<< "${GPU_IDS_CSV}"
RUNS=(
  "edgespot-ko-proxy-noise50far-margin-confusable-bothdevaug-p20-tau4"
  "edgespot-ko-proxy-noise50far-margin-confusable-bothdevaug-p35-tau4"
  "edgespot-ko-proxy-noise50far-margin-confusable-bothdevaug-p50-tau4"
  "edgespot-ko-proxy-noise50far-margin-confusable-bothdevaug-p65-tau4"
  "edgespot-ko-proxy-noise50far-margin-confusable-bothdevaug-p75-tau4"
  "edgespot-ko-proxy-noise50far-margin-confusable-bothdevaug-p90-tau4"
)

if [[ "${#GPU_IDS_ARR[@]}" -lt "${#RUNS[@]}" ]]; then
  echo "GPU_IDS must contain at least ${#RUNS[@]} ids" >&2
  exit 2
fi

eval_one() {
  local gpu_id="$1"
  local run_name="$2"
  local checkpoint="${RUN_ROOT}/${run_name}/best.pt"

  if [[ ! -f "${checkpoint}" ]]; then
    echo "missing checkpoint: ${checkpoint}" >&2
    exit 1
  fi

  export CUDA_VISIBLE_DEVICES="${gpu_id}"
  echo "evaluating ${run_name} on gpu=${gpu_id}"

  "${UV_BIN}" run python -m edgespot.eval \
    --checkpoint "${checkpoint}" \
    --support-manifest "${PROJECT_DIR}/data/manifests/device_record/device_split_support_train_positive.jsonl" \
    --query-manifest "${PROJECT_DIR}/data/manifests/device_record/device_split_query_holdout.jsonl" \
    --out "${RUN_ROOT}/${run_name}/prototype_eval_device_split_holdout_k5.json" \
    --k-shot 5 \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --device cuda

  "${UV_BIN}" run python -m edgespot.eval \
    --checkpoint "${checkpoint}" \
    --support-manifest "${PROJECT_DIR}/data/manifests/device_record/device_todak_support_k5.jsonl" \
    --query-manifest "${PROJECT_DIR}/data/manifests/device_record/device_todak_query_all45.jsonl" \
    --out "${RUN_ROOT}/${run_name}/prototype_eval_device_all45_k5.json" \
    --k-shot 5 \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --device cuda

  "${UV_BIN}" run python -m edgespot.eval \
    --checkpoint "${checkpoint}" \
    --support-manifest "${PROJECT_DIR}/data/manifests/splits/val.jsonl" \
    --query-manifest "${PROJECT_DIR}/data/manifests/eval_large_general_conservative.jsonl" \
    --out "${RUN_ROOT}/${run_name}/prototype_eval_large_general_conservative_k10.json" \
    --k-shot 10 \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --device cuda

  "${UV_BIN}" run python -m edgespot.eval \
    --checkpoint "${checkpoint}" \
    --support-manifest "${PROJECT_DIR}/data/manifests/splits/val.jsonl" \
    --query-manifest "${PROJECT_DIR}/data/manifests/aihub_71405_noise_train50_val_far/validation_far_negatives_extracted.jsonl" \
    --out "${RUN_ROOT}/${run_name}/prototype_eval_aihub71405_validation_far_noise_only_k10.json" \
    --k-shot 10 \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --device cuda
}

pids=()
for idx in "${!RUNS[@]}"; do
  run_name="${RUNS[${idx}]}"
  gpu_id="${GPU_IDS_ARR[${idx}]}"
  log_file="logs/eval-ko-bothdevaug-${run_name}-${SLURM_JOB_ID:-manual}.log"
  (eval_one "${gpu_id}" "${run_name}") > "${log_file}" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done
