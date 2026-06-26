#!/usr/bin/env bash
#SBATCH --job-name=edgespot-paper-followup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --gres=gpu:6
#SBATCH --mem=192G
#SBATCH --time=36:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/users/voice/zoey/todak/impl/EdgeSpot}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${PROJECT_DIR}/data/manifests/splits/train_with_hard_negatives_aihub_reviewed_conservative_noise50far.jsonl}"
VALID_MANIFEST="${VALID_MANIFEST:-${PROJECT_DIR}/data/manifests/splits/val_aihub_reviewed_conservative_noise50far.jsonl}"
INIT_CKPT="${INIT_CKPT:-${RUN_ROOT}/edgespot-ko-aihub537-paper-distill-noise50far-finetune-tau4/best.pt}"
DEVICE_LIKE_PROFILE="${DEVICE_LIKE_PROFILE:-${PROJECT_DIR}/configs/device_like_profile_top_level.json}"
GPU_IDS_CSV="${GPU_IDS:-2,3,4,5,6,7}"
EPOCHS="${EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-3}"
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
if [[ "${#GPU_IDS_ARR[@]}" -lt 6 ]]; then
  echo "GPU_IDS must contain at least six GPU ids" >&2
  exit 2
fi

echo "PROJECT_DIR=${PROJECT_DIR}"
echo "TRAIN_MANIFEST=${TRAIN_MANIFEST}"
echo "VALID_MANIFEST=${VALID_MANIFEST}"
echo "INIT_CKPT=${INIT_CKPT}"
echo "DEVICE_LIKE_PROFILE=${DEVICE_LIKE_PROFILE}"
echo "GPU_IDS=${GPU_IDS_CSV}"
echo "EPOCHS=${EPOCHS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "NUM_WORKERS=${NUM_WORKERS}"

run_train() {
  local gpu_id="$1"
  local out_dir="$2"
  local lr="$3"
  local log_file="$4"
  shift 4

  echo "starting train gpu=${gpu_id} out=${out_dir} lr=${lr}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    "${UV_BIN}" run python -m edgespot.train \
      --manifest "${TRAIN_MANIFEST}" \
      --valid-manifest "${VALID_MANIFEST}" \
      --out-dir "${out_dir}" \
      --init-encoder-checkpoint "${INIT_CKPT}" \
      --tau 4 \
      --objective proxy \
      --loss subcenter_arcface \
      --epochs "${EPOCHS}" \
      --batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --lr "${lr}" \
      --balanced-sampler \
      --validation-metric balanced_accuracy \
      --confusable-pairs "${PROJECT_DIR}/configs/confusable_command_pairs.tsv" \
      --confusable-margin-weight 0.05 \
      --confusable-margin 0.25 \
      --negative-prototype-margin-weight 0.1 \
      --negative-prototype-margin 0.35 \
      --negative-antisaturation-weight 0.01 \
      --negative-antisaturation-margin 0.8 \
      --negative-antisaturation-top-k 16 \
      "$@" \
      --device cuda
  ) > "${log_file}" 2>&1
}

eval_one() {
  local gpu_id="$1"
  local run_name="$2"
  local checkpoint="${RUN_ROOT}/${run_name}/best.pt"

  echo "starting eval gpu=${gpu_id} run=${run_name}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    "${UV_BIN}" run python scripts/eval_multiclass_prototypes.py \
      --checkpoint "${checkpoint}" \
      --manifest "${PROJECT_DIR}/data/manifests/aihub_537_invocation/val.jsonl" \
      --out "${RUN_ROOT}/${run_name}/prototype_eval_aihub537_invocation_val_multiclass_k5.json" \
      --k-shot 5 \
      --seed 0 \
      --batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --device cuda

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
  ) > "logs/eval-ko-aihub537-paper-followup-${run_name}-${SLURM_JOB_ID:-manual}.log" 2>&1
}

RUNS=(
  "edgespot-ko-aihub537-paper-distill-noise50far-finetune-lr1e4-cont-tau4"
  "edgespot-ko-aihub537-paper-distill-noise50far-finetune-lr5e5-cont-tau4"
  "edgespot-ko-aihub537-paper-distill-noise50far-posdevaug-w35-finetune-tau4"
  "edgespot-ko-aihub537-paper-distill-noise50far-posdevaug-w50-antisat-w01-finetune-tau4"
  "edgespot-ko-aihub537-paper-distill-noise50far-bothdevaug-p20-finetune-tau4"
  "edgespot-ko-aihub537-paper-distill-noise50far-bothdevaug-p75-finetune-tau4"
)

pids=()
run_train "${GPU_IDS_ARR[0]}" "${RUN_ROOT}/${RUNS[0]}" 1e-4 \
  "logs/train-${RUNS[0]}-${SLURM_JOB_ID:-manual}.log" &
pids+=("$!")

run_train "${GPU_IDS_ARR[1]}" "${RUN_ROOT}/${RUNS[1]}" 5e-5 \
  "logs/train-${RUNS[1]}-${SLURM_JOB_ID:-manual}.log" &
pids+=("$!")

run_train "${GPU_IDS_ARR[2]}" "${RUN_ROOT}/${RUNS[2]}" 1e-4 \
  "logs/train-${RUNS[2]}-${SLURM_JOB_ID:-manual}.log" \
  --device-like-profile "${DEVICE_LIKE_PROFILE}" \
  --device-like-label-regex '^(?!__negative__$).+' \
  --device-like-augment-prob 0.35 \
  --device-like-coloration-scale 0.42 \
  --device-like-noise-scale 0.04 \
  --device-like-gain-scale 0.16 &
pids+=("$!")

run_train "${GPU_IDS_ARR[3]}" "${RUN_ROOT}/${RUNS[3]}" 1e-4 \
  "logs/train-${RUNS[3]}-${SLURM_JOB_ID:-manual}.log" \
  --device-like-profile "${DEVICE_LIKE_PROFILE}" \
  --device-like-label-regex '^(?!__negative__$).+' \
  --device-like-augment-prob 0.50 \
  --device-like-coloration-scale 0.50 \
  --device-like-noise-scale 0.05 \
  --device-like-gain-scale 0.20 \
  --negative-antisaturation-weight 0.10 \
  --negative-antisaturation-margin 0.97 \
  --negative-antisaturation-top-k 4 &
pids+=("$!")

run_train "${GPU_IDS_ARR[4]}" "${RUN_ROOT}/${RUNS[4]}" 1e-4 \
  "logs/train-${RUNS[4]}-${SLURM_JOB_ID:-manual}.log" \
  --device-like-profile "${DEVICE_LIKE_PROFILE}" \
  --device-like-augment-prob 0.20 \
  --device-like-coloration-scale 0.32 \
  --device-like-noise-scale 0.03 \
  --device-like-gain-scale 0.12 &
pids+=("$!")

run_train "${GPU_IDS_ARR[5]}" "${RUN_ROOT}/${RUNS[5]}" 1e-4 \
  "logs/train-${RUNS[5]}-${SLURM_JOB_ID:-manual}.log" \
  --device-like-profile "${DEVICE_LIKE_PROFILE}" \
  --device-like-augment-prob 0.75 \
  --device-like-coloration-scale 0.65 \
  --device-like-noise-scale 0.07 \
  --device-like-gain-scale 0.25 &
pids+=("$!")

for pid in "${pids[@]}"; do
  wait "${pid}"
done

echo "all training jobs finished; starting evaluations"
pids=()
for idx in "${!RUNS[@]}"; do
  eval_one "${GPU_IDS_ARR[${idx}]}" "${RUNS[${idx}]}" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

echo "all follow-up training and evaluation jobs finished"
