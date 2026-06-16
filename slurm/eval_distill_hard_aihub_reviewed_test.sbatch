#!/usr/bin/env bash
#SBATCH --job-name=edgespot-reviewed-eval-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/users/voice/zoey/todak/impl/EdgeSpot}"
RUN_DIR="${RUN_DIR:-${PROJECT_DIR}/runs/edgespot-ko-distill-hard-aihub-reviewed-tau4}"
export PROJECT_DIR
export CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/best.pt}"
export SUPPORT_MANIFEST="${SUPPORT_MANIFEST:-${PROJECT_DIR}/data/manifests/splits/val_aihub_reviewed.jsonl}"
export QUERY_MANIFEST="${QUERY_MANIFEST:-${PROJECT_DIR}/data/manifests/splits/test_aihub_reviewed.jsonl}"
export OUT="${OUT:-${RUN_DIR}/prototype_eval_augmented_reviewed_test_k10.json}"
export K_SHOT="${K_SHOT:-10}"
export GPU_ID="${GPU_ID:-2}"
export BATCH_SIZE="${BATCH_SIZE:-512}"
export NUM_WORKERS="${NUM_WORKERS:-2}"

exec bash "${PROJECT_DIR}/slurm/eval_single_prototype.sbatch"
