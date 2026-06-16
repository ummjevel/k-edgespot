#!/usr/bin/env bash
#SBATCH --job-name=edgespot-conservative-tau4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/users/voice/zoey/todak/impl/EdgeSpot}"
export PROJECT_DIR
export TRAIN_MANIFEST="${TRAIN_MANIFEST:-${PROJECT_DIR}/data/manifests/splits/train_with_hard_negatives_aihub_reviewed_conservative.jsonl}"
export VALID_MANIFEST="${VALID_MANIFEST:-${PROJECT_DIR}/data/manifests/splits/val_aihub_reviewed_conservative.jsonl}"
export TEACHER_EMBEDDINGS="${TEACHER_EMBEDDINGS:-${PROJECT_DIR}/data/teacher/xls-r-300m_teacher64_all_hard_aihub_reviewed.npz}"
export RUN_PREFIX="${RUN_PREFIX:-edgespot-ko-distill-hard-aihub-conservative-tau}"
export GPU_IDS="${GPU_IDS:-4}"
export TAUS="${TAUS:-4}"
export EPOCHS="${EPOCHS:-40}"
export BATCH_SIZE="${BATCH_SIZE:-512}"
export NUM_WORKERS="${NUM_WORKERS:-2}"

exec bash "${PROJECT_DIR}/slurm/train_edgespot_distill.sbatch"
