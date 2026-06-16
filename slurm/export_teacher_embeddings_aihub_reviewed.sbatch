#!/usr/bin/env bash
#SBATCH --job-name=edgespot-teacher-reviewed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/users/voice/zoey/todak/impl/EdgeSpot}"
export PROJECT_DIR
export MANIFEST="${MANIFEST:-${PROJECT_DIR}/data/manifests/splits/all_with_hard_negatives_aihub_reviewed.jsonl}"
export OUT="${OUT:-${PROJECT_DIR}/data/teacher/xls-r-300m_teacher64_all_hard_aihub_reviewed.npz}"
export GPU_ID="${GPU_ID:-4}"
export BATCH_SIZE="${BATCH_SIZE:-32}"

exec bash "${PROJECT_DIR}/slurm/export_teacher_embeddings.sbatch"
