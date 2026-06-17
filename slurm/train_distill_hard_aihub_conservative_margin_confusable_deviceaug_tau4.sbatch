#!/usr/bin/env bash
#SBATCH --job-name=edgespot-deviceaug-tau4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/users/voice/zoey/todak/impl/EdgeSpot}"

export TRAIN_MANIFEST="${TRAIN_MANIFEST:-${PROJECT_DIR}/data/manifests/splits/train_with_hard_negatives_aihub_reviewed_conservative.jsonl}"
export VALID_MANIFEST="${VALID_MANIFEST:-${PROJECT_DIR}/data/manifests/splits/val_aihub_reviewed_conservative.jsonl}"
export TEACHER_EMBEDDINGS="${TEACHER_EMBEDDINGS:-${PROJECT_DIR}/data/teacher/xls-r-300m_teacher64_all_hard_aihub_reviewed.npz}"
export RUN_PREFIX="${RUN_PREFIX:-edgespot-ko-distill-hard-aihub-conservative-margin-confusable-deviceaug-tau}"
export GPU_IDS="${GPU_IDS:-4}"
export TAUS="${TAUS:-4}"
export EPOCHS="${EPOCHS:-40}"
export BATCH_SIZE="${BATCH_SIZE:-512}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export DEVICE_LIKE_PROFILE="${DEVICE_LIKE_PROFILE:-${PROJECT_DIR}/configs/device_like_profile_top_level.json}"
export EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:---negative-prototype-margin-weight 0.1 --negative-prototype-margin 0.35 --confusable-pairs ${PROJECT_DIR}/configs/confusable_command_pairs.tsv --confusable-margin-weight 0.05 --confusable-margin 0.25 --device-like-profile ${DEVICE_LIKE_PROFILE} --device-like-augment-prob 0.5 --device-like-coloration-scale 0.5 --device-like-noise-scale 0.05 --device-like-gain-scale 0.2}"

exec bash "${PROJECT_DIR}/slurm/train_edgespot_distill.sbatch"
