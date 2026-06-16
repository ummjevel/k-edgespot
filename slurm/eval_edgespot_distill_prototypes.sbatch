#!/usr/bin/env bash
#SBATCH --job-name=edgespot-distill-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

export RUN_PREFIX="${RUN_PREFIX:-edgespot-ko-distill-tau}"
export GPU_IDS="${GPU_IDS:-2,3}"

exec "$(dirname "$0")/eval_edgespot_prototypes.sbatch"
