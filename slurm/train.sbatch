#!/bin/bash
#SBATCH --job-name=sota2-train
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out

# Smoke-Lauf und Läufe Nr. 2, 3 und 5 bis 8. Abschicken aus dem eigenen Klon, die
# Knotenzahl beim Abschicken. Alles hinter dem Skript geht an cluster.train:
#   sbatch slurm/train.sbatch --config configs/smoke.toml
#   sbatch --nodes=2 slurm/train.sbatch --config configs/lotr-small-main.toml
set -euo pipefail

if [[ ! -f "$SLURM_SUBMIT_DIR/pyproject.toml" ]]; then
  echo "$SLURM_SUBMIT_DIR ist kein Klon des Repos. Bitte im eigenen Klon abschicken: cd <Klon> && sbatch slurm/train.sbatch --config ..." >&2
  exit 1
fi
cd "$SLURM_SUBMIT_DIR"
source slurm/common.sh

srun torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc-per-node="$SLURM_GPUS_ON_NODE" \
  --rdzv-id="$SLURM_JOB_ID" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  -m infm_sota2.cluster.train "$@"
