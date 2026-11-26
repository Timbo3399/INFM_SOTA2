#!/bin/bash
#SBATCH --job-name=sota2-check
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

# Job 1: prüft die Umgebung auf allen Knoten und misst auf einem Knoten den
# Benchmark (Lauf Nr. 1). Abschicken aus dem eigenen Klon:
#   sbatch slurm/check.sbatch             ein Knoten, mit Benchmark
#   sbatch --nodes=2 slurm/check.sbatch   zwei Knoten, prüft NCCL zwischen ihnen
set -euo pipefail

if [[ ! -f "$SLURM_SUBMIT_DIR/pyproject.toml" ]]; then
  echo "$SLURM_SUBMIT_DIR ist kein Klon des Repos. Bitte im eigenen Klon abschicken: cd <Klon> && sbatch slurm/check.sbatch" >&2
  exit 1
fi
cd "$SLURM_SUBMIT_DIR"
source slurm/common.sh

nvidia-smi

srun --kill-on-bad-exit=1 torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc-per-node="$SLURM_GPUS_ON_NODE" \
  --rdzv-id="$SLURM_JOB_ID" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  -m infm_sota2 check

# Der Benchmark misst eine einzelne GPU, bei mehreren Knoten fällt er weg
if ((SLURM_NNODES == 1)); then
  srun --kill-on-bad-exit=1 python -m infm_sota2 benchmark
fi
