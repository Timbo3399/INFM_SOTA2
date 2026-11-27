#!/bin/bash
#SBATCH --job-name=sota2-search
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --array=0-11%4
#SBATCH --output=logs/%x-%A_%a.out

# Lauf Nr. 4: Suche nach Lernrate und Batchgröße als Job-Array. Jede Aufgabe ist ein
# eigener Job auf einer GPU und trainiert die Kombination Nr. SLURM_ARRAY_TASK_ID
# aus configs/lotr-small-search.toml, höchstens 4 laufen gleichzeitig.
#   sbatch slurm/search.sbatch
set -euo pipefail

if [[ ! -f "$SLURM_SUBMIT_DIR/pyproject.toml" ]]; then
  echo "$SLURM_SUBMIT_DIR ist kein Klon des Repos. Bitte im eigenen Klon abschicken: cd <Klon> && sbatch slurm/search.sbatch" >&2
  exit 1
fi
cd "$SLURM_SUBMIT_DIR"
source slurm/common.sh

# --standalone: Jede Aufgabe sucht sich einen freien Port, auch wenn zwei Aufgaben
# auf demselben Knoten landen
srun --kill-on-bad-exit=1 torchrun --standalone --nproc-per-node=1 \
  -m infm_sota2 train \
  --config configs/lotr-small-search.toml \
  --grid-index "$SLURM_ARRAY_TASK_ID"
