#!/bin/bash
#SBATCH --job-name=gpt2-test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=2:00            # 2 minutes, test only
#SBATCH --mem=8gb
# Uncomment to get email notifications about your job
#SBATCH --mail-type=ALL        # ALL, START, END, FAIL
#SBATCH --mail-user=vwoehrle@stud.hs-offenburg.de

set -euo pipefail

# Must be the same interpreter used for `uv sync` on the login node
module load devel/python/3.13.3-gnu-14.2

# uv is installed once on the login node, not in every job
export PATH="$HOME/.local/bin:$PATH"
export UV_NO_DEV=1
export TIKTOKEN_CACHE_DIR="$HOME/.cache/tiktoken"
# export WANDB_MODE=offline    # only if compute nodes have no internet

cd "${SLURM_SUBMIT_DIR:-$PWD}"
uv run python main.py --test
