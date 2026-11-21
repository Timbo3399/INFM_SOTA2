# Before you submit the training job

Do these checks on the **login node** unless stated otherwise. `main.py` can download
the book itself, but prefetching keeps the compute node independent of the internet.

1. **Push your changes first.** A fresh clone only contains files that are on the
   remote branch, including `main.py` and this checklist.

2. **Environment** (once per machine)
   ```bash
   module load devel/python/3.13.3-gnu-14.2   # same module as in the job script
   export UV_NO_DEV=1
   uv sync --no-dev                           # uv >= 0.13; install with `pip install --user uv` if missing
   ```
   `~/.local/bin` must be on `PATH` in the job script if uv was installed with `--user`.
   The torch build from PyPI needs a recent NVIDIA driver (CUDA 13.0); V100 nodes do not work.

3. **W&B login** (once)
   ```bash
   uv run wandb login --verify
   ```
   The key is stored in `~/.netrc`, which the compute nodes must share. Never put a
   key in code. Runs land in `hochschule-offenburg/SOTA_2`, group `gpt2-scratch-lotr`.

4. **Tokenizer cache** (tiktoken downloads its GPT-2 data on first use)
   ```bash
   export TIKTOKEN_CACHE_DIR="$HOME/.cache/tiktoken"
   uv run python -c "import tiktoken; tiktoken.get_encoding('gpt2')"
   ```

5. **Book text** — `main.py` skips the download if the file exists
   ```bash
   mkdir -p data
   curl -L --fail -o data/lotr_djvu.txt \
     'https://dn721603.ca.archive.org/0/items/tolkien-j.-the-lord-of-the-rings-harper-collins-ebooks-2010/Tolkien-J.-The-lord-of-the-rings-HarperCollins-ebooks-2010_djvu.txt'
   ```
   `data/` is gitignored: do not commit the text.

6. **Test the pipeline** before the real run:
   ```bash
   uv run python main.py --smoke-test   # local: tiny model, no W&B, no checkpoint
   uv run python main.py --test         # cluster check: tiny model, W&B on, logs every step, no checkpoint
   ```
   `--test` finishes in well under a minute. With `#SBATCH --time=2:00` and
   `#SBATCH --mem=8gb` it is enough to confirm the GPU, data loading and W&B endpoint.

7. **Job script** for the real run (`train.slurm`, adjust partition/time to your cluster)
   ```bash
   #!/bin/bash
   #SBATCH --job-name=gpt2-scratch
   #SBATCH --gres=gpu:1
   #SBATCH --cpus-per-task=4
   #SBATCH --mem=32G
   #SBATCH --time=24:00:00
   export UV_NO_DEV=1
   export TIKTOKEN_CACHE_DIR="$HOME/.cache/tiktoken"
   # export WANDB_MODE=offline   # only if compute nodes have no internet
   cd "$SLURM_SUBMIT_DIR"
   uv run python main.py
   ```
   Submit with `sbatch train.slurm`, watch with `squeue -u $USER`, read the log with
   `tail -f slurm-<jobid>.out`.

8. **Offline W&B** is optional: `main.py` honours `WANDB_MODE=offline`. Upload after
   the job with `uv run wandb sync wandb/offline-run-*`.

9. **Disk space** for the real run: the 163M-parameter checkpoint is about 650 MB.
   The trainer writes it under `checkpoints/runs/` and copies the best one to
   `checkpoints/`, so reserve roughly 1.3 GB. Both folders are gitignored.

Overrides: `uv run python main.py --epochs 4 --lr 4e-4`. The learning rate is part of
the W&B run name, so different runs are easy to tell apart.
