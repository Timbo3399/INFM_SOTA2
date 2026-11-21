# Gemeinsamer Teil aller Jobskripte in slurm/. Jedes Jobskript wechselt erst in
# den eigenen Klon und lädt dann diese Datei:
#   cd "$SLURM_SUBMIT_DIR"
#   source slurm/common.sh

# Nur die Pakete aus `uv sync --no-dev`, ohne die Notebook-Pakete nachzuladen
export UV_NO_DEV=1
# Rechenknoten haben womöglich kein Internet, setup.sh füllt den Cache vorher
export TIKTOKEN_CACHE_DIR="$HOME/.cache/tiktoken"

# .venv direkt aktivieren statt `uv run`: Sonst synchronisieren mehrere Knoten
# gleichzeitig dieselbe Umgebung auf dem gemeinsamen Dateisystem.
if [[ ! -f .venv/bin/activate ]]; then
  echo "Keine .venv in $PWD. Einmal auf dem Login-Knoten ausführen: bash slurm/setup.sh" >&2
  exit 1
fi
# shellcheck disable=SC1091  # entsteht erst durch setup.sh
source .venv/bin/activate

# srun übernimmt --cpus-per-task seit Slurm 22.05 nicht mehr vom Job. Ohne die
# Variable liefe torchrun auf jedem Knoten mit einem einzigen Kern.
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

# Ab zwei Knoten zeigt das Log, ob NCCL InfiniBand nutzt oder auf TCP ausweicht
if ((SLURM_NNODES > 1)); then
  export NCCL_DEBUG=INFO
fi

# Treffpunkt für torchrun: der erste Knoten des Jobs. Der Port hängt an der
# Job-ID, damit sich zwei Jobs auf demselben Knoten nicht in die Quere kommen.
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export MASTER_ADDR MASTER_PORT
