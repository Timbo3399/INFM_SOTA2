#!/bin/bash
# Einrichtung auf dem Login-Knoten, einmal pro Person, im eigenen Klon:
#   bash slurm/setup.sh
#
# Ändert nichts außerhalb des Klons und des eigenen Caches. Nur uv landet bei
# Bedarf in ~/.local/bin. Das Skript lässt sich gefahrlos wiederholen.
set -euo pipefail

# In den Klon wechseln, egal von wo aus das Skript aufgerufen wird
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ORIGINAL_PATH=$PATH
export PATH="$HOME/.local/bin:$PATH"
export UV_NO_DEV=1
export TIKTOKEN_CACHE_DIR="$HOME/.cache/tiktoken"
UV_MIN_MINOR=13  # pyproject.toml verlangt uv 0.13 oder neuer

uv_is_recent() {
  command -v uv > /dev/null || return 1
  local version major minor
  read -r _ version _ < <(uv --version)
  IFS=. read -r major minor _ <<< "$version"
  ((major > 0 || minor >= UV_MIN_MINOR))
}

echo "== 1/5 uv"
if ! uv_is_recent; then
  echo "uv fehlt oder ist älter als 0.$UV_MIN_MINOR. Installiere uv mit dem offiziellen Skript nach ~/.local/bin."
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh
fi
uv --version
UV_PATH=$(command -v uv)

echo "== 2/5 Pakete ohne die Notebook-Pakete (beim ersten Mal mehrere GB für torch)"
uv sync --no-dev --locked

echo "== 3/5 tiktoken-Cache und Ordner logs/"
uv run python -c "import tiktoken; tiktoken.get_encoding('gpt2')"
mkdir -p logs

echo "== 4/5 Mini-Buch für den Smoke-Lauf"
DATA_DIR=${SOTA2_DATA_DIR:-data}
uv run python -m infm_sota2.data.mini_book "$DATA_DIR"
DATA_DIR=$(cd "$DATA_DIR" && pwd)  # absolut, für die Befehle unten

echo "== 5/5 Prüfen"
missing=()
if [[ -f "$DATA_DIR/The-lord-of-the-rings.txt" ]]; then
  echo "ok     Buchtext in $DATA_DIR"
else
  missing+=("Buchtext")
  echo "fehlt  Buchtext für die Läufe Nr. 2 bis 8. Auf dem eigenen Rechner in PowerShell,"
  echo "       im Ordner mit der Datei (\"dachs\" ist der Eintrag aus docs/dachs.md):"
  echo "         scp The-lord-of-the-rings.txt dachs:$DATA_DIR/"
fi

if [[ -f "$DATA_DIR/gpt2-small-124M.pth" ]]; then
  echo "ok     GPT-2-Gewichte von OpenAI in $DATA_DIR"
else
  missing+=("GPT-2-Gewichte")
  echo "fehlt  GPT-2-Gewichte von OpenAI für Lauf Nr. 2. Hier auf dem Login-Knoten laden:"
  echo "         uv run python -c \"from pathlib import Path; from infm_sota2.model.pretrained import GPT2_SMALL_WEIGHTS_SHA256, GPT2_SMALL_WEIGHTS_URL, download_weights; download_weights(GPT2_SMALL_WEIGHTS_URL, GPT2_SMALL_WEIGHTS_SHA256, Path('$DATA_DIR/gpt2-small-124M.pth'))\""
fi

wandb_ready=1
case "${WANDB_MODE:-online}" in
  offline | disabled)
    echo "ok     W&B: WANDB_MODE=$WANDB_MODE, kein Login nötig"
    ;;
  online)
    if [[ -n "${WANDB_API_KEY:-}" ]] || grep -qs 'api.wandb.ai' "$HOME/.netrc"; then
      echo "ok     W&B angemeldet"
    else
      wandb_ready=0
      echo "fehlt  W&B-Login. Entweder anmelden, mit dem API-Key von wandb.ai/authorize:"
      echo "         uv run wandb login --verify"
      echo "       oder ohne W&B trainieren, dafür vor jedem sbatch:"
      echo "         export WANDB_MODE=disabled"
    fi
    ;;
  *)
    wandb_ready=0
    echo "fehlt  gültiger WANDB_MODE. Gesetzt ist \"$WANDB_MODE\", erlaubt sind online, offline und disabled."
    ;;
esac

echo
if ((${#missing[@]} == 0)); then
  echo "Einrichtung fertig: bereit für alle Läufe."
else
  printf -v missing_list '%s, ' "${missing[@]}"
  echo "Einrichtung fertig: bereit für den Smoke-Lauf. Für die Läufe mit dem Buch fehlt noch: ${missing_list%, }."
fi
if ((!wandb_ready)); then
  echo "Vor dem ersten Job W&B anmelden oder WANDB_MODE setzen, siehe oben."
fi
if [[ $UV_PATH == "$HOME/.local/bin/uv" && ":$ORIGINAL_PATH:" != *":$HOME/.local/bin:"* ]]; then
  echo "Neue Terminals finden uv erst mit ~/.local/bin im PATH. Einmal ausführen:"
  echo "  echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc"
fi
echo "Nächster Schritt: Job 1 mit sbatch slurm/check.sbatch, siehe docs/dachs.md."
