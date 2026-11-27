# INFM_SOTA2

Gruppenprojekt im Modul SOTA 2, Master Informatik an der Hochschule Offenburg. Wir bauen ein Large Language Model von Grund auf in PyTorch und trainieren es auf dem Cluster.

Wie der Code aufgebaut ist und wie man ihn aufruft, steht kurz in [docs/code.md](docs/code.md). Alles zum Cluster steht in [docs/dachs.md](docs/dachs.md).

## Aufbau

| Pfad | Inhalt |
|---|---|
| `src/infm_sota2/` | der Code, ein Modul pro Schritt, siehe [docs/code.md](docs/code.md) |
| `configs/` | eine TOML-Datei pro Lauf, jede Einstellung steht dort |
| `slurm/` | Einrichtung (`setup.sh`) und Jobskripte für den Cluster |
| `notebooks/` | `workbook_timo_cluster.ipynb` wertet die Läufe von Woche 6 aus |
| `results/` | Ergebnisse der Läufe als JSON, `results/week06/<benutzer>/<gruppe>/` |
| `tests/` | Unittests für `src/` und die Skripte in `slurm/` |
| `scripts/` | `smoke_notebook.py` führt das Notebook im Smoke-Test aus |
| `docs/` | Anleitungen und der Entwurf von Woche 6 |
| `old/` | was nicht mehr gebraucht wird, siehe [old/README.md](old/README.md) |
| `.github/`, `.githooks/` | CI und Git-Hook, beide mit denselben Checks |

## Einrichten

Einmal pro Rechner:

1. [uv installieren](https://docs.astral.sh/uv/getting-started/installation/), mindestens Version 0.13. Ein vorhandenes uv aktualisiert `uv self update`.
2. Repo klonen und die Umgebung anlegen. uv lädt dabei Python 3.13 und legt `.venv` an:
   ```bash
   git clone https://github.com/Timbo3399/INFM_SOTA2.git
   cd INFM_SOTA2
   uv sync
   ```
3. Den Git-Hook aktivieren. Er führt vor jedem Commit dieselben Checks aus wie die CI:
   ```bash
   git config core.hooksPath .githooks
   ```
4. In VS Code `"files.eol": "\n"` setzen, weil die `.gitattributes` auch unter Windows LF verlangt. Hat ein älterer Klon noch Dateien mit CRLF, stellen `git rm -r --cached -q .` und danach `git reset --hard` sie um, aber nur bei einem Stand ohne offene Änderungen.
5. Für das Notebook in VS Code `.venv` als Kernel wählen. Ohne VS Code geht `uv run --with jupyterlab jupyter lab`.
6. Wer nach W&B loggen will, meldet sich einmal an, siehe [W&B](#wb).

Unter Linux lädt `uv sync` torch mit CUDA 13.0, das sind mehrere GB. Windows und Macs mit Apple Silicon bekommen die CPU-Version. Für Intel-Macs gibt es die festgelegte torch-Version nicht.

## Benutzen

Alle Befehle laufen über einen Einstieg, `uv run python -m infm_sota2 -h` zeigt sie. Ein Smoke-Lauf auf dem eigenen Rechner dauert Sekunden:

```bash
uv run python -m infm_sota2 mini-book data
uv run python -m infm_sota2 train --config configs/smoke-cpu.toml
```

Das Ergebnis landet unter `results/week06/<benutzer>/week6-smoke/` und endet auf `-local.json`. Solche Dateien gehören nicht ins Repo.

Die echten Läufe starten auf dem Cluster mit `sbatch`, die Anleitung steht in [docs/dachs.md](docs/dachs.md). Ausgewertet werden sie in `notebooks/workbook_timo_cluster.ipynb`.

## Checks

```bash
uv run ruff check
uv run ruff format --check
uv run pyright
uv run python -m unittest discover -s tests -p 'test*.py'
```

Dieselben Checks laufen in der CI bei jedem Pull Request und bei jedem Push auf `main` und `week/**`, der Git-Hook führt sie vor jedem Commit aus. Ruff-Fehler behebt meist `uv run ruff check --fix` und danach `uv run ruff format`. `git commit --no-verify` überspringt den Hook im Notfall, die CI prüft trotzdem.

## Versionen

torch, tiktoken und wandb sind auf die Versionen der Colab-Laufzeit festgelegt, Python auf 3.13. Die Pins stehen nur in `pyproject.toml`. So läuft lokal, auf dem Cluster und in Colab dasselbe. Welche Versionen Colab gerade hat, steht in [googlecolab/backend-info](https://github.com/googlecolab/backend-info). Angehoben wird nur zusammen mit Colab.

- Pakete nur für das Notebook oder die Checks: `uv add --dev <paket>`.
- Pakete, die Ergebnisse beeinflussen, mit der exakten Colab-Version, z. B. `uv add "numpy==2.1.3"`.
- `uv.lock` immer mit committen.

Die torch-Pakete von PyPI brauchen CUDA 13.0, also einen aktuellen NVIDIA-Treiber, und laufen nicht auf V100. Kann ein Cluster das nicht, kommt dieser Block in die `pyproject.toml`, danach `uv lock`. Er gilt dann für alle Linux-Rechner:

```toml
[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu126", marker = "sys_platform == 'linux'" }]
```

## W&B

Die Läufe loggen nach [Weights & Biases](https://wandb.ai) in das Projekt [`SOTA_2`](https://wandb.ai/hochschule-offenburg/SOTA_2) des Teams `hochschule-offenburg`. Der Link leitet auf forge.coreweave.com weiter. Wer das Projekt nicht öffnen kann, braucht eine Einladung ins Team. W&B ist kein Muss.

Angemeldet wird einmal pro Rechner, auf dem Cluster auf dem Login-Knoten:

```bash
uv run wandb login --verify
```

wandb fragt nach dem API-Key von [wandb.ai/authorize](https://wandb.ai/authorize) und speichert ihn in `~/.netrc`. Ohne W&B setzt man vor dem Lauf `export WANDB_MODE=disabled`. Kommen die Rechenknoten nicht ins Internet, heißt es `WANDB_MODE=offline`, und nach dem Job lädt `uv run wandb sync wandb/offline-run-*` die Läufe hoch.

## Arbeitsweise

| Branch | Zweck |
|---|---|
| `main` | abgestimmter Stand, geschützt, Änderungen nur per Pull Request mit Review |
| `week/NN` | Wochen-Branch, zu Wochenbeginn von `main` angelegt |
| `feature/…`, `fix/…`, `chore/…` | eine Aufgabe, zweigt von `week/NN` ab und geht per Pull Request zurück |

Gemergt wird mit „Create a merge commit“, nicht mit Squash. Am Ende der Woche geht `week/NN` per Pull Request nach `main`, und eine andere Person reviewt.

Das Notebook hat oben einen „Open in Colab“-Link auf `main`. In Colab öffnet man es über *Datei → Notebook öffnen → GitHub* auf dem eigenen Feature-Branch und speichert mit *Datei → Kopie in GitHub speichern* dorthin zurück. Colab setzt den Link dabei auf den Feature-Branch, er muss vor dem Pull Request wieder auf `blob/main/notebooks/…` zeigen. Kleine Änderungen macht man besser lokal, weil Colab beim Speichern die ganze Datei neu formatiert.

## Regeln

- **Ein Notebook, eine Person.** Gleichzeitige Änderungen an derselben Notebook-Datei lassen sich kaum mergen.
- **Keine Secrets im Code.** API-Keys nur über `wandb login` oder Colab-Secrets. Ein versehentlich gepushter Key gilt als kompromittiert und muss sofort widerrufen werden.
- **Keine Modellgewichte, Buchtexte oder Datensätze committen.** Die `.gitignore` fängt `data/`, `checkpoints/`, `*.pth` und Ähnliches ab.
- **Outputs klein halten.** Lange Ausgaben vor dem Speichern löschen.
