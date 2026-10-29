# INFM_SOTA2

Gruppenprojekt im Modul SOTA 2, Master Informatik an der Hochschule Offenburg.

Wir bauen ein Large Language Model von Grund auf – Schritt für Schritt in PyTorch.

## Projektstruktur

| Pfad | Inhalt |
|---|---|
| `notebooks/` | Notebooks zum Anschauen, lokal oder in Colab. Eins pro Person: `workbook_<name>.ipynb`. |
| `src/infm_sota2/` | Gemeinsamer Python-Code, später auch die Trainingsskripte für den Cluster. |
| `src/infm_sota2/data/` | `BPEDataset`, `Embeddings` und `create_dataloader`: Text → Token-Fenster → Embeddings. |
| `src/infm_sota2/transformer/` | `SelfAttention`, `CausalAttention` und `MultiHeadAttention`. |
| `tests/` | Unittests für `src/infm_sota2/`. |
| `.github/workflows/ci.yml` | CI, führt die Checks aus [Checks lokal ausführen](#checks-lokal-ausführen) aus. |
| `.githooks/pre-commit` | Git-Hook, der dieselben Checks vor jedem Commit lokal ausführt. |
| `pyproject.toml`, `uv.lock` | Abhängigkeiten und die exakt festgelegten Versionen. |

Entwickelt wird in `.py`-Dateien unter `src/`. Die Notebooks zeigen und erklären, was der Code macht.

## Quick Start

Einmal pro Rechner:

1. [uv installieren](https://docs.astral.sh/uv/getting-started/installation/), mindestens Version 0.13. Ein vorhandenes uv mit `uv self update` aktualisieren.
2. Repo klonen und die Umgebung anlegen. uv lädt dabei Python 3.13 und legt `.venv` an:
   ```bash
   git clone https://github.com/Timbo3399/INFM_SOTA2.git
   cd INFM_SOTA2
   uv sync
   ```
3. Den Git-Hook aktivieren, damit vor jedem Commit dieselben Checks laufen wie in der CI (siehe [Git-Hook vor dem Commit](#git-hook-vor-dem-commit)):
   ```bash
   git config core.hooksPath .githooks
   ```
4. In VS Code `"files.eol": "\n"` setzen. Die `.gitattributes` legt LF als Zeilenende fest, auch unter Windows. Ohne die Einstellung warnt git bei neuen Dateien „CRLF will be replaced by LF“.
5. Notebooks in VS Code öffnen und als Kernel `.venv` wählen. Ohne VS Code geht `uv run --with jupyterlab jupyter lab`.

Wer das Repo schon geklont hat, braucht nur `git pull`, `uv sync` und die Schritte 3 und 4.
In einem älteren Klon können die Dateien noch CRLF haben. Bei einem Stand ohne offene Änderungen stellen `git rm -r --cached -q .` und danach `git reset --hard` sie auf LF um.

Danach geht es weiter wie unter [Arbeitsweise](#arbeitsweise): Branch von `week/NN` anlegen, ändern, committen, Pull Request.

Unter Linux lädt `uv sync` torch mit CUDA 13.0, das sind mehrere GB. Windows und Macs mit Apple Silicon bekommen die CPU-Version.
Für Intel-Macs gibt es kein torch 2.11, dort bitte Colab nutzen.

### Checks lokal ausführen

Nach `uv sync` können Ruff, Pyright, Coverage und die Unittests so ausgeführt werden:

```bash
uv run ruff check
uv run ruff format --check
uv run pyright
uv run python -m unittest discover -s tests -p 'test*.py'
uv run coverage run -m unittest discover -s tests -p 'test*.py'
uv run coverage report
```

Dieselben Checks laufen in der CI bei jedem Pull Request und bei jedem Push auf `main` und `week/**`.
Vor dem Pull Request lokal ausführen. Gemergt wird erst, wenn die CI grün ist.

### Git-Hook vor dem Commit

`.githooks/pre-commit` führt vor jedem Commit `ruff check`, `ruff format --check`, `pyright` und die Unittests aus, mit den Versionen aus `uv.lock` wie in der CI.
Schlägt ein Check fehl, bricht der Commit ab. Ruff-Fehler behebt meist `uv run ruff check --fix` und danach `uv run ruff format`.

- Der Hook prüft die Arbeitskopie, nicht nur die gestagten Änderungen, und braucht rund 15 Sekunden.
- `git commit --no-verify` überspringt ihn im Notfall. Die CI prüft trotzdem.
- Commits aus Colab („Kopie in GitHub speichern“) und über die GitHub-Weboberfläche laufen nicht durch den Hook.

### Versionen

torch und tiktoken sind exakt auf die Versionen der Colab-Laufzeit festgelegt (`torch==2.11.0`, `tiktoken==0.14.0`, Python 3.13).
So läuft lokal, auf dem Cluster und in Colab dasselbe. `requires-python` erlaubt zusätzlich 3.12, das hat die ältere Colab-Laufzeit 2026.07.
Welche Versionen Colab gerade hat, steht in [googlecolab/backend-info](https://github.com/googlecolab/backend-info). Angehoben wird nur zusammen mit Colab.

Pakete hinzufügen:

- Nur für Notebooks (Plots, Tabellen, Widgets): `uv add --dev <paket>`.
- Pakete, die Ergebnisse beeinflussen, mit der exakten Colab-Version, z. B. numpy, sobald es im Code direkt verwendet wird: `uv add "numpy==2.1.3"`.
- `uv.lock` immer mit committen.

### Auf dem Cluster

```bash
export UV_NO_DEV=1
uv sync --no-dev
```

`uv sync --no-dev` installiert nur, was die Skripte brauchen (torch, tiktoken), ohne die Notebook-Pakete.
`UV_NO_DEV=1` sorgt dafür, dass auch `uv run` sie nicht wieder nachinstalliert. Die Variable gehört deshalb auch ins Jobskript,
sonst will der Job auf einem Rechenknoten ohne Internet Pakete herunterladen und bricht ab.

Rechenknoten haben oft kein Internet, tiktoken lädt die GPT-2-Daten aber beim ersten Aufruf herunter.
Deshalb einmal auf dem Login-Knoten ausführen (im selben Terminal wie oben) und `TIKTOKEN_CACHE_DIR` im Jobskript genauso setzen:

```bash
export TIKTOKEN_CACHE_DIR="$HOME/.cache/tiktoken"
uv run python -c "import tiktoken; tiktoken.get_encoding('gpt2')"
```

Die torch-Pakete von PyPI brauchen CUDA 13.0, also einen aktuellen NVIDIA-Treiber, und laufen nicht auf V100.
Kann der Cluster das nicht, kommt dieser Block in die `pyproject.toml`, danach `uv lock`. Er gilt dann für alle Linux-Rechner:

```toml
[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu126", marker = "sys_platform == 'linux'" }]
```

## Arbeitsweise

### Branches

| Branch | Zweck |
|---|---|
| `main` | Abgestimmter Stand. Geschützt: Änderungen nur per Pull Request mit Review. |
| `week/NN` | Wochen-Branch, wird zu Wochenbeginn von `main` angelegt, z. B. `week/03`. |
| `feature/…`, `fix/…`, `chore/…` | Eine Aufgabe. Zweigt von `week/NN` ab, geht per Pull Request zurück und wird nach dem Merge gelöscht. |

Ablauf:

1. Zu Wochenbeginn `week/NN` von `main` anlegen.
2. Für jede Aufgabe einen Branch von `week/NN` anlegen, z. B. `feature/attention`.
3. Pull Request mit `base: week/NN` ← `compare: feature/attention`.
4. Am Ende der Woche Pull Request `week/NN` → `main`. Eine andere Person reviewt und approved.
5. Neuer `week/NN+1` von `main`, dann von vorne.

Gemergt wird immer mit **„Create a merge commit“**, nicht Squash.
Wer im Feature-Branch die Änderungen der anderen braucht, holt sie mit `git pull origin week/NN`.

### Mit Colab arbeiten

Jedes Workbook hat oben einen „Open in Colab“-Link auf `main`:
`https://colab.research.google.com/github/Timbo3399/INFM_SOTA2/blob/main/notebooks/workbook_<name>.ipynb`

Um ein Notebook in Colab zu bearbeiten:

1. In Colab *Datei → Notebook öffnen → GitHub*, Repo `Timbo3399/INFM_SOTA2` und den eigenen Feature-Branch wählen.
2. Speichern mit *Datei → Kopie in GitHub speichern*, dabei den Feature-Branch auswählen.
   Speichern auf `main` schlägt fehl, das ist Absicht.
   Colab setzt dabei den „Open in Colab“-Link oben im Notebook auf den Feature-Branch, und der wird nach dem Merge gelöscht.
   Deshalb vor dem Pull Request den Link in der ersten Zelle lokal wieder auf `blob/main/notebooks/…` setzen. Prüfen mit:
   `grep -o 'colab.research.google.com/github/[^"\\]*' notebooks/*.ipynb`
3. Aussagekräftige Commit-Message statt „Mit Colab erstellt“, z. B. `ch03: Attention-Gewichte geplottet`.

Colab formatiert beim Speichern die ganze Datei neu. Kleine Änderungen wie Tippfehler oder Links deshalb lokal machen.

Sobald ein Notebook Code aus `src/infm_sota2/` nutzt, kommt diese Zelle an den Anfang:

```python
# Colab: eigenes Paket installieren, Colabs torch/tiktoken bleiben unangetastet
%pip install -q --no-deps "git+https://github.com/Timbo3399/INFM_SOTA2.git@main"

from importlib.metadata import version
assert version("torch").split("+")[0] == "2.11.0", f"Colab hat torch {version('torch')}, pyproject erwartet 2.11.0"
assert version("tiktoken") == "0.14.0", f"Colab hat tiktoken {version('tiktoken')}"
```

`--no-deps` verhindert, dass pip torch neu installiert. Statt `@main` geht auch ein Branch, z. B. `@week/02`, solange der Code noch nicht in `main` ist.
Schlägt ein `assert` fehl, hat Colab aktualisiert: dann die Versionen in `pyproject.toml` anheben und `uv lock` ausführen.
Lokal ist das Paket nach `uv sync` schon installiert. Dort meldet `%pip` nur „No module named pip“, weil uv kein pip ins `.venv` legt. Das kann man ignorieren.

## Regeln

- **Ein Notebook, eine Person.** Notebooks sind JSON-Dateien, gleichzeitige Änderungen an derselben Datei lassen sich kaum mergen.
  Wer ein bestehendes Notebook erweitern will, legt eine Kopie an, z. B. `workbook_marvin_attention.ipynb`, oder spricht sich vorher ab.
- **Keine Secrets im Code.** API-Tokens (Hugging Face, OpenAI, Kaggle …) nur über Colab-Secrets (Schlüssel-Symbol links):
  ```python
  from google.colab import userdata
  hf_token = userdata.get("HF_TOKEN")
  ```
  Tokens nie in eine Zelle schreiben und nie ausgeben. Ein versehentlich gepushter Token gilt als kompromittiert und muss sofort widerrufen werden.
- **Keine Modellgewichte oder Datensätze committen.** Die `.gitignore` fängt die üblichen Fälle ab (`*.pth`, `gpt2/`, `data/` …).
  Daten gehören nach `data/`. Notebooks, die lokal laufen, legen sie unter `notebooks/data/` ab, auch das wird ignoriert.
  Colab speichert nur das Notebook selbst, Dateien aus `/content` landen nicht im Repo.
- **Outputs klein halten.** Vor dem Speichern lange Ausgaben (Trainings-Logs, große Tabellen) löschen.
