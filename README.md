# INFM_SOTA2

Gruppenprojekt im Modul SOTA 2, Master Informatik an der Hochschule Offenburg.

Wir bauen ein Large Language Model von Grund auf – Schritt für Schritt in PyTorch.

## Projektstruktur

| Pfad | Inhalt |
|---|---|
| `notebooks/` | Notebooks zum Anschauen, lokal oder in Colab. Eins pro Person: `workbook_<name>.ipynb`. |
| `src/infm_sota2/` | Gemeinsamer Python-Code, später auch die Trainingsskripte für den Cluster. |
| `pyproject.toml`, `uv.lock` | Abhängigkeiten und die exakt festgelegten Versionen. |

Entwickelt wird in `.py`-Dateien unter `src/`. Die Notebooks zeigen und erklären, was der Code macht.

## Lokale Umgebung mit uv

1. [uv installieren](https://docs.astral.sh/uv/getting-started/installation/), mindestens Version 0.13. Ein vorhandenes uv mit `uv self update` aktualisieren.
2. Im Repo `uv sync` ausführen. uv lädt dabei Python 3.12 und legt `.venv` an.
3. Notebooks in VS Code öffnen und als Kernel `.venv` wählen. Ohne VS Code geht `uv run --with jupyterlab jupyter lab`.

Unter Linux lädt `uv sync` torch mit CUDA 13.0, das sind mehrere GB. Windows und Macs mit Apple Silicon bekommen die CPU-Version.
Für Intel-Macs gibt es kein torch 2.11, dort bitte Colab nutzen.

### Versionen

torch und tiktoken sind exakt auf die Versionen der Colab-Laufzeit festgelegt (`torch==2.11.0`, `tiktoken==0.14.0`, Python 3.12 oder 3.13).
So läuft lokal, auf dem Cluster und in Colab dasselbe.
Welche Versionen Colab gerade hat, steht in [googlecolab/backend-info](https://github.com/googlecolab/backend-info). Angehoben wird nur zusammen mit Colab.

Pakete hinzufügen:

- Nur für Notebooks (Plots, Tabellen, Widgets): `uv add --dev <paket>`.
- Pakete, die Ergebnisse beeinflussen, mit der exakten Colab-Version, z. B. numpy, sobald es im Code direkt verwendet wird: `uv add "numpy==2.1.3"`.
- `uv.lock` immer mit committen.

### Auf dem Cluster

```bash
uv sync --no-dev
```

installiert nur, was die Skripte brauchen (torch, tiktoken), ohne die Notebook-Pakete.

Rechenknoten haben oft kein Internet, tiktoken lädt die GPT-2-Daten aber beim ersten Aufruf herunter.
Deshalb einmal auf dem Login-Knoten ausführen und `TIKTOKEN_CACHE_DIR` im Jobskript genauso setzen:

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

`--no-deps` verhindert, dass pip torch neu installiert. Statt `@main` geht auch ein Branch, z. B. `@feature/attention`.
Schlägt ein `assert` fehl, hat Colab aktualisiert: dann die Versionen in `pyproject.toml` anheben und `uv lock` ausführen.

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
