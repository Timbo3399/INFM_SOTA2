# Der Code

Der Code trainiert GPT-2 auf dem Text von *The Lord of the Rings*, auf dem Cluster auch mit mehreren GPUs, und schreibt für jeden Lauf eine JSON-Datei mit allen Einstellungen und Ergebnissen. Er liegt in `src/infm_sota2/`, ein Modul pro Schritt.

## Module

In der Reihenfolge, in der ein Lauf sie braucht:

| Modul | Wofür |
|---|---|
| `__main__.py` | der Einstieg `python -m infm_sota2 <befehl>` |
| `config.py` | liest eine TOML-Datei aus `configs/` und prüft jeden Schlüssel |
| `env.py` | Arbeitsordner, Gerät, Rang unter `torchrun`, W&B-Modus und die Prozessgruppe für DDP |
| `data.py` | bereinigt den Buchtext, teilt ihn in Training, Validierung und Test und baut Token-Fenster und Loader, dazu das Mini-Buch für Smoke-Läufe |
| `model.py` | `GPTModel` mit eigenen Bausteinen, `GPTModelV2` aus den Bausteinen von PyTorch, die GPT-2-Konfigurationen und die OpenAI-Gewichte |
| `loop.py` | die Trainingsschleife `train`, dazu `evaluate` und die Checkpoints |
| `report.py` | Test-Loss, Perplexity, erzeugte Texte und wie viel davon wörtlich aus dem Trainingstext stammt |
| `train.py` | ein ganzer Lauf: `fit` setzt DDP, `torch.compile`, AdamW und W&B auf, `run` führt alles bis zur JSON-Datei zusammen |
| `results.py` | schreibt und liest die JSON-Dateien und rechnet die Tabellen für Suche und Skalierung |
| `check.py` | Job 1: prüft Rechner, GPU, Treiber, NCCL und Datenordner |
| `benchmark.py` | misst Tokens/s, Speicher und MFU, das ist Lauf Nr. 1 |

## Ein Lauf

`python -m infm_sota2 train --config configs/<lauf>.toml` geht so vor:

1. `config.load_config` liest die TOML-Datei. Fehlt ein Schlüssel oder ist einer unbekannt, bricht der Lauf mit dem Namen des Schlüssels ab.
2. `env.Environment.detect` liest Rang und Zahl der Prozesse aus den Variablen von `torchrun`. Ab zwei Prozessen startet `env.init_distributed` die Prozessgruppe.
3. `data.TextSplits.from_book` bereinigt den Buchtext aus dem Datenordner und teilt ihn.
4. `model.build_model` baut das Modell mit festem Seed. Ist `evaluate_pretrained` gesetzt, misst `report.ModelReport` vorher GPT-2 mit den OpenAI-Gewichten.
5. `train.fit` trainiert mit `loop.train`. Ist `save_checkpoint` gesetzt, lädt es danach den Stand mit dem besten Validierungs-Loss zurück.
6. `report.ModelReport` misst den Test-Loss und erzeugt die Texte.
7. `results.write_result` schreibt alles nach `results/week06/<benutzer>/<gruppe>/<lauf>-job<id>.json`.

Unter `torchrun` trainieren alle Prozesse mit. Nur Rang 0 loggt nach W&B, schreibt den Checkpoint, misst danach und schreibt die JSON-Datei. Scheitert ein Lauf an NaN im Loss oder an zu wenig GPU-Speicher, steht der Fehler in der JSON-Datei, und der Lauf bricht trotzdem mit dem Fehler ab.

## Befehle

| Befehl | Wofür |
|---|---|
| `python -m infm_sota2 check [--no-internet]` | Job 1, Teil 1: Umgebung prüfen |
| `python -m infm_sota2 benchmark [--smoke]` | Job 1, Teil 2: Benchmark, Lauf Nr. 1 |
| `python -m infm_sota2 train --config <datei> [--grid-index N]` | ein Trainingslauf, `--grid-index` wählt in der Suche eine Kombination |
| `python -m infm_sota2 mini-book <ordner>` | schreibt das Mini-Buch für den Smoke-Lauf |

Lokal stellt man `uv run` davor. Auf dem Cluster laufen die Befehle über die Jobskripte, etwa `sbatch slurm/train.sbatch --config configs/smoke.toml`. Welche Jobs es gibt, steht in [dachs.md](dachs.md), welche Läufe es gibt und warum, in [cluster-training.md](cluster-training.md).

## Etwas ändern

- **Ein neuer Lauf** ist eine neue TOML-Datei in `configs/`. Am einfachsten kopiert man eine ähnliche und ändert die Werte. Jede Einstellung steht in der Datei, Defaults gibt es nicht.
- **Eine neue Einstellung** braucht ein Feld in der passenden Datenklasse in `config.py`, einen Eintrag in der Schlüsseltabelle darunter, den Schlüssel in jeder Datei in `configs/` und einen Test in `tests/test_config.py`.
- **Tests** liegen in `tests/test_<modul>.py`. `tests/helpers.py` hat ein winziges Modell, das Mini-Buch und eine Umgebung auf der CPU.
- **Vor dem Commit** laufen die Checks aus der [README](../README.md#checks). Nach Änderungen an `config.py`, `results.py` oder `train.py` gehört das Notebook im Smoke-Test dazu:
  ```bash
  uv run --with nbclient python scripts/smoke_notebook.py notebooks/workbook_timo_cluster.ipynb ausgabe.txt
  ```

Was nicht mehr gebraucht wird, auch alter Code, liegt in [old/](../old/README.md). Nichts davon wird importiert.
