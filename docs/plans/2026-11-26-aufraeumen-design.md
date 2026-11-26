# Aufräumen nach Woche 6: Entwurf

Stand: freigegeben am 26.11.2026. Ausgangsstand ist `week/06` bei Commit `9bbd922`. Umgesetzt wird auf `chore/aufraeumen`, der Pull Request geht nach `week/06`.

## Warum

Die Gruppe versteht den Code nicht mehr. Nach Woche 6 hat `src/` 34 Dateien mit rund 3.100 Zeilen. Dazu kommen vier Einstiege über `python -m`, ein Hilfsskript, vier Slurm-Skripte und acht Notebooks. Vieles gibt es doppelt: Zwei Module heißen `benchmark`, das Training hat drei Schichten (`train()`, `Trainer.fit` und `cluster.train.run`), dieselben Werte stehen in `TrainConfig` und in `TrainSettings`, und Text kommt auf drei Wegen in einen Loader (`create_dataloader`, `BPEDataset`, `TextSplits`). Toter Code ist dagegen wenig. Ungenutzt sind vor allem die Bausteine aus Woche 2 und 3.

Nach dem Aufräumen soll jede Person aus der Gruppe schnell sehen, wo was liegt und wie man es aufruft.

## Ziele und Grenzen

Ziele:

- Alles, was Woche 6 braucht, läuft weiter, mit denselben Befehlen wie bisher: `bash slurm/setup.sh` und die `sbatch`-Befehle aus `docs/dachs.md`.
- Ein Einstieg für alle Befehle: `python -m infm_sota2 <befehl>`.
- Ein Modul pro Schritt, ohne Unterpakete. Jeder Name lässt sich auf genau einem Weg importieren.
- Was nicht mehr gebraucht wird, liegt in `old/`. Gelöscht wird nichts.
- Eine kürzere README und eine kurze Doku zum Code in `docs/code.md`.

Nicht dabei:

- Neue Funktionen. Neu ist nur der Einstieg `__main__.py`.
- Andere Werte in `configs/`. Dort ändern sich nur Kommentare mit Modulpfaden.
- Andere Ressourcen oder Befehle in den Slurm-Skripten.
- Valentins Branch `feat/slurmTraining`. Er ist nicht in `week/06` und bleibt, wie er ist.
- Die Notebooks in `old/` lauffähig halten.
- Funktionierenden Code umschreiben, wenn dabei nichts Doppeltes wegfällt.

Als Fehlschlag gilt, auch wenn alles läuft:

- Ein Lauf unter `torchrun` rechnet mit derselben Konfiguration andere Zahlen als vorher.
- Jemand aus der Gruppe muss neue `sbatch`-Befehle lernen.
- Code verschwindet, ohne dass sein alter Stand in `old/` liegt.
- Etwas in `old/` wird importiert, geprüft oder getestet.
- `src/` und `tests/` haben danach zusammen mehr Zeilen als vorher, heute sind es 3.095 und 3.029.
- Buchtext, Namen oder Pfade einer Person landen im Repo.

## Rahmenbedingungen

- torch 2.14.1, tiktoken 0.14.0 und wandb 0.28.1 bleiben fest. Neue Laufzeit-Abhängigkeiten kommen nicht dazu.
- Die Regeln aus Woche 4 bis 6 gelten weiter: keine Default-Werte für Hyperparameter, `train()` bleibt frei von W&B, Tests laufen auf der CPU in Sekunden und auch unter Windows, kein Buchtext im Repo, Notebooks ändert nur ein `nbformat`-Skript, und in Skripten und Konfigurationen stehen keine Namen und Pfade einer Person.
- Ausgangsstand der Tests: 210 Tests, alle grün, 12 davon übersprungen (die bash-Tests unter Windows). Ein Durchlauf dauert rund 100 Sekunden.
- Die Läufe von Woche 6 stehen noch aus. Der Cluster holt den Code von GitHub, dort wird also erst nach dem Merge gerechnet.
- Der Umsetzer kommt nicht auf den Cluster und prüft lokal auf der CPU unter Windows.
- Git: Timo ist Autor, ohne Co-Autor- und Session-Zeilen. Gemergt wird mit Merge-Commit.

## Ansätze

| | Weg | Dafür | Dagegen |
|---|---|---|---|
| A | Nur ausmisten: Unterpakete und Importe bleiben, Ungenutztes kommt nach `old/` | kleinster Umbau, kaum Risiko | Was verwirrt, bleibt: zwei `benchmark`, drei Schichten Training, zwei Config-Typen, vier Einstiege |
| B | Ausmisten und zusammenlegen: ein Modul pro Schritt, ein Einstieg | 13 statt 34 Dateien, ein Importweg pro Name, kaum neuer Code | größerer Umbau, Importe in Tests, Slurm-Skripten und Notebook ändern sich |
| C | Neu schreiben, ein Skript pro Job wie `main.py` auf `feat/slurmTraining` | von oben nach unten lesbar | viel neuer Code, die Tests für DDP, Konfiguration und Ergebnisse müssten neu entstehen |

Timo hat am 26.11. Weg B gewählt. Die Verwirrung kommt aus der Struktur, und B ändert sie fast nur durch Verschieben. C ist verworfen.

## Was nach `old/` kommt

`old/` spiegelt die alten Pfade. Dorthin kommt jede Datei, die ganz wegfällt, und der alte Stand jeder Datei, aus der Funktionen, Klassen oder Tests entfernt werden. Code, der nur in ein anderes Modul umzieht, kommt nicht nach `old/`, er steht ja weiter im Code. Ruff und Pyright prüfen nur `src/`, `tests/` und `scripts/`, `unittest` sucht nur in `tests/`. Damit bleibt `old/` außen vor, ohne dass sich an ihrer Konfiguration oder an der CI etwas ändert.

**Notebooks** ziehen unverändert mit ihren Ausgaben nach `old/notebooks/`: `workbook_artur`, `workbook_marvin`, `workbook_marvin_training_v1`, `workbook_timo`, `workbook_timo_attention_v2`, `workbook_timo_lotr` und `workbook_valentin`. In `notebooks/` bleibt nur `workbook_timo_cluster.ipynb`.

**Doku** zieht nach `old/docs/`: `lotr-training.md`, `attention-v2.md` und die drei Pläne aus `docs/plans/`.

**Code**, jeweils der alte Stand der Datei unter `old/src/infm_sota2/`:

| Datei | Was in `src/` wegfällt |
|---|---|
| `data/dataloader.py` | `create_dataloader`, die ganze Datei |
| `data/embedding.py` | `Embeddings`, die ganze Datei |
| `data/dataset.py` | `BPEDataset.decode` und `BPEDataset.vocab_size` |
| `data/splits.py` | der `train_loader` ohne `DistributedSampler`, `summary`, `split_summary` und die Felder, die nur diese brauchen: `stats`, `raw_chars`, `cleaned_chars`, `max_chars`, `seq_len` |
| `transformer/attention.py` | `SelfAttention` und `CausalAttention` |
| `env.py` | der Colab-Teil (`in_colab`, Google Drive, `run_dir`), `login_wandb`, `check_versions`, `version_mismatches`, `describe` und die Felder `local_rank` und `launched_by_torchrun` |
| `training/loop.py` | `max_batches` in `evaluate` und `eval_batches` in `train` |
| `training/trainer.py` | `Trainer`, `TrainConfig` und die Kopie des Checkpoints an einen zweiten Ort |
| `training/report.py` | `ModelReport.add_texts` und `ModelReport.losses` |

**Tests**, jeweils der alte Stand unter `old/tests/`: `test_attention.py`, `test_dataloader.py`, `test_env.py`, `test_splits.py`, `test_training.py`, `test_trainer.py` und `test_report.py`. Die übrigen Tests aus diesen Dateien ziehen in die neuen Testdateien um.

**Pakete:** Aus der Gruppe `dev` in `pyproject.toml` fallen `datasets`, `requests` und `ipywidgets` weg. Sie wurden nur von den alten Notebooks gebraucht. `uv lock` schreibt `uv.lock` neu.

Dazu kommt `old/README.md`. Sie sagt, was dort liegt und warum, dass nichts davon importiert oder getestet wird und dass alles zuletzt auf Commit `9bbd922` zusammen lief (`git switch --detach 9bbd922`). Die Colab-Links in den alten Notebooks zeigen noch auf `notebooks/` in `main` und funktionieren nicht mehr.

`.gitignore` bekommt die Ausnahme `!old/**/data/`, sonst würde eine neue Datei in `old/src/infm_sota2/data/` ignoriert.

## Neuer Aufbau

| Modul | Inhalt | kommt aus |
|---|---|---|
| `__init__.py` | nur ein Docstring, keine Re-Exporte | |
| `__main__.py` | Einstieg für alle Befehle | neu |
| `env.py` | `Environment`, `has_bf16`, Prozessgruppe starten und beenden | `env.py`, `cluster/distributed.py`, `has_bf16` aus `benchmark.py` |
| `config.py` | TOML-Datei lesen und prüfen, `with_grid` | `cluster/config.py` |
| `data.py` | Buchtext bereinigen und teilen, `BPEDataset`, `TextSplits`, Mini-Buch | `data/cleaning.py`, `dataset.py`, `splits.py`, `mini_book.py` |
| `model.py` | GPT-2-Konfigurationen, `GPTModel` samt Bausteinen, `GPTModelV2`, `build_model`, Gewichte laden | `model/*`, `transformer/*` ohne `SelfAttention` und `CausalAttention` |
| `loop.py` | `batch_loss`, `evaluate`, `train`, `TrainResult`, Checkpoints | `training/loop.py` |
| `report.py` | `generate`, `complete`, `longest_copied_span`, `ModelReport` | `training/report.py`, `generate.py`, `analysis.py` |
| `train.py` | `fit` und `run`: ein Lauf nach einer TOML-Datei bis zur JSON-Datei | `cluster/train.py`, `training/trainer.py` |
| `benchmark.py` | Varianten, Messung, FLOPs und MFU, Lauf Nr. 1 | `benchmark.py`, `cluster/benchmark.py` |
| `check.py` | Job 1: Umgebung prüfen | `cluster/check.py` |
| `results.py` | JSON-Dateien schreiben und lesen, Tabellen für Suche und Skalierung | `cluster/results.py` |
| `py.typed` | bleibt | |

Importe gehen nur in eine Richtung:

- `env`, `data`, `model`, `loop` und `results` importieren nichts aus dem Paket.
- `config` nutzt `model` und `loop`, `report` nutzt `data` und `loop`, `check` nutzt `env`.
- `benchmark` nutzt `env`, `model`, `loop`, `check` und `results`.
- `train` nutzt alle bisher genannten Module.
- `__main__` nutzt `benchmark`, `check`, `data` und `train`.

`has_bf16` zieht nach `env.py`, weil `check` sie braucht und `benchmark` jetzt `check` importiert. Sonst entstünde ein Kreis.

### Ein Einstieg

```
python -m infm_sota2 check [--no-internet]                    Job 1, Teil 1
python -m infm_sota2 benchmark [--smoke]                      Job 1, Teil 2 (Lauf Nr. 1)
python -m infm_sota2 train --config <datei> [--grid-index N]  ein Trainingslauf
python -m infm_sota2 mini-book <ordner>                       Mini-Buch für den Smoke-Lauf
```

`__main__.py` ordnet jedem Befehl die `main(argv)` seines Moduls zu. Jedes Modul liest seine Optionen weiter selbst mit `argparse`. Fehlt der Befehl oder ist er unbekannt, gibt `__main__.py` eine Zeile mit allen Befehlen aus und endet mit Code 2. Unter `torchrun` heißt der Aufruf `torchrun ... -m infm_sota2 train --config ...`. Die bisherigen Einstiege `python -m infm_sota2.cluster.*` und `python -m infm_sota2.data.mini_book` gibt es danach nicht mehr.

## Änderungen im Einzelnen

Was hier nicht steht, zieht unverändert um.

**`env.py`:** `Environment` behält `base_dir`, `device`, `wandb_mode`, `data_dir`, `rank` und `world_size`, dazu `checkpoint_dir` (`base_dir / "checkpoints"`) und `gpu_name`. `detect(smoke_test)` nimmt immer den Arbeitsordner als `base_dir`. Das Gerät ist `cuda:<LOCAL_RANK>`, ohne `torchrun` also `cuda:0`, ohne GPU die CPU. Die Prüfung der torchrun-Variablen (alle drei oder keine) und von `WANDB_MODE` bleibt, wie sie ist. `init_distributed` und `destroy_distributed` ziehen aus `cluster/distributed.py` hierher.

**`data.py`:** `clean_book_text`, `split_text` und `CleaningStats` bleiben unverändert. `TextSplits` hat nur noch die Felder `train`, `val`, `test` und `datasets`, `from_book` behält seine Parameter. Es gibt einen einzigen `train_loader(batch_size, seed, rank, world_size)`, das ist der bisherige `distributed_train_loader`. Ohne `torchrun` läuft er mit Rang 0 von 1. Dazu kommen `loader(split, batch_size)`, `tokenizer` und das Mini-Buch mit `MINI_BOOK`, `write_mini_book` und `main`.

**`model.py`:** Die Klassen behalten ihre Namen und Attribute. Damit bleiben die Schlüssel im `state_dict` gleich, und Checkpoints und die OpenAI-Gewichte laden wie bisher. `MODEL_CLASSES = {"GPTModel": GPTModel, "GPTModelV2": GPTModelV2}` steht nur noch hier. `config.py` prüft `model_class` gegen diese Schlüssel, `train.py` holt daraus die Klasse.

**`loop.py`:** `evaluate(model, loader, device)` nimmt immer alle Batches, `train()` hat kein `eval_batches` mehr. Woche 6 wertet ohnehin den ganzen Validierungsteil aus.

**`report.py`:** `ModelReport` behält `add`, `test_loss` und `completions`.

**`train.py`:** `Trainer` und `TrainConfig` fallen weg. An ihre Stelle tritt eine Funktion:

```python
def fit(env, model, data, settings: TrainSettings, seed, name, group,
        checkpoint_name, wandb_config) -> TrainResult
```

Sie macht dasselbe wie bisher `Trainer.fit` unter `torchrun`, mit vier Unterschieden:

1. Der Train-Loader ist immer der mit `DistributedSampler`. Rang und Zahl der Prozesse kommen aus `env`.
2. Der Checkpoint wird direkt nach `env.checkpoint_dir / checkpoint_name` geschrieben. Die Kopie aus `checkpoints/runs/` fällt weg.
3. Lernrate und Epochen kommen aus `settings`.
4. In der W&B-Konfiguration fehlt die Tabelle `run`, weil Lernrate und Epochen schon unter `train` stehen.

`run()` und `main()` bleiben, wie sie in `cluster/train.py` sind. `run_label` und `config_dict` aus `config.py` stehen dort direkt im Code, als `f"{config.name}-gpus{env.world_size}"` und `asdict(config)`. `REPO_DIR` liegt eine Ebene höher: `Path(__file__).resolve().parents[2]`.

**`benchmark.py`, `check.py`, `results.py` und `config.py`** ziehen um, nur ihre Importe ändern sich.

## Tests

Die Testdateien heißen wie die Module:

| Neu | kommt aus |
|---|---|
| `test_env.py` | `test_env.py`, `test_cluster_distributed.py` |
| `test_config.py` | `test_cluster_config.py` |
| `test_data.py` | `test_cleaning.py`, `test_splits.py`, `test_mini_book.py` und die `BPEDataset`-Tests aus `test_dataloader.py` |
| `test_model.py` | `test_model.py`, `test_build.py`, `test_gpt_v2.py`, `test_pretrained.py` und die `MultiHeadAttention`-Tests aus `test_attention.py` |
| `test_loop.py` | `test_training.py` |
| `test_report.py` | `test_report.py`, `test_generate.py`, `test_analysis.py` |
| `test_train.py` | `test_cluster_train.py`, `test_trainer.py` |
| `test_benchmark.py` | `test_benchmark.py`, `test_cluster_benchmark.py` |
| `test_check.py` | `test_cluster_check.py` |
| `test_results.py` | `test_cluster_results.py` |
| `test_main.py` | neu, für `__main__.py` |

`test_ddp.py`, `test_scripts.py`, `test_notebooks.py` und `helpers.py` bleiben, wo sie sind.

- Jeder übrige Test prüft weiter dasselbe. Es ändern sich nur Importe, Namen und Fixtures.
- Wo sich das Verhalten mit Absicht ändert, also beim Train-Loader ohne `torchrun` und beim Ort des Checkpoints, wird der Test angepasst. Jede andere Abweichung ist ein Fehler im Umbau.
- Tests, die einen Befehl als eigenen Prozess starten, rufen `python -m infm_sota2 <befehl>` auf. `mini-book` startet weiter mit `-W error`.
- `mock.patch` zielt auf die neuen Pfade, etwa `infm_sota2.train.download_weights`.
- Das nachgebaute `uv` in `test_scripts.py` erkennt `run python -m infm_sota2 mini-book` und startet dafür das echte Paket mit `"$TEST_PYTHON" -m infm_sota2 mini-book`.
- `test_notebooks.py` prüft nur noch `workbook_timo_cluster.ipynb`.
- `test_main.py` prüft zwei Fälle: Ein bekannter Befehl ruft die passende `main` mit den restlichen Argumenten auf. Ein unbekannter oder fehlender Befehl endet mit Code 2 und nennt die Befehle.

## Skripte, Konfigurationen und Notebook

- `slurm/check.sbatch`, `train.sbatch` und `search.sbatch` rufen `-m infm_sota2 check`, `python -m infm_sota2 benchmark` und `-m infm_sota2 train` auf. Sonst ändert sich an ihnen nichts, auch nicht an den `sbatch`-Befehlen für die Gruppe.
- `slurm/setup.sh` schreibt das Mini-Buch mit `uv run python -m infm_sota2 mini-book "$DATA_DIR"`. Der ausgegebene Befehl zum Laden der GPT-2-Gewichte importiert aus `infm_sota2.model`.
- Der Kommentar in `configs/smoke-cpu.toml` nennt `uv run python -m infm_sota2 train --config configs/smoke-cpu.toml`. Der veraltete Hinweis auf „Schritt 3“ fällt weg.
- `notebooks/workbook_timo_cluster.ipynb` ändert ein `nbformat`-Skript, das nicht ins Repo kommt, jeweils in dem Schritt, in dem sich die Stelle ändert. Die drei Links auf `docs/attention-v2.md` zeigen auf `old/docs/attention-v2.md`, `run_dir=...` fällt aus dem `replace(...)` in `smoke_results`, und die Importe zeigen auf die neuen Module. Die gespeicherten Ausgaben bleiben, `BRANCH` bleibt `"main"`.
- `scripts/smoke_notebook.py` bleibt. Damit lässt sich prüfen, ob das Notebook nach einer Änderung am Code noch läuft.

## Doku

- **`README.md`** wird neu geschrieben, mit dem Ziel von höchstens 130 Zeilen. Sie enthält: was das Projekt ist, den Aufbau des Repos als kurze Tabelle mit Verweis auf `docs/code.md`, die Einrichtung auf dem eigenen Rechner (`uv sync`, Git-Hook, Zeilenenden, W&B-Login), die Befehle, die Checks, die Versionen samt dem Block für einen anderen CUDA-Index, die Arbeitsweise mit Branches und mit Colab für das eine Notebook und die Regeln. Was nur zum Cluster gehört, steht allein in `docs/dachs.md`. Das Beispiel mit `wandb.init` von Hand fällt weg, weil `fit` den Run anlegt. Die Zelle mit dem Versionsvergleich für Colab fällt mit `check_versions` weg.
- **`docs/code.md`** ist neu und hat höchstens 120 Zeilen: was der Code macht, die Module mit je einem Satz, der Ablauf eines Laufs von der TOML-Datei bis zur JSON-Datei, die Befehle, wie man einen Lauf oder eine Einstellung hinzufügt und was man vor dem Commit prüft.
- **`docs/dachs.md`:** Der Link auf `lotr-training.md` zeigt auf `../old/docs/lotr-training.md`. Der Verweis auf den CUDA-Index nennt den Abschnitt „Versionen“ der README. Die Befehle bleiben.
- **`docs/cluster-training.md`** bleibt in `docs/`, weil dort die Läufe und die Erwartungen für die Auswertung stehen. Wo der Entwurf Module und Klassen nennt, beschreibt er den Aufbau vor dem Aufräumen, und das nicht nur in Teil 4. Ein Satz unter der Zeile „Stand:“ sagt das und verweist auf `docs/code.md`. Sonst bleibt der Text, wie er ist.
- Der Verweis in `data.py` auf die Regeln der Bereinigung zeigt auf `old/docs/lotr-training.md`.
- In `old/docs/` werden nur Links zwischen den Dokumenten repariert, etwa von `lotr-training.md` auf `../../docs/cluster-training.md`. Die Pfade im Text bleiben, wie sie waren.

## Prüfen

1. **Gleiche Zahlen:** Vor dem Umbau, auf `9bbd922`, einen Referenzlauf machen: `cluster.train.run` mit `configs/smoke-cpu.toml` auf der CPU, Rang 0 von 1 und `launched_by_torchrun=True`, also wie unter `torchrun`, in einem temporären Ordner mit dem Mini-Buch. Er läuft zweimal, damit sich zeigt, ob die Werte genau gleich bleiben. Nach dem Umbau läuft derselbe Lauf mit `train.run`. In der JSON-Datei müssen Losses, Schritte, Tokens, Texte und kopierte Folgen gleich sein, Zeiten und Umgebung dürfen sich unterscheiden. Ist schon der Referenzlauf nicht genau wiederholbar, reicht eine Abweichung in derselben Größe.
2. **Checks:** `ruff check`, `ruff format --check`, `pyright` und alle Tests sind grün. Die bash-Tests werden unter Windows übersprungen. Lassen sie sich in Git Bash erzwingen, laufen sie einmal lokal, sonst prüft sie erst die CI.
3. **Befehle:** In einem temporären Ordner, damit nichts unter `results/` im Repo landet, laufen `mini-book`, `train --config configs/smoke-cpu.toml`, `benchmark --smoke` und `check --no-internet` ohne Fehler.
4. **Notebook:** `uv run --with nbclient python scripts/smoke_notebook.py notebooks/workbook_timo_cluster.ipynb <ausgabe>` läuft ohne Fehler.
5. **Reste:** In `src/`, `tests/`, `scripts/`, `slurm/`, `configs/`, `docs/` und im Notebook steht kein alter Modulpfad mehr (`infm_sota2.cluster`, `infm_sota2.training`, `infm_sota2.transformer`, `infm_sota2.data.`, `infm_sota2.model.`). Ausgenommen sind `old/`, `docs/plans/` und `docs/cluster-training.md`.
6. **Umfang:** `src/` und `tests/` haben zusammen weniger Zeilen als vorher.
7. **Cluster, nach dem Merge:** Timo startet `sbatch slurm/check.sbatch` und den Smoke-Lauf auf einem und auf zwei Knoten, bevor die echten Läufe beginnen.

## Reihenfolge und Commits

Auf `chore/aufraeumen`. Nach jedem Commit sind alle Tests grün.

1. `docs:` Entwurf und Plan, der Plan als `docs/plans/2026-11-26-aufraeumen-plan.md`
2. `chore:` Notebooks, Doku und Pläne von Woche 1 bis 5 nach `old/`. Dazu kommen `old/README.md`, `test_notebooks.py`, die Pakete in `pyproject.toml` samt `uv.lock` und alle Links auf verschobene Dateien, auch im Notebook und in `data/cleaning.py`.
3. `refactor:` Ungenutzter Code und seine Tests nach `old/`, noch in den bisherigen Dateien, dazu `fit` statt `Trainer`. `old/README.md` bekommt den Teil über Code und Tests, das Notebook verliert `run_dir=...`.
4. `feat:` Ein Einstieg `python -m infm_sota2`, dazu Slurm-Skripte, `setup.sh`, Kommentare in `configs/` und die Tests dafür
5. `refactor:` Ein Modul pro Schritt, die Tests nach Modulen und die Importe im Notebook
6. `docs:` README, `docs/code.md`, der Hinweis in `docs/cluster-training.md` und der Verweis auf die README in `docs/dachs.md`

Schritt 3 kommt vor Schritt 5, damit der Diff in den bekannten Dateien zeigt, was wegfällt. Schritt 4 kommt vor Schritt 5, damit sich die Slurm-Skripte nur einmal ändern und zwischen zwei Commits nie auf ein fehlendes Modul zeigen.

## Entscheidungen beim Entwurf

Diese Punkte hat der Entwurf festgelegt, ohne dass sie besprochen wurden:

1. Die Notebooks von Artur, Marvin und Valentin kommen mit nach `old/`, wie in Variante 1.
2. Der Colab-Teil von `Environment` kommt nach `old/`. Das Notebook von Woche 6 braucht ihn nicht. Wer in Woche 7 wieder in Colab trainiert, holt ihn von dort zurück.
3. Ohne `torchrun` mischt der Train-Loader wie unter `torchrun`. Lokale Smoke-Läufe und die Smoke-Läufe im Notebook rechnen deshalb andere Zahlen als vorher, die ohnehin keine Aussagekraft hatten.
4. Checkpoints landen direkt in `checkpoints/`. Bricht ein Lauf ab, liegt dort trotzdem sein bis dahin bester Stand.
5. `evaluate` und `train` verlieren `max_batches` und `eval_batches`, weil Woche 6 sie nicht nutzt.
6. `docs/cluster-training.md` bleibt in `docs/`, mit einem Hinweis oben, dass seine Modulnamen den alten Aufbau meinen.
7. `scripts/smoke_notebook.py` bleibt.
8. Entwurf und Plan liegen in `docs/plans/`, die alten Pläne in `old/docs/plans/`.

## Offen für die Umsetzung

- Die Reihenfolge der Funktionen in einem Modul, am besten so, wie sie im Ablauf vorkommen.
- Ob die `argparse`-Parser `prog="python -m infm_sota2 <befehl>"` bekommen.
- Die Formulierungen in README und `docs/code.md`, solange beide kurz bleiben.
- Das `nbformat`-Skript für das Notebook.
