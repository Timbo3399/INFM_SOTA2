# Aufräumen nach Woche 6: Implementierungsplan

> **Für Agenten:** Pflicht-Skill zum Umsetzen: superpowers:subagent-driven-development oder superpowers:executing-plans. Die Schritte sind Checkboxen (`- [ ]`).

**Ziel:** Der Code von Woche 6 liegt in elf flachen Modulen mit einem Einstieg `python -m infm_sota2 <befehl>`. Alles, was nicht mehr gebraucht wird, liegt in `old/`, und README und `docs/code.md` erklären kurz, wie man den Code benutzt.

**Architektur:** Weg B aus dem Entwurf. Zuerst wandern Notebooks und Doku nach `old/`, dann ungenutzter Code und seine Tests, noch in den bisherigen Dateien. Danach kommt der Einstieg `__main__.py`, dann werden die Unterpakete eins nach dem anderen zu Modulen, zuletzt kommt die Doku. Die Schritte 3 und 5 des Entwurfs sind auf mehrere Commits verteilt. Nach jeder Aufgabe sind alle Tests grün.

**Technik:** Python 3.13 (Syntax bis 3.12), torch 2.14.1, tiktoken 0.14.0, wandb 0.28.1, unittest, nbformat, Bash, Slurm.

**Spezifikation:** [`docs/plans/2026-11-26-aufraeumen-design.md`](2026-11-26-aufraeumen-design.md)

## Globale Vorgaben

- Keine neuen Abhängigkeiten. Keine Default-Werte für Hyperparameter in Signaturen. `train()` bleibt frei von W&B.
- Tests mit `unittest` auf der CPU, in Sekunden, auch unter Windows. Tests mit bash oder `torchrun` überspringen sich unter Windows, sie laufen in der CI.
- Vor jedem Commit grün: `uv run ruff check`, `uv run ruff format --check`, `uv run pyright`, `uv run python -m unittest discover -s tests -p 'test*.py'`.
- Regel für `old/`: Fällt eine Datei ganz weg, zieht sie mit `git mv` nach `old/<alter Pfad>`. Verliert eine Datei Funktionen, Klassen oder Tests, kommt vorher ihr Stand von `9bbd922` nach `old/<alter Pfad>`, mit `git show 9bbd922:<pfad> > old/<pfad>`. Code, der nur umzieht, kommt nicht nach `old/`.
- Nichts in `old/` wird importiert. Ruff, Pyright und `unittest` bleiben auf `src/`, `tests/` und `scripts/` beschränkt.
- Notebooks ändert nur `<tmp>/nb_replace.py` aus Task 2. `<tmp>` ist ein Ordner außerhalb des Repos, auch für Probeläufe, deren Ergebnisse nie unter `results/` im Repo landen.
- Kein Buchtext und keine Namen, E-Mail-Adressen oder Pfade einer Person in Code, Skripten oder Konfigurationen.
- Git: Branch `chore/aufraeumen`. Commit-Messages mit Präfix und deutscher Beschreibung, ohne Co-Autor- und Session-Zeilen. Gepusht wird erst am Ende nach Freigabe.

## Review-Fokus

1. **Das Notebook in Colab** installiert das Paket von `main` und importiert die neuen Module. Erwartet: Jedes `from infm_sota2... import ...` im Notebook findet Modul und Namen. Test in Task 11 (`test_notebook_imports_exist`).
2. **Das Notebook liest die JSON-Dateien** über feste Schlüssel. Erwartet: Ein Lauf schreibt dieselben Schlüssel wie vorher. Test in Task 5 (`test_record_keeps_its_keys`).
3. **`torchrun` startet den neuen Einstieg** auf dem Cluster. Erwartet: `torchrun --standalone --nproc-per-node=2 -m infm_sota2 check --no-internet` meldet beide Ränge. Test in Task 6 (`CheckMainTest`, nur Linux, also in der CI).
4. **Den Befehl zum Laden der GPT-2-Gewichte**, den `setup.sh` ausgibt, kopiert man und führt ihn auf dem Login-Knoten aus. Erwartet: Sein Import funktioniert. Test in Task 6 (`test_setup_download_command_imports`, auch unter Windows).
5. **Aufruf ohne Befehl, mit Tippfehler oder mit `-h`.** Erwartet: die Zeile mit allen Befehlen, Code 2 bei einem Fehler, Code 0 bei `-h`. Test in Task 6 (`test_main.py`).

---

### Task 1: Ausgangswerte festhalten, Entwurf und Plan committen

Läuft vor jeder Änderung an `src/`.

**Dateien:**
- Neu, außerhalb des Repos: `<tmp>/reference_run.py`
- Ändern und committen: `docs/plans/2026-11-26-aufraeumen-design.md` (Zeile „Stand:“: freigegeben am 26.11.2026), `docs/plans/2026-11-26-aufraeumen-plan.md`

**Schnittstelle:**
- `reference_run.py <repo> <ausgabe.json>` stellt `<repo>` an den Anfang von `sys.path` und baut mit `tests.helpers.local_env` eine Umgebung in einem temporären Ordner. Solange `Environment` das Feld `launched_by_torchrun` hat, setzt es dieses auf `True`, so rechnet der Lauf wie unter `torchrun`. Es schreibt das Mini-Buch nach `env.data_dir` und ruft ohne Ausgabe `run(load_config(<repo>/configs/smoke-cpu.toml), env, "ref", None, <repo>)` auf. `load_config` und `run` kommen aus `infm_sota2.config` und `infm_sota2.train`, solange es die nicht gibt, aus `infm_sota2.cluster.config` und `infm_sota2.cluster.train`. Gespeichert werden `label`, `result` ohne `tokens_per_second`, `test_loss`, `perplexity`, `texts` und `copied`.

- [ ] **Schritt 1:** Skript schreiben und zweimal laufen lassen: `uv run python <tmp>/reference_run.py . <tmp>/ref-1.json`, dann dasselbe nach `<tmp>/ref-2.json`. Erwartet: beide Dateien gleich. Sonst notieren, welche Werte schwanken. Task 5, 11 und 13 vergleichen dann mit dieser Toleranz.
- [ ] **Schritt 2:** `uv run --with nbclient python scripts/smoke_notebook.py notebooks/workbook_timo_cluster.ipynb <tmp>/cluster-before.txt`. Erwartet: läuft ohne Fehler.
- [ ] **Schritt 3:** Zeilen zählen: `cat $(git ls-files 'src/*.py') | wc -l` ergibt 3095, `cat $(git ls-files 'tests/*.py') | wc -l` ergibt 3029.
- [ ] **Schritt 4:** Commit `docs: Entwurf und Plan fürs Aufräumen nach Woche 6`.

### Task 2: Notebooks, Doku und Pläne von Woche 1 bis 5 nach `old/`

**Dateien:**
- `git mv`: `notebooks/workbook_{artur,marvin,marvin_training_v1,timo,timo_attention_v2,timo_lotr,valentin}.ipynb` nach `old/notebooks/`; `docs/lotr-training.md` und `docs/attention-v2.md` nach `old/docs/`; `docs/plans/2026-11-12-lotr-training-plan.md`, `2026-11-19-attention-v2-plan.md` und `2026-11-21-cluster-training-plan.md` nach `old/docs/plans/`
- Neu: `old/README.md`, außerhalb des Repos `<tmp>/nb_replace.py`
- Ändern: `tests/test_notebooks.py`, `pyproject.toml`, `uv.lock`, `README.md`, `docs/dachs.md`, `docs/cluster-training.md`, `src/infm_sota2/data/cleaning.py`, `notebooks/workbook_timo_cluster.ipynb`, `old/docs/lotr-training.md`, `old/docs/plans/2026-11-21-cluster-training-plan.md`

**Schnittstelle:**
- `nb_replace.py <notebook> <ersetzungen.json>`: Die JSON-Datei ist eine Liste von Paaren `[alt, neu]`. Das Skript liest das Notebook mit `nbformat` (Version 4), ersetzt in den Quelltexten aller Zellen jedes `alt` durch `neu`, bricht ab, wenn ein `alt` nirgends vorkommt, und schreibt mit `nbformat.write`. Ausgaben bleiben unberührt. Lesen und Schreiben ohne Ersetzung ändert die Datei nicht, das ist geprüft.
- `old/README.md`, erste Fassung: was in `old/` liegt und warum, dass nichts davon importiert, geprüft oder getestet wird, dass alles zuletzt auf Commit `9bbd922` zusammen lief (`git switch --detach 9bbd922`) und dass die Colab-Links der alten Notebooks nicht mehr gehen. Dazu eine Tabelle „Datei | warum hier“, die Task 3 bis 5 fortschreiben.

- [ ] **Schritt 1:** In `tests/test_notebooks.py` heißt die Liste `NOTEBOOKS` und enthält nur `workbook_timo_cluster.ipynb`. Danach die Dateien verschieben und die Tests laufen lassen. Erwartet: grün.
- [ ] **Schritt 2:** Links ersetzen. Im Notebook mit `nb_replace.py` `docs/attention-v2.md` durch `old/docs/attention-v2.md`. In `README.md`, `docs/cluster-training.md` und `cleaning.py` `docs/lotr-training.md` durch `old/docs/lotr-training.md`, in `README.md` außerdem `docs/attention-v2.md` durch `old/docs/attention-v2.md`. In `docs/dachs.md` `[Anleitung](lotr-training.md)` durch `[Anleitung](../old/docs/lotr-training.md)`. In `old/docs/lotr-training.md` `(cluster-training.md)` durch `(../../docs/cluster-training.md)`, im Plan vom 21.11. `(../cluster-training.md)` durch `(../../../docs/cluster-training.md)`.
- [ ] **Schritt 3:** Prüfen: Jeder relative Link in `old/docs/` zeigt auf eine vorhandene Datei, und `grep -rn "lotr-training.md\|attention-v2.md" --include=*.md --include=*.py --include=*.ipynb README.md docs src notebooks` findet außerhalb von `docs/plans/` nur noch Pfade mit `old/docs/`.
- [ ] **Schritt 4:** In `pyproject.toml` fallen `datasets`, `requests` und `ipywidgets` aus `[dependency-groups] dev`, der Kommentar dazu nennt das Notebook und die Checks. Dann `uv lock` und `uv sync`. Erwartet: `git diff uv.lock` entfernt nur diese drei und Pakete, die nur sie brauchten.
- [ ] **Schritt 5:** `old/README.md` schreiben, Checks laufen lassen. Erwartet: grün.
- [ ] **Schritt 6:** Commit `chore: Notebooks, Doku und Pläne von Woche 1 bis 5 nach old/`.

### Task 3: Bausteine aus Woche 2 und 3 nach `old/`

**Dateien:**
- `git mv`: `src/infm_sota2/data/dataloader.py` und `src/infm_sota2/data/embedding.py` nach `old/src/infm_sota2/data/`
- Alter Stand nach `old/`: `src/infm_sota2/data/dataset.py`, `src/infm_sota2/transformer/attention.py`, `tests/test_dataloader.py`, `tests/test_attention.py`
- Ändern: `.gitignore`, `src/infm_sota2/__init__.py`, `src/infm_sota2/data/__init__.py`, `src/infm_sota2/data/dataset.py`, `src/infm_sota2/transformer/__init__.py`, `src/infm_sota2/transformer/attention.py`, `tests/helpers.py`, `tests/test_build.py`, `tests/test_dataloader.py`, `tests/test_attention.py`, `old/README.md`

**Schnittstelle:**
- `src/infm_sota2/__init__.py` hat nur einen Docstring, der auf `docs/code.md` verweist, und keine Re-Exporte. `GPTModel` kommt aus `infm_sota2.model`.
- `BPEDataset` hat kein `decode` und kein `vocab_size` mehr. `infm_sota2.data` exportiert `BPEDataset`, `CleaningStats`, `TextSplits`, `clean_book_text` und `split_text`. `infm_sota2.transformer` exportiert `FeedForward`, `LayerNorm`, `MultiHeadAttention` und `TransformerBlock`.

- [ ] **Schritt 1:** In `.gitignore` bei den Ausnahmen für Paket-Code `!old/**/data/` ergänzen. Prüfen: `git check-ignore -v old/src/infm_sota2/data/x.py` meldet nichts.
- [ ] **Schritt 2:** Alten Stand der vier Dateien nach `old/` schreiben, dann die zwei `git mv`.
- [ ] **Schritt 3:** Tests kürzen. In `tests/test_dataloader.py` bleibt `BPEDatasetTest` ohne `test_decodes_tokens_and_exposes_vocabulary_size`. In `tests/test_attention.py` bleiben `split_heads` und `MultiHeadAttentionTest` ohne `test_processes_embedded_batches_from_dataloader`. `tests/helpers.py` und `tests/test_build.py` importieren `GPTModel` aus `infm_sota2.model`.
- [ ] **Schritt 4:** Code kürzen wie unter Schnittstelle.
- [ ] **Schritt 5:** Prüfen: `grep -rnw "create_dataloader\|Embeddings\|SelfAttention\|CausalAttention\|vocab_size" src tests scripts notebooks` findet nur `vocab_size` als Schlüssel einer Modell-Konfiguration, und `grep -rn "from infm_sota2 import" src tests scripts notebooks` findet nichts.
- [ ] **Schritt 6:** `old/README.md` fortschreiben, Checks, Commit `refactor: Bausteine aus Woche 2 und 3 nach old/`.

### Task 4: Optionen, die nur die alten Notebooks nutzten

**Dateien:**
- Alter Stand nach `old/`: `src/infm_sota2/training/loop.py`, `src/infm_sota2/training/report.py`, `tests/test_training.py`, `tests/test_report.py`
- Ändern: diese vier, `src/infm_sota2/training/trainer.py`, `src/infm_sota2/cluster/train.py`, `tests/test_trainer.py`, `tests/test_ddp.py`, `old/README.md`

**Schnittstelle:**
- `evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float` nimmt immer alle Batches.
- `train(model, optimizer, train_loader, val_loader, device, epochs, eval_every, grad_clip, precision, checkpoint_path, checkpoint_metadata, log, eval_model) -> TrainResult`, also ohne `eval_batches`. `TrainConfig` verliert das Feld `eval_batches`.
- `ModelReport` hat kein `add_texts` und kein `losses` mehr.

- [ ] **Schritt 1:** Alten Stand der vier Dateien nach `old/` schreiben.
- [ ] **Schritt 2:** Tests anpassen. Es fallen weg: `test_evaluate_respects_max_batches` und `test_external_texts_appear_only_in_completions`. Jedes `evaluate(..., None)` verliert das letzte Argument, `eval_batches` fällt aus `run_train` und aus `CONFIG` in `test_trainer.py` und `test_ddp.py`. `test_losses_match_evaluate` prüft nur noch `report.test_loss("A") == evaluate(self.model, self.data.loader("test", 4), CPU)`. `test_completion_row_counts_copied_words` nimmt die Texte von `self.add(report, "B")`. Für jede Zeile gilt mit `copied = longest_copied_span(row["Text"].removeprefix(PROMPT), self.data.train)`: `row["Längste kopierte Folge"] == copied` und `row["Kopiert (Wörter)"] == len(copied.split())`.
- [ ] **Schritt 3:** Tests laufen lassen. Erwartet: FAIL, weil `evaluate` noch `max_batches` verlangt.
- [ ] **Schritt 4:** Code anpassen wie unter Schnittstelle, auch den Aufruf in `trainer.py` und `eval_batches=None` in `cluster/train.py`.
- [ ] **Schritt 5:** Checks. Erwartet: grün.
- [ ] **Schritt 6:** `old/README.md` fortschreiben, Commit `refactor: Optionen der alten Notebooks nach old/`.

### Task 5: `Environment` ohne Colab, ein Train-Loader, `fit` statt `Trainer`

**Dateien:**
- `git mv`: `src/infm_sota2/training/trainer.py` nach `old/src/infm_sota2/training/trainer.py`
- Alter Stand nach `old/`: `src/infm_sota2/env.py`, `src/infm_sota2/data/splits.py`, `tests/test_env.py`, `tests/test_splits.py`, `tests/test_trainer.py`
- Ändern: `src/infm_sota2/env.py`, `src/infm_sota2/data/splits.py`, `src/infm_sota2/training/__init__.py`, `src/infm_sota2/cluster/train.py`, `tests/helpers.py`, `tests/test_env.py`, `tests/test_splits.py`, `tests/test_trainer.py`, `tests/test_cluster_train.py`, `tests/test_ddp.py`, `notebooks/workbook_timo_cluster.ipynb`, `old/README.md`

**Schnittstelle:**
- `Environment(base_dir: Path, device: torch.device, wandb_mode: WandbMode, data_dir: Path, rank: int, world_size: int)`, frozen, mit den Properties `checkpoint_dir` (`base_dir / "checkpoints"`) und `gpu_name`. `detect(smoke_test: bool)` nimmt den Arbeitsordner als `base_dir`. Das Gerät ist die CPU ohne CUDA, sonst `torch.device("cuda", local_rank)`. `_torchrun_ranks() -> tuple[int, int, int]` gibt `(rank, local_rank, world_size)` zurück, ohne die Variablen `(0, 0, 1)`. Es fallen weg: `in_colab`, `run_dir`, `local_rank`, `launched_by_torchrun`, `login_wandb`, `check_versions`, `describe`, `version_mismatches` und die Colab-Konstanten.
- `TextSplits(train: str, val: str, test: str, datasets: dict[Split, BPEDataset])` mit `from_book` (Parameter wie bisher), `tokenizer`, `train_loader(batch_size: int, seed: int, rank: int, world_size: int) -> DataLoader` und `loader(split, batch_size)`. Der neue `train_loader` hat den Körper des bisherigen `distributed_train_loader`. Es fallen weg: `summary`, `split_summary` und der alte `train_loader`.
- In `src/infm_sota2/cluster/train.py`: `fit(env: Environment, model: nn.Module, data: TextSplits, settings: TrainSettings, seed: int, name: str, group: str, checkpoint_name: str | None, wandb_config: dict[str, object]) -> TrainResult`. Es macht dasselbe wie `Trainer.fit`, mit diesen Unterschieden: Der Loader ist immer `data.train_loader(settings.batch_size // env.world_size, seed, env.rank, env.world_size)`. Der Checkpoint geht auf Rang 0 nach `env.checkpoint_dir / checkpoint_name`, danach folgen `load_checkpoint` und `print("Checkpoint:", path)`, ohne Kopie. Lernrate und Epochen kommen aus `settings`. Die W&B-`config` ist `{**wandb_config, "world_size": ..., "batch_size_per_process": ...}`.
- `run()` ruft `fit(env, model, data, config.train, config.seed, name=f"{label}-{user}", group=config.group, checkpoint_name=..., wandb_config=config_dict(config) | {"user": user, "job_id": job_id})` auf. `infm_sota2.training` exportiert `Trainer` und `TrainConfig` nicht mehr.

- [ ] **Schritt 1:** Alten Stand der fünf Dateien nach `old/` schreiben.
- [ ] **Schritt 2:** Tests anpassen.
  - `tests/helpers.py`: `local_env` ohne `in_colab`, `run_dir`, `local_rank` und `launched_by_torchrun`.
  - `test_env.py`: `test_describe_names_device_and_torch` und `VersionMismatchTest` fallen weg. `test_detect_local_uses_working_directory` prüft `base_dir`, `data_dir` und `checkpoint_dir`. `test_detect_reads_torchrun_variables` erwartet `(env.rank, env.world_size) == (1, 2)`, `test_detect_without_torchrun_is_single_process` erwartet `(0, 1)`.
  - `test_splits.py`: `test_summary_lists_cleaning_counts` und `test_split_summary_names_steps_per_epoch` fallen weg. Alle Loader-Tests rufen `train_loader(batch_size, seed, rank, world_size)` auf.
  - `test_trainer.py` testet `fit` aus `infm_sota2.cluster.train` mit `SETTINGS = TrainSettings(batch_size=4, learning_rate=1e-2, epochs=1, weight_decay=0.1, grad_clip=1.0, eval_every=5, precision="fp32", compile_backend=None, fused_adamw=False)` und Seed 1, mit denselben sieben Tests. `test_fit_gives_same_result_as_train` nimmt `self.data.train_loader(4, seed=1, rank=0, world_size=1)`. `test_fit_keeps_best_checkpoint` prüft zusätzlich, dass `checkpoints/runs` nicht existiert. `test_fit_under_torchrun_uses_distributed_loader` heißt jetzt `test_fit_uses_train_loader_of_its_rank` und erwartet mit `mock.patch.object(TextSplits, "train_loader", autospec=True, side_effect=...)` den Aufruf `(self.data, 4, 1, 0, 1)`. `test_batch_size_must_divide_world_size` patcht `infm_sota2.cluster.train.train`.
  - `test_cluster_train.py`: `mock.patch("infm_sota2.cluster.train.fit", side_effect=...)` statt `mock.patch.object(Trainer, "fit", ...)`. Neu ist `test_record_keeps_its_keys`: Der Datensatz eines Smoke-Laufs hat genau die Schlüssel `user`, `label`, `group`, `config`, `environment`, `result`, `mfu`, `test_loss`, `perplexity`, `texts`, `copied`, `pretrained`, `peak_memory_gb`, `fit_seconds` und `error`, und `record["environment"]` enthält `world_size`, `gpu` und `git_commit`.
  - `test_ddp.py`: `SETTINGS` wie oben, aber `batch_size=8`. `train_once` ruft `fit(env, model, data, SETTINGS, 1, "ddp", "tests", "best.pth", {"test": True})` auf. Die Umgebungen entstehen mit `replace(local_env(...), rank=..., world_size=...)`.
- [ ] **Schritt 3:** Tests laufen lassen. Erwartet: FAIL, weil `fit` fehlt und `local_env` unbekannte Felder setzt.
- [ ] **Schritt 4:** Umsetzen wie unter Schnittstelle. Im Notebook entfernt `nb_replace.py` die Zeile `run_dir=base / "checkpoints" / "runs",` samt Einrückung und Zeilenumbruch aus `smoke_results`.
- [ ] **Schritt 5:** Checks. Erwartet: grün.
- [ ] **Schritt 6:** `uv run python <tmp>/reference_run.py . <tmp>/ref-task5.json` und mit `<tmp>/ref-1.json` vergleichen. Erwartet: gleich, oder in der Toleranz aus Task 1.
- [ ] **Schritt 7:** `old/README.md` fortschreiben, Commit `refactor: Environment ohne Colab, ein Train-Loader, fit statt Trainer`.

### Task 6: Ein Einstieg `python -m infm_sota2`

**Dateien:**
- Neu: `src/infm_sota2/__main__.py`, `tests/test_main.py`
- Ändern: `slurm/check.sbatch`, `slurm/train.sbatch`, `slurm/search.sbatch`, `slurm/setup.sh`, `configs/smoke-cpu.toml`, `tests/test_scripts.py`, `tests/test_mini_book.py`, `tests/test_cluster_check.py`

**Schnittstelle:**
- `COMMANDS: dict[str, Callable[[list[str] | None], None]]` ordnet `"check"`, `"benchmark"`, `"train"` und `"mini-book"` die `main` von `infm_sota2.cluster.check`, `infm_sota2.cluster.benchmark`, `infm_sota2.cluster.train` und `infm_sota2.data.mini_book` zu.
- `USAGE`: eine Zeile `Aufruf: python -m infm_sota2 {check,benchmark,train,mini-book} [optionen]`, aus `COMMANDS` gebaut.
- `main(argv: list[str] | None = None) -> None` liest `COMMANDS` beim Aufruf und ruft `COMMANDS[argv[0]](argv[1:])` auf. Ohne Befehl oder bei unbekanntem Befehl schreibt es `USAGE` nach stderr und wirft `SystemExit(2)`. Bei `-h` oder `--help` schreibt es `USAGE` nach stdout und kehrt zurück. Am Ende steht `if __name__ == "__main__": main()`.

- [ ] **Schritt 1:** Fehlschlagende Tests in `tests/test_main.py`:
  - `test_command_gets_remaining_arguments`: Mit `mock.patch.dict(COMMANDS, {"train": fake})` ruft `main(["train", "--config", "x.toml"])` genau einmal `fake(["--config", "x.toml"])` auf.
  - `test_every_command_is_listed`: `set(COMMANDS) == {"check", "benchmark", "train", "mini-book"}`, und jeder Name steht in `USAGE`.
  - `test_unknown_or_missing_command_exits_with_2`: `main(["nope"])` und `main([])` werfen `SystemExit` mit `code == 2`, stderr enthält `USAGE`.
  - `test_help_prints_usage`: `main(["-h"])` schreibt `USAGE` nach stdout und wirft nichts.
  - `test_python_m_runs_the_entry_point`: `subprocess.run([sys.executable, "-m", "infm_sota2"], capture_output=True, text=True)` endet mit Code 2, stderr enthält `USAGE`.
- [ ] **Schritt 2:** Laufen lassen. Erwartet: FAIL, weil `infm_sota2.__main__` fehlt.
- [ ] **Schritt 3:** `__main__.py` schreiben, Tests laufen lassen. Erwartet: grün.
- [ ] **Schritt 4:** Tests der Aufrufer anpassen:
  - `test_scripts.py`: Das nachgebaute `uv` erkennt `"$*" == "run python -m infm_sota2 mini-book "*` und führt `exec "$TEST_PYTHON" -m infm_sota2 mini-book "${@:6}"` aus. `MINI_BOOK_FILE` fällt weg, erwartet wird `"run python -m infm_sota2 mini-book data\n"`. Neu ist `test_setup_download_command_imports`, ohne `needs_bash`: Er nimmt aus der Zeile von `setup.sh` mit `download_weights` den Teil von `from infm_sota2` bis zum ersten `;` und führt ihn mit `exec` aus.
  - `test_mini_book.py`: `test_module_call_runs_without_warning` startet `[sys.executable, "-W", "error", "-m", "infm_sota2", "mini-book", <ordner>]`.
  - `test_cluster_check.py`: `CheckMainTest` startet `torchrun --standalone --nproc-per-node=2 -m infm_sota2 check --no-internet`.
- [ ] **Schritt 5:** Aufrufer umstellen: In `check.sbatch` steht `-m infm_sota2 check` und `python -m infm_sota2 benchmark`, in `train.sbatch` und `search.sbatch` `-m infm_sota2 train`. Der Kommentar in `train.sbatch` sagt, dass alles hinter dem Skript an `python -m infm_sota2 train` geht. `setup.sh` ruft `uv run python -m infm_sota2 mini-book "$DATA_DIR"` auf, und sein Befehl für die Gewichte importiert mit `from infm_sota2.model import GPT2_SMALL_WEIGHTS_SHA256, GPT2_SMALL_WEIGHTS_URL, download_weights`. Im Kommentar von `smoke-cpu.toml` steht `uv run python -m infm_sota2 train --config configs/smoke-cpu.toml`, die Zeile zu „Schritt 3“ fällt weg.
- [ ] **Schritt 6:** Checks und `bash -n` für jede Datei in `slurm/`. Erwartet: grün. Die übrigen bash-Tests laufen in der CI.
- [ ] **Schritt 7:** Commit `feat: ein Einstieg python -m infm_sota2 für alle Befehle`.

### Task 7: `data.py`

**Dateien:**
- Neu: `src/infm_sota2/data.py`, `tests/test_data.py`
- Entfernen, der Code zieht um: `src/infm_sota2/data/` (`__init__.py`, `cleaning.py`, `dataset.py`, `splits.py`, `mini_book.py`), `tests/test_cleaning.py`, `tests/test_splits.py`, `tests/test_mini_book.py`, `tests/test_dataloader.py`
- Ändern: `src/infm_sota2/__main__.py`, `src/infm_sota2/training/report.py`, `tests/helpers.py`, `notebooks/workbook_timo_cluster.ipynb`

**Schnittstelle:**
- Das Modul `infm_sota2.data` enthält in dieser Reihenfolge `CleaningStats`, `clean_book_text`, `split_text`, `BPEDataset`, `Split`, `TextSplits`, `MINI_BOOK`, `write_mini_book` und `main`, ohne `if __name__ == "__main__"`. Der Docstring sagt, was das Modul macht, und dass die Regeln der Bereinigung in `old/docs/lotr-training.md` stehen.
- `__main__.py` importiert `from infm_sota2 import data` und ordnet `"mini-book"` `data.main` zu.

- [ ] **Schritt 1:** `tests/test_data.py` aus den vier Testdateien zusammenstellen, alles aus `infm_sota2.data` importiert. `DistributedTrainLoaderTest` heißt jetzt `TrainLoaderTest`. Die vier alten Dateien entfernen und die Tests laufen lassen. Erwartet: FAIL beim Import von `MINI_BOOK` aus `infm_sota2.data`.
- [ ] **Schritt 2:** `data.py` aus den vier Modulen zusammensetzen, das Paket `data/` entfernen und die Importe in `__main__.py`, `training/report.py` und `tests/helpers.py` anpassen. Im Notebook ersetzt `nb_replace.py` `from infm_sota2.data.mini_book import write_mini_book` durch `from infm_sota2.data import write_mini_book`.
- [ ] **Schritt 3:** Checks. `grep -rn "infm_sota2\.data\." src tests scripts slurm notebooks` findet nichts.
- [ ] **Schritt 4:** Commit `refactor: data.py statt des Pakets data/`.

### Task 8: `model.py`

**Dateien:**
- Neu: `src/infm_sota2/model.py`; `tests/test_model.py` wird neu zusammengestellt
- Entfernen, der Code zieht um: `src/infm_sota2/model/`, `src/infm_sota2/transformer/`, `tests/test_build.py`, `tests/test_gpt_v2.py`, `tests/test_pretrained.py`, `tests/test_attention.py`
- Ändern: `src/infm_sota2/cluster/config.py`, `src/infm_sota2/cluster/train.py`

**Schnittstelle:**
- `infm_sota2.model` enthält in dieser Reihenfolge: `GPT2_SMALL_CONFIG`, `GPT2_MEDIUM_CONFIG` und `GPT2_LARGE_CONFIG`; `LayerNorm`, `FeedForward`, `MultiHeadAttention`, `TransformerBlock` und `GPTModel`; `CausalSelfAttention`, `TransformerBlockV2`, `GPTModelV2` und `convert_v1_state_dict`; `MODEL_CLASSES = {"GPTModel": GPTModel, "GPTModelV2": GPTModelV2}`; `build_model`; `GPT2_SMALL_WEIGHTS_URL`, `GPT2_SMALL_WEIGHTS_SHA256`, `sha256_of`, `download_weights` und `load_pretrained`. Klassen und Attribute bleiben, wie sie sind.
- `cluster/config.py` prüft `model_class` gegen `tuple(MODEL_CLASSES)`, die eigene Konstante fällt weg. `cluster/train.py` nimmt `MODEL_CLASSES` aus `infm_sota2.model`.

- [ ] **Schritt 1:** `tests/test_model.py` aus `test_model.py`, `test_build.py`, `test_gpt_v2.py`, `test_pretrained.py` und `test_attention.py` zusammenstellen, alles aus `infm_sota2.model` importiert, auch `FeedForward` und `MultiHeadAttention`. Neu ist `test_model_classes_name_both_models`: `MODEL_CLASSES == {"GPTModel": GPTModel, "GPTModelV2": GPTModelV2}`. Alte Dateien entfernen, Tests laufen lassen. Erwartet: FAIL, weil `MODEL_CLASSES` und `MultiHeadAttention` in `infm_sota2.model` fehlen.
- [ ] **Schritt 2:** `model.py` zusammensetzen, die Pakete entfernen, `config.py` und `train.py` anpassen.
- [ ] **Schritt 3:** Checks. `grep -rn "infm_sota2\.transformer\|infm_sota2\.model\." src tests scripts slurm notebooks` findet nichts.
- [ ] **Schritt 4:** Commit `refactor: model.py statt der Pakete model/ und transformer/`.

### Task 9: `env.py` startet die Prozessgruppe und kennt `has_bf16`

**Dateien:**
- Ändern: `src/infm_sota2/env.py`, `src/infm_sota2/benchmark.py`, `src/infm_sota2/cluster/check.py`, `src/infm_sota2/cluster/benchmark.py`, `src/infm_sota2/cluster/train.py`, `tests/test_env.py`, `tests/test_benchmark.py`
- Entfernen, der Code zieht um: `src/infm_sota2/cluster/distributed.py`, `tests/test_cluster_distributed.py`

**Schnittstelle:**
- `infm_sota2.env` bekommt `has_bf16(device: torch.device) -> bool`, `init_distributed(env: Environment) -> None` und `destroy_distributed(env: Environment) -> None`, unverändert übernommen. Der Docstring des Moduls nennt die Prozessgruppe.

- [ ] **Schritt 1:** `test_env.py` bekommt `DistributedTest` aus `test_cluster_distributed.py` und `test_cpu_has_bf16` aus `test_benchmark.py`, beide mit Importen aus `infm_sota2.env`. Die alte Datei entfernen, Tests laufen lassen. Erwartet: FAIL beim Import.
- [ ] **Schritt 2:** Funktionen verschieben, Importe anpassen.
- [ ] **Schritt 3:** Checks, Commit `refactor: env.py startet auch die Prozessgruppe`.

### Task 10: `loop.py` und `report.py`

**Dateien:**
- `git mv`: `src/infm_sota2/training/loop.py` nach `src/infm_sota2/loop.py`, `tests/test_training.py` nach `tests/test_loop.py`
- Neu: `src/infm_sota2/report.py`; `tests/test_report.py` wird neu zusammengestellt
- Entfernen, der Code zieht um: `src/infm_sota2/training/` (`__init__.py`, `generate.py`, `analysis.py`, `report.py`), `tests/test_generate.py`, `tests/test_analysis.py`
- Ändern: `src/infm_sota2/benchmark.py`, `src/infm_sota2/cluster/config.py`, `src/infm_sota2/cluster/train.py`, `tests/test_trainer.py`, `tests/test_ddp.py`, `tests/test_cluster_train.py`, `tests/test_benchmark.py`

**Schnittstelle:**
- `infm_sota2.loop` enthält `TrainPrecision`, `batch_loss`, `evaluate`, `TrainResult`, `train`, `save_checkpoint` und `load_checkpoint`.
- `infm_sota2.report` enthält in dieser Reihenfolge `generate`, `complete`, `longest_copied_span` und `ModelReport`.

- [ ] **Schritt 1:** `tests/test_report.py` aus `test_report.py`, `test_generate.py` und `test_analysis.py` zusammenstellen, `test_loop.py` mit `git mv` anlegen, die Importe aller Tests auf `infm_sota2.loop` und `infm_sota2.report` umstellen. Tests laufen lassen. Erwartet: FAIL beim Import.
- [ ] **Schritt 2:** Module anlegen, das Paket entfernen, Importe anpassen.
- [ ] **Schritt 3:** Checks. `grep -rn "infm_sota2\.training" src tests scripts slurm notebooks` findet nichts. Commit `refactor: loop.py und report.py statt des Pakets training/`.

### Task 11: Paket `cluster/` auflösen

**Dateien:**
- `git mv`: `src/infm_sota2/cluster/config.py`, `results.py`, `check.py` und `train.py` nach `src/infm_sota2/`; `tests/test_cluster_config.py` nach `tests/test_config.py`, `tests/test_cluster_results.py` nach `tests/test_results.py`, `tests/test_cluster_check.py` nach `tests/test_check.py`
- Zusammenlegen: `src/infm_sota2/cluster/benchmark.py` in `src/infm_sota2/benchmark.py`, `tests/test_cluster_benchmark.py` in `tests/test_benchmark.py`, `tests/test_cluster_train.py` und `tests/test_trainer.py` zu `tests/test_train.py`
- Entfernen: `src/infm_sota2/cluster/__init__.py`
- Ändern: `src/infm_sota2/__main__.py`, `tests/test_ddp.py`, `tests/test_notebooks.py`, `notebooks/workbook_timo_cluster.ipynb`

**Schnittstelle:**
- `infm_sota2.config` ohne `config_dict` und `run_label`. `train.py` schreibt dafür `asdict(config)` und `f"{config.name}-gpus{env.world_size}"`. `REPO_DIR = Path(__file__).resolve().parents[2]`.
- `infm_sota2.benchmark` enthält erst den bisherigen Inhalt von `benchmark.py`, dann den von `cluster/benchmark.py` samt `main`.
- `check.py`, `benchmark.py` und `train.py` haben kein `if __name__ == "__main__"` mehr. `__main__.py` importiert `from infm_sota2 import benchmark, check, data, train`.
- mock-Pfade in den Tests: `infm_sota2.train.download_weights`, `infm_sota2.train.load_pretrained`, `infm_sota2.train.fit` und `infm_sota2.train.train`.

- [ ] **Schritt 1:** Tests umstellen und zusammenlegen wie oben. `test_run_label_names_gpus` fällt weg, weil `test_smoke_run_without_wandb_writes_result` `record["label"] == "smoke-cpu-gpus1"` prüft. `test_config_dict_is_ready_for_json` prüft `json.dumps(asdict(...))`. Neu in `test_notebooks.py` ist `test_notebook_imports_exist`: Aus den Code-Zellen des Notebooks, ohne Zeilen, die mit `%` oder `!` beginnen, sammelt `ast` jedes `from infm_sota2... import ...`. Es gibt mindestens eins, und für jedes findet `importlib.import_module` das Modul und `hasattr` jeden Namen. Tests laufen lassen. Erwartet: FAIL.
- [ ] **Schritt 2:** Module verschieben und zusammenlegen, die Importe in `src/` anpassen. Im Notebook ersetzt `nb_replace.py` `infm_sota2.cluster.config` durch `infm_sota2.config`, `infm_sota2.cluster.results` durch `infm_sota2.results`, `from infm_sota2.cluster import benchmark` durch `from infm_sota2 import benchmark` und `infm_sota2.cluster.train` durch `infm_sota2.train`.
- [ ] **Schritt 3:** Checks. `grep -rn "infm_sota2\.cluster" src tests scripts slurm configs notebooks` findet nichts, und `ls src/infm_sota2` zeigt genau die 13 Dateien aus dem Entwurf.
- [ ] **Schritt 4:** Referenzlauf wie in Task 5 schreiben und mit `<tmp>/ref-1.json` vergleichen. Erwartet: gleich, oder in der Toleranz aus Task 1.
- [ ] **Schritt 5:** Commit `refactor: Paket cluster/ in einzelne Module aufgelöst`.

### Task 12: README, `docs/code.md` und Verweise

**Dateien:**
- Neu: `docs/code.md`
- Ändern: `README.md`, `docs/cluster-training.md` (ein Satz unter der Zeile „Stand:“), `docs/dachs.md` (der Verweis auf den CUDA-Index nennt den Abschnitt „Versionen“ der README), `old/README.md` (letzte Fassung)

**Inhalt:** wie im Entwurf unter „Doku“. Die README hat höchstens 130 Zeilen, `docs/code.md` höchstens 120. Kurze, schlichte Sätze ohne Gedankenstriche, Fachbegriffe wie in den übrigen Dokumenten.

- `docs/code.md`: was der Code macht; die Module als Tabelle in der Reihenfolge des Ablaufs, je ein Satz; der Ablauf eines Laufs von der TOML-Datei über `config.load_config` und `train.run` (`data.TextSplits.from_book`, `model.build_model`, `train.fit` mit `loop.train`, `report.ModelReport`, `results.write_result`) bis zur JSON-Datei, dazu was unter `torchrun` nur Rang 0 tut; die Befehle aus dem Entwurf und die `sbatch`-Befehle mit Verweis auf `docs/dachs.md`; wie man etwas ändert: ein neuer Lauf ist eine neue TOML-Datei, eine neue Einstellung braucht Datenklasse und Schlüsseltabelle in `config.py`, einen Eintrag in jeder Datei in `configs/` und einen Test in `test_config.py`; vor dem Commit die Checks, nach Änderungen an `config.py`, `results.py` oder `train.py` zusätzlich `scripts/smoke_notebook.py` für das Notebook.
- `README.md`: Titel und zwei Sätze; Aufbau als Tabelle mit `src/infm_sota2/`, `configs/`, `slurm/`, `notebooks/`, `tests/`, `scripts/`, `docs/`, `results/` und `old/`; Einrichten mit uv, `uv sync`, Git-Hook, Zeilenenden, Notebook-Kernel und W&B-Login; Benutzen mit dem lokalen Smoke-Lauf `uv run python -m infm_sota2 train --config configs/smoke-cpu.toml`, dem Cluster über `docs/dachs.md` und der Auswertung im Notebook; Checks; Versionen mit Pins, Kopplung an Colab, dem Block für den CUDA-Index und dem Hinzufügen von Paketen; W&B kurz mit Projekt, Login, `WANDB_MODE` und `wandb sync`; Arbeitsweise mit Branches, Merge-Commit und Colab für das Notebook; Regeln.

- [ ] **Schritt 1:** Texte schreiben.
- [ ] **Schritt 2:** Jede Datei, jedes Modul und jeder Befehl in README und `docs/code.md` existiert. `uv run python -m infm_sota2 -h` nennt dieselben Befehle wie die Doku.
- [ ] **Schritt 3:** Jeder relative Link in `README.md`, `docs/*.md` und `old/README.md` zeigt auf eine vorhandene Datei.
- [ ] **Schritt 4:** Checks, Commit `docs: README und kurze Doku zum Code`.

### Task 13: Abschluss

- [ ] **Schritt 1:** Checks und alle Tests. Erwartet: grün, übersprungen sind nur die Tests für bash und `torchrun`.
- [ ] **Schritt 2:** Befehle in einem leeren Ordner `<tmp>/run`, mit `SOTA2_DATA_DIR=<tmp>/run/data` und `WANDB_MODE=disabled`, jeweils `uv run --project <repo> python -m infm_sota2 ...`: `mini-book <tmp>/run/data`, `train --config <repo>/configs/smoke-cpu.toml`, `benchmark --smoke` und `check --no-internet`. Erwartet: kein Fehler, die JSON-Dateien liegen unter `<tmp>/run/results/`, und unter `<repo>/results/` ist nichts Neues.
- [ ] **Schritt 3:** `uv run --with nbclient python scripts/smoke_notebook.py notebooks/workbook_timo_cluster.ipynb <tmp>/cluster-after.txt`. Erwartet: läuft ohne Fehler. Der `diff` mit `<tmp>/cluster-before.txt` zeigt nur andere Zahlen und Pfade, kein Abschnitt fehlt.
- [ ] **Schritt 4:** `grep -rn "infm_sota2\.\(cluster\|training\|transformer\|data\.\|model\.\)" src tests scripts slurm configs docs notebooks README.md` findet nur Treffer in `docs/plans/` und `docs/cluster-training.md`.
- [ ] **Schritt 5:** `src/` und `tests/` haben zusammen weniger als 6.124 Zeilen.
- [ ] **Schritt 6:** Referenzlauf ein letztes Mal, Vergleich mit `<tmp>/ref-1.json`.
- [ ] **Schritt 7:** Weiter mit superpowers:finishing-a-development-branch. Push von `chore/aufraeumen` und der Pull Request nach `week/06` erst nach Freigabe durch Timo. Danach muss die CI grün sein, auch mit den bash-Tests. Auf dem Cluster startet Timo `sbatch slurm/check.sbatch` und den Smoke-Lauf auf einem und auf zwei Knoten, bevor die echten Läufe beginnen.
