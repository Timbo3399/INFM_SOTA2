# Woche 6, Training auf dem Cluster: Implementierungsplan

> **Für Agenten:** Pflicht-Skill zum Umsetzen: superpowers:subagent-driven-development oder superpowers:executing-plans. Die Schritte sind Checkboxen (`- [ ]`).

**Ziel:** Jede Person aus der Gruppe kann sich auf DACHS einrichten, einen Smoke-Lauf ohne Buchtext starten und GPT-2 mit den Optimierungen aus Woche 5 per DDP auf mehreren Knoten trainieren. Jeder Lauf hinterlässt eine JSON-Datei mit Ergebnis und Einstellungen.

**Architektur:** Weg A. `train()`, `Trainer`, `TextSplits` und `Environment` werden erweitert, das LOTR-Notebook rechnet danach dieselben Zahlen. Neu ist das Paket `infm_sota2.cluster` mit den Einstiegspunkten `check`, `benchmark` und `train`, gestartet über `torchrun` aus Slurm-Skripten in `slurm/`. Die Einstellungen stehen in TOML-Dateien in `configs/`.

**Technik:** Python 3.13 (Syntax bis 3.12), torch 2.14.1, tiktoken 0.14.0, wandb 0.28.1, `tomllib`, unittest, Bash, Slurm.

**Spezifikation:** [`docs/cluster-training.md`](../../../docs/cluster-training.md)

## Globale Vorgaben

- Keine neuen Laufzeit-Abhängigkeiten. pandas, matplotlib und nbformat nur in Notebooks und Skripten außerhalb von `src/`.
- Keine Default-Werte für Hyperparameter in Signaturen.
- `train()` bleibt frei von W&B.
- Tests mit `unittest` auf der CPU, in Sekunden, auch unter Windows. Tests, die Bash oder mehrere Prozesse brauchen, überspringen sich dort, wo das nicht geht.
- ruff (`E, F, I, UP, B, SIM`), `ruff format`, pyright `standard` und alle Tests grün vor jedem Commit.
- Kein Buchtext in Repo, Tests, Logs oder Notebook-Ausgaben. Tests nutzen das Mini-Buch.
- Notebooks nur per `nbformat`-Skript ändern, nie das JSON von Hand.
- Fehler laut melden: `ValueError` bei falschen Eingaben, kein breites `except`. Ausnahme: `cluster.train` fängt `FloatingPointError` und `torch.OutOfMemoryError`, schreibt den Fehler in die JSON-Datei und wirft ihn erneut.
- In Skripten und Konfigurationen stehen keine Namen, E-Mail-Adressen oder Pfade einer Person.
- Git: Autor „Timo <timo.lahres@hotmail.com>“ ist im Klon eingestellt. Commit-Messages haben ein Präfix und eine deutsche Beschreibung, ohne Co-Autor- und Session-Zeilen. Nach jeder Aufgabe committen und pushen, mit bis zu 4 Wiederholungen bei Netzfehlern.
- Branches: Task 1 bis 7 auf `feature/cluster-setup`, Task 8 bis 13 auf `feature/cluster-training`, abgezweigt vom Stand nach Task 7.
- Checks: `uv run ruff check`, `uv run ruff format --check`, `uv run pyright`, `uv run python -m unittest discover -s tests -p 'test*.py'`.

## Review-Fokus

1. **Daten fehlen**, weil jemand im falschen Ordner steht, den Buchtext nicht kopiert hat oder `SOTA2_DATA_DIR` falsch setzt. Erwartet: `FileNotFoundError` mit dem Pfad und dem Hinweis auf `slurm/setup.sh` und `SOTA2_DATA_DIR`. Test in Task 7.
2. **Job außerhalb des Klons abgeschickt.** Erwartet: Das Jobskript bricht vor `srun` mit einer Meldung ab. Test in Task 7.
3. **Ein Lauf wird wiederholt** oder zwei Jobs heißen gleich. Erwartet: Der Dateiname enthält die Job-ID, nichts wird überschrieben. Test in Task 6.
4. **Dieselbe Kombination der Suche liegt doppelt vor.** Erwartet: Die Tabelle bricht mit beiden Dateinamen ab, statt still eine zu wählen. Test in Task 6.
5. **Person ohne W&B-Zugang.** Erwartet: Mit `WANDB_MODE=disabled` läuft der ganze Smoke-Lauf samt JSON-Datei. Test in Task 7 (Schritt 2) und Task 8 (Modus aus der Umgebung).

---

## Schritt 1: Einstieg und Job 1 (`feature/cluster-setup`)

### Task 1: Ausgangswerte des LOTR-Notebooks festhalten

Muss vor jeder Änderung an `src/` laufen.

**Dateien:**
- Neu: `scripts/smoke_notebook.py`
- Ändern: `pyproject.toml` (`scripts/**/*.py` in `[tool.ruff] include`, `scripts` in `[tool.pyright] include`)

**Schnittstelle:**
- `main(argv: list[str] | None = None) -> None` mit den Argumenten `notebook` (Pfad) und `output` (Pfad). Setzt in der Zelle `SMOKE_TEST = False` den Wert auf `True`, führt das Notebook mit `nbclient` im Ordner des Notebooks aus und schreibt alle `stdout`-Ausgaben in `output`. Speichert das Notebook nicht.
- Aufruf: `uv run --with nbclient python scripts/smoke_notebook.py <notebook> <ausgabe>`.

- [ ] **Schritt 1:** Skript schreiben. `nbclient` wird nur innerhalb von `main` importiert, mit `# type: ignore` für pyright, weil es nicht in den Abhängigkeiten steht.
- [ ] **Schritt 2:** `uv run --with nbclient python scripts/smoke_notebook.py notebooks/workbook_timo_lotr.ipynb <scratchpad>/lotr-before.txt` ausführen.
  Erwartet: Das Notebook läuft durch. Die Zeilen der Bereinigung stimmen mit der gespeicherten Ausgabe vom 12.11. überein (bereinigt 2,522,191 Zeichen, 926 Kopfzeilen, 8,749 Absätze), roh ist es ein Zeichen mehr, der Zeilenumbruch am Dateiende.
- [ ] **Schritt 3:** Checks ausführen. Erwartet: alles grün.
- [ ] **Schritt 4:** Commit `chore: Skript für den Smoke-Test von Notebooks`, pushen.

### Task 2: Mini-Buch im Paket

**Dateien:**
- Neu: `src/infm_sota2/data/mini_book.py`
- Ändern: `src/infm_sota2/data/__init__.py` (Export `MINI_BOOK`, `write_mini_book`), `tests/helpers.py` (importiert beides von dort, die eigene Definition fällt weg)
- Test: `tests/test_mini_book.py`

**Schnittstelle:**
- `MINI_BOOK: dict[str, str]` und `write_mini_book(directory: Path) -> str` mit genau dem Inhalt und Verhalten wie heute in `tests/helpers.py` (Rückgabe: Dateiname `mini-book.txt`).
- `main(argv: list[str] | None = None) -> None`: `python -m infm_sota2.data.mini_book <ordner>` legt den Ordner bei Bedarf an, schreibt das Mini-Buch und gibt den Pfad aus.

- [ ] **Schritt 1:** Fehlschlagende Tests schreiben.
  - `test_writes_book_that_cleans_to_201_paragraphs`: Nach `write_mini_book(tmp)` liefert `clean_book_text` mit den Markern aus `MINI_BOOK` `stats.paragraphs == 201` (`PROLOGUE` und 200 Sätze).
  - `test_cli_creates_directory_and_book`: `main([str(tmp / "neu")])` legt `tmp / "neu" / "mini-book.txt"` an.
  - `test_helpers_use_package_version`: `tests.helpers.write_mini_book is infm_sota2.data.mini_book.write_mini_book`.
- [ ] **Schritt 2:** `uv run python -m unittest tests.test_mini_book -v`. Erwartet: FAIL (`No module named 'infm_sota2.data.mini_book'`).
- [ ] **Schritt 3:** Umsetzen.
- [ ] **Schritt 4:** Tests erneut und alle Checks. Erwartet: alles grün, die bisherigen Tests mit dem Mini-Buch unverändert grün.
- [ ] **Schritt 5:** Commit `feat: Mini-Buch im Paket, mit Aufruf über die Kommandozeile`, pushen.

### Task 3: Anleitung und Einrichtung

**Dateien:**
- Neu: `docs/dachs.md`, `slurm/common.sh`, `slurm/setup.sh`, `tests/test_scripts.py`
- Ändern: `.gitignore` (`logs/`), `README.md` (Abschnitt „Auf dem Cluster“ verweist oben auf `docs/dachs.md`; Projektstruktur um `slurm/`, `scripts/`, `configs/`, `results/` ergänzen)

**Schnittstelle:**
- `slurm/common.sh` wird von allen Jobskripten mit `source` geladen. Es setzt `UV_NO_DEV=1`, `TIKTOKEN_CACHE_DIR="$HOME/.cache/tiktoken"`, aktiviert `.venv` und stellt `MASTER_ADDR` (erster Knoten aus `scontrol show hostnames`) und `MASTER_PORT` (`20000 + SLURM_JOB_ID % 20000`) bereit.
- `slurm/setup.sh` läuft im Klon auf dem Login-Knoten, mit `set -euo pipefail`, und lässt sich wiederholen:
  1. uv fehlt oder ist älter als 0.13: installieren mit dem offiziellen Skript nach `~/.local/bin`.
  2. `UV_NO_DEV=1 uv sync --no-dev --locked`.
  3. tiktoken-Cache füllen, `mkdir -p logs`.
  4. Mini-Buch schreiben: `uv run python -m infm_sota2.data.mini_book "${SOTA2_DATA_DIR:-data}"`.
  5. Prüfen und melden: Buchtext `The-lord-of-the-rings.txt` und `gpt2-small-124M.pth` im Datenordner, W&B-Login (`~/.netrc` mit `api.wandb.ai`). Für jedes fehlende Stück steht der Befehl da, der es behebt. Ohne W&B der Hinweis auf `export WANDB_MODE=disabled`.
  6. Schluss: „bereit für den Smoke-Lauf“ oder „bereit für alle Läufe“.
- `docs/dachs.md` für jede Person, mit den Abschnitten aus Teil 1 der Spezifikation: Konto, Einloggen (PowerShell, Fingerabdruck, `~/.ssh/config`, SSH-Schlüssel), Einrichten (Klon, Branch, `setup.sh`, Buchtext per `scp`, gemeinsamer Workspace mit `ws_allocate` und `ws_share` als Möglichkeit), Slurm kennenlernen, unsere Jobs (Tabelle der Skripte und Konfigurationen), Ergebnisse holen, typische Fehler, bwUniCluster später.

- [ ] **Schritt 1:** Fehlschlagende Tests schreiben, mit `@unittest.skipIf(os.name == "nt" or shutil.which("bash") is None, ...)` wo Bash nötig ist.
  - `test_shell_scripts_have_valid_syntax`: `bash -n` gibt für jede Datei `slurm/*.sh` und `slurm/*.sbatch` den Rückgabewert 0.
  - `test_scripts_name_nobody`: Kein Skript in `slurm/` und keine Datei in `configs/` enthält `/home/`, ein `@` oder eine Zeichenfolge, die zu `\bof_[a-z]` passt.
  - `test_setup_creates_logs_folder`: `setup.sh` enthält `mkdir -p logs`.
- [ ] **Schritt 2:** `uv run python -m unittest tests.test_scripts -v`. Erwartet: FAIL, `slurm/` fehlt.
- [ ] **Schritt 3:** Skripte, Anleitung, `.gitignore` und README schreiben. Die Anleitung mit dem Skill humanizer überarbeiten.
- [ ] **Schritt 4:** Tests und alle Checks. Erwartet: alles grün.
- [ ] **Schritt 5:** Commit `docs: Anleitung für DACHS und Einrichtung für alle`, pushen.

### Task 4: Job 1, Umgebung prüfen und Benchmark

**Dateien:**
- Ändern: `src/infm_sota2/benchmark.py`
- Neu: `src/infm_sota2/cluster/__init__.py`, `src/infm_sota2/cluster/check.py`, `src/infm_sota2/cluster/benchmark.py`, `slurm/check.sbatch`
- Test: `tests/test_benchmark.py` (ergänzen), `tests/test_cluster_check.py`, `tests/test_cluster_benchmark.py`

**Schnittstelle:**
- In `benchmark.py`: `slide20_variants(mixed: Precision, compile_backend: str) -> list[Variant]` mit den sechs Varianten aus `workbook_timo_attention_v2.ipynb`, gleiche Namen und Reihenfolge. Das Notebook bleibt unverändert.
- `cluster/check.py`:
  - `collect_info(env: Environment, check_internet: bool) -> dict[str, object]` mit den Schlüsseln `host`, `slurm_job_id`, `torch`, `cuda_runtime`, `cuda_available`, `gpu`, `compute_capability`, `driver` (aus `nvidia-smi --query-gpu=driver_version`, sonst `None`), `bf16_supported`, `flash_attention` (Ergebnis eines kleinen `scaled_dot_product_attention`-Aufrufs unter `sdpa_kernel(SDPBackend.FLASH_ATTENTION)`, sonst die Fehlermeldung), `nccl`, `c_compiler`, `internet` (Host zu erreichbar ja oder nein für `api.wandb.ai` und `pypi.org`, Timeout 5 s), `tiktoken_cache`, `data_dir` und `data_files` (vorhandene Eingabedateien).
  - `format_info(info: dict[str, object]) -> str`, `main(argv: list[str] | None = None) -> None` mit `--no-internet`. Mit mehreren Prozessen (`WORLD_SIZE` > 1, Prozessgruppe über `torch.distributed`) gibt jeder Rang eine Zeile mit Rechner und GPU aus, Rang 0 zusätzlich das Ergebnis eines `all_reduce` der Ränge.
- `cluster/benchmark.py`:
  - `SCAN_BATCH_SIZES = (8, 16, 32, 64, 128, 256, 512)`.
  - `batch_scan(model_config: dict[str, Any], seq_len: int, batch_sizes: Sequence[int], variant: Variant, warmup_steps: int, measure_steps: int, device: torch.device, seed: int) -> list[tuple[int, BenchmarkResult]]` hört nach dem ersten Ergebnis mit `error` auf.
  - `main(argv: list[str] | None = None) -> None`: (a) `slide20_variants` mit GPT-2 Small, Batch 4, 1024 Tokens, 3 und 10 Schritten, Seed 123, (b) `batch_scan` mit Variante 6 bei 256 Tokens und `context_length` 256. Gibt beide Tabellen aus und schreibt sie als JSON nach `results/week06/<benutzer>/benchmark/<gpu>-job<id>.json` (Task 6 liefert die Pfadfunktion, bis dahin baut `main` den Pfad selbst nach demselben Muster). `--smoke` nimmt das winzige Modell aus dem Notebook von Woche 5, Batchgrößen `(2, 4)` und `eager`.
- `slurm/check.sbatch`: Partition `gpu1`, 1 Knoten, `--gres=gpu:1`, 16 CPUs, 64G, 30 Minuten, Log `logs/%x-%j.out`. Prüft, dass `$SLURM_SUBMIT_DIR/pyproject.toml` existiert, lädt `common.sh`, ruft `nvidia-smi` auf, startet `cluster.check` über `srun torchrun` auf allen Knoten und den Benchmark nur bei einem Knoten.

- [ ] **Schritt 1:** Fehlschlagende Tests schreiben.
  - `test_slide20_variants_match_week5`: Die Namen sind `["1 Ausgangsstand", "2 16 Bit mit autocast", "3 16 Bit, ganzes Modell", "4 torch.nn ohne Flash", "5 torch.nn mit Flash", "6 torch.compile"]`. Nr. 2, 4, 5 und 6 haben die übergebene Präzision, Nr. 3 hat `bf16`, Nr. 4 den Kernel `math`, nur Nr. 6 das übergebene Backend.
  - `test_batch_scan_stops_after_first_error`: Mit einem Testmodell, das ab Batch 4 `torch.OutOfMemoryError` wirft, liefert der Scan über `(2, 4, 8)` genau zwei Einträge, der zweite mit `error == "zu wenig Speicher"`.
  - `test_benchmark_smoke_writes_json`: `main(["--smoke"])` in einem temporären Arbeitsordner schreibt eine JSON-Datei mit den Schlüsseln `variants` und `batch_scan`, alle sechs Varianten ohne Fehler.
  - `test_collect_info_on_cpu`: `collect_info(env, check_internet=False)` ergibt `cuda_available is False`, `gpu is None` und `internet` gleich `None`.
  - `test_format_info_lists_every_key`: Jeder Schlüssel steht in der Ausgabe.
- [ ] **Schritt 2:** Tests ausführen. Erwartet: FAIL.
- [ ] **Schritt 3:** Umsetzen.
- [ ] **Schritt 4:** Tests und alle Checks. Erwartet: alles grün.
- [ ] **Schritt 5:** Commit `feat: Job 1 prüft die Umgebung und misst den Benchmark auf dem Cluster`, pushen. Danach kann Timo sich einloggen, `setup.sh` ausführen und `sbatch slurm/check.sbatch` starten.

## Schritt 2: Smoke-Lauf und Job 2 (`feature/cluster-setup`)

### Task 5: Konfiguration aus TOML

**Dateien:**
- Neu: `src/infm_sota2/cluster/config.py`, `configs/smoke-cpu.toml`, `configs/smoke.toml`, `configs/lotr-small-week4.toml`
- Test: `tests/test_cluster_config.py`

**Schnittstelle:**
- Datenklassen, alle `@dataclass(frozen=True)`:
  - `DataSettings(file: str, start_line: str, end_line: str, book_title: str, train_ratio: float, val_ratio: float, seq_len: int, max_chars: dict[str, int] | None)`; `max_chars` kommt aus der optionalen Tabelle `[data.max_chars]`.
  - `TrainSettings(batch_size: int, learning_rate: float, epochs: int, weight_decay: float, grad_clip: float, eval_every: int, precision: str, compile_backend: str | None, fused_adamw: bool)`; `compile_backend = "none"` in TOML wird `None`. `precision` ist `fp32`, `fp16-mixed` oder `bf16-mixed`.
  - `GenerationSettings(prompt: str, samples: int, max_new_tokens: int, temperature: float, top_k: int)`.
  - `SearchSettings(learning_rates: tuple[float, ...], batch_sizes: tuple[int, ...], epochs: int)`.
  - `RunConfig(name: str, group: str, seed: int, model_class: str, evaluate_pretrained: bool, save_checkpoint: bool, report: bool, data: DataSettings, model: dict[str, int | float | bool], train: TrainSettings, generation: GenerationSettings, search: SearchSettings | None)`; `model_class` ist `GPTModel` oder `GPTModelV2`, `model` hat genau die Schlüssel von `GPT2_SMALL_CONFIG`.
- `load_config(path: Path) -> RunConfig`: Fehlende und unbekannte Schlüssel, falsche Typen (ein `bool` zählt nicht als `int`) und unbekannte Werte für `precision` oder `model_class` ergeben `ValueError` mit dem Namen des Schlüssels.
- `with_grid(config: RunConfig, index: int) -> RunConfig`: Kombination Nr. `index` zeilenweise, also `learning_rates[index // len(batch_sizes)]` und `batch_sizes[index % len(batch_sizes)]`. Setzt Lernrate, Batchgröße und `search.epochs` und hängt `-lr{lr:g}-bs{bs}` an den Namen.
- `run_label(config: RunConfig, world_size: int) -> str` gibt `f"{config.name}-gpus{world_size}"` zurück.
- `config_dict(config: RunConfig) -> dict[str, object]` für W&B und JSON.
- Konfigurationen: `smoke-cpu.toml` (Mini-Buch, `seq_len` 32, Modell aus dem Smoke-Test von Woche 5 mit `context_length` 32, Batch 4, Lernrate 1e-2, 2 Epochen, Prüfung alle 5 Schritte, `bf16-mixed`, Backend `eager`, ohne fused, ohne GPT-2 original, mit Report). `smoke.toml` wie `smoke-cpu.toml`, aber `inductor` und fused AdamW für die GPU. `lotr-small-week4.toml` mit den Werten von Nr. 2 aus der Spezifikation, `evaluate_pretrained = true`.

- [ ] **Schritt 1:** Fehlschlagende Tests schreiben.
  - `test_loads_smoke_cpu_config`: `name == "smoke-cpu"`, `train.compile_backend == "eager"`, `train.fused_adamw is False`, `model["n_layers"] == 2`.
  - `test_every_config_file_loads`: Jede Datei in `configs/*.toml` lädt.
  - `test_missing_key_is_named`: Ohne `[train] epochs` gibt es `ValueError` mit `epochs` in der Meldung.
  - `test_unknown_key_is_named`: Ein zusätzliches `epoch` in `[train]` ergibt `ValueError` mit `epoch`.
  - `test_wrong_type_is_an_error`: `batch_size = true` und `learning_rate = "4e-4"` ergeben `ValueError`.
  - `test_compile_backend_none_means_no_compile`.
  - `test_grid_is_row_major`: Mit Lernraten `[1e-4, 3e-4, 1e-3, 3e-3]` und Batchgrößen `[8, 32, 128]` ergibt Index 0 (1e-4, 8), 1 (1e-4, 32), 3 (3e-4, 8) und 11 (3e-3, 128); `train.epochs == search.epochs`; Name endet auf `-lr0.0003-bs8` für Index 3.
  - `test_grid_index_out_of_range`: Index 12 und ein Index bei fehlender `[search]`-Tabelle ergeben `ValueError`.
- [ ] **Schritt 2:** Tests ausführen. Erwartet: FAIL.
- [ ] **Schritt 3:** Umsetzen.
- [ ] **Schritt 4:** Tests und alle Checks. Erwartet: alles grün.
- [ ] **Schritt 5:** Commit `feat: Einstellungen der Läufe als TOML-Dateien`, pushen.

### Task 6: Ergebnisdateien und Tabellen

**Dateien:**
- Neu: `src/infm_sota2/cluster/results.py`
- Ändern: `src/infm_sota2/cluster/benchmark.py` (nutzt `result_path` und `write_result`)
- Test: `tests/test_cluster_results.py`

**Schnittstelle:**
- `result_path(base_dir: Path, user: str, group: str, label: str, job_id: str | None) -> Path` ergibt `base_dir / "results" / "week06" / user / group / f"{label}-job{job_id}.json"`, ohne Job-ID `-local.json`.
- `write_result(path: Path, record: dict[str, object]) -> None`: legt Ordner an, schreibt JSON mit `indent=2` und `ensure_ascii=False`, wirft `FileExistsError`, wenn die Datei schon existiert.
- `load_results(directory: Path) -> list[dict[str, object]]`: alle JSON-Dateien darunter, sortiert, jede mit dem Zusatzschlüssel `file`.
- `git_commit(repo_dir: Path) -> str`: `git rev-parse HEAD`, Fehler bricht ab.
- `scaling_table(records: list[dict[str, object]]) -> list[dict[str, object]]`: eine Zeile pro `environment.world_size` mit `tokens_per_second`, `speedup` und `efficiency`; ohne Lauf mit einer GPU `ValueError`; Läufe mit Fehler bleiben als Zeile mit Fehlertext.
- `search_table(records: list[dict[str, object]], learning_rates: Sequence[float], batch_sizes: Sequence[int]) -> list[dict[str, object]]`: eine Zeile pro Kombination mit `best_val_loss` und `status` (`"ok"`, Fehlertext oder `"fehlt"`); zwei Dateien für dieselbe Kombination ergeben `ValueError` mit beiden Dateinamen.
- Felder einer Ergebnisdatei, die die Tabellen lesen: `config.train.learning_rate`, `config.train.batch_size`, `environment.world_size`, `result.tokens_per_second`, `result.best_val_loss`, `error`.

- [ ] **Schritt 1:** Fehlschlagende Tests schreiben.
  - `test_result_path_layout` mit und ohne Job-ID.
  - `test_write_refuses_to_overwrite`: Der zweite `write_result` auf denselben Pfad wirft `FileExistsError`.
  - `test_roundtrip_keeps_record`: `load_results` liefert den geschriebenen Datensatz plus `file`.
  - `test_scaling_table_by_hand`: Tokens/s 100, 150 und 200 bei 1, 2 und 4 GPUs ergeben Speedup 1, 1,5 und 2 und Effizienz 1, 0,75 und 0,5.
  - `test_scaling_table_needs_single_gpu_run`.
  - `test_search_table_marks_failed_and_missing`: Gitter 2×2, drei Datensätze, einer davon mit Fehler. Ergebnis: vier Zeilen mit `"ok"`, `"ok"`, dem Fehlertext und `"fehlt"`.
  - `test_duplicate_combination_names_both_files`.
  - `test_git_commit_is_hex`: 40 Hex-Zeichen für den Repo-Ordner.
- [ ] **Schritt 2:** Tests ausführen. Erwartet: FAIL.
- [ ] **Schritt 3:** Umsetzen, `cluster/benchmark.py` auf `result_path` und `write_result` umstellen.
- [ ] **Schritt 4:** Tests und alle Checks. Erwartet: alles grün.
- [ ] **Schritt 5:** Commit `feat: Ergebnisse jedes Laufs als JSON, Tabellen für Suche und Skalierung`, pushen.

### Task 7: Einstiegspunkt `cluster.train` mit dem heutigen Trainer

**Dateien:**
- Neu: `src/infm_sota2/cluster/train.py`, `slurm/train.sbatch`
- Ändern: `tests/helpers.py` (`local_env(base_dir: Path) -> Environment` mit CPU und W&B `disabled`)
- Test: `tests/test_cluster_train.py`, `tests/test_scripts.py` (ergänzen)

**Schnittstelle:**
- `run(config: RunConfig, env: Environment, user: str, job_id: str | None, repo_dir: Path) -> dict[str, object]` gibt den geschriebenen Datensatz zurück. Ablauf:
  1. Datei `env.data_dir / config.data.file` fehlt: `FileNotFoundError` mit Pfad und Hinweis auf `slurm/setup.sh` und `SOTA2_DATA_DIR`.
  2. `TextSplits.from_book(env.data_dir, seq_len=..., **Datenfelder ohne seq_len)`.
  3. Mit `report` oder `evaluate_pretrained`: `ModelReport(data, env.device, batch_size=config.train.batch_size, seed=config.seed, **generation)`.
  4. Mit `evaluate_pretrained`: `download_weights(GPT2_SMALL_WEIGHTS_URL, GPT2_SMALL_WEIGHTS_SHA256, env.data_dir / "gpt2-small-124M.pth")`, `load_pretrained(GPTModel, GPT2_SMALL_CONFIG, ...)`, `report.add("GPT-2 original", ..., 1024)`.
  5. `build_model(...)`, `Trainer(env, wandb_config=config_dict(config) | {"user": user, "job_id": job_id})`, `trainer.fit(...)` mit `name=f"{label}-{user}"`, `group=config.group`, Checkpoint `f"{label}.pth"` nur mit `save_checkpoint`. Die Wanduhrzeit von `fit` wird gemessen.
  6. Mit `report`: `report.add(label, model, config.model["context_length"])`.
  7. Datensatz mit `user`, `label`, `group`, `config`, `environment` (Rechner, `SLURM_JOB_NODELIST`, Job-ID, `world_size`, GPU, Versionen, Git-Commit), `result` (Felder von `TrainResult`), `test_loss`, `perplexity`, `texts`, `copied`, `pretrained`, `peak_memory_gb`, `fit_seconds` und `error`. Schreiben mit `result_path` und `write_result`.
  8. `FloatingPointError` und `torch.OutOfMemoryError`: Datensatz mit `error` und ohne Loss schreiben, dann erneut werfen.
- Schritt-2-Fassung: `train` übersetzt `precision` auf `TrainConfig(use_amp=...)` und bricht bei `bf16-mixed`, gesetztem `compile_backend`, `fused_adamw` oder `WORLD_SIZE` > 1 mit `ValueError` „kommt mit Schritt 3“ ab.
- `main(argv: list[str] | None = None) -> None` mit `--config` und `--grid-index`. Benutzer aus `getpass.getuser()`, Job-ID aus `SLURM_JOB_ID`, `Environment.detect(smoke_test=False)`, Repo-Ordner ist der Arbeitsordner.
- `slurm/train.sbatch` nach der Skizze der Spezifikation, mit derselben Prüfung des Ordners wie `check.sbatch`. Ab zwei Knoten setzt es `NCCL_DEBUG=INFO`.

- [ ] **Schritt 1:** Fehlschlagende Tests schreiben.
  - `test_smoke_run_without_wandb_writes_result`: Mini-Buch in `tmp / "data"`, `run(load_config(Path("configs/smoke-cpu.toml")) mit Präzision fp32 und ohne compile, local_env(tmp), "tester", None, Path.cwd())`. Erwartet: Datei unter `tmp/results/week06/tester/week6-smoke/`, `error is None`, `result.best_val_loss < math.log(50257) - 1`, drei Texte.
  - `test_missing_book_names_path_and_hint`: ohne Mini-Buch `FileNotFoundError`, die Meldung enthält den Pfad und `SOTA2_DATA_DIR`.
  - `test_step3_features_are_rejected_for_now`: `smoke-cpu.toml` unverändert (bf16, eager) ergibt `ValueError`.
  - `test_failed_training_writes_error_record`: `Trainer.fit` per `unittest.mock.patch` wirft `FloatingPointError`. Erwartet: `run` wirft weiter, die Datei hat `error` und `test_loss is None`.
  - In `tests/test_scripts.py`: `test_train_job_refuses_foreign_directory`: `bash slurm/train.sbatch` mit `SLURM_SUBMIT_DIR` auf einen leeren temporären Ordner endet mit Rückgabewert ungleich 0 und einer Meldung, die „Klon“ enthält, ohne `srun` aufzurufen.
- [ ] **Schritt 2:** Tests ausführen. Erwartet: FAIL.
- [ ] **Schritt 3:** Umsetzen.
- [ ] **Schritt 4:** `uv run python -m infm_sota2.cluster.train --config configs/smoke-cpu.toml` mit einer lokalen Kopie der Konfiguration ohne Schritt-3-Optionen aus einem temporären Arbeitsordner ausführen. Erwartet: Ausgabe mit Test-Loss und Texten, JSON-Datei geschrieben.
- [ ] **Schritt 5:** Tests und alle Checks. Erwartet: alles grün.
- [ ] **Schritt 6:** Commit `feat: Einstiegspunkt für Trainingsläufe auf dem Cluster`, pushen. Danach kann Timo den Smoke-Lauf auf einem Knoten und Nr. 2 starten. Timo fragen, ob der Pull Request nach `week/06` erstellt werden soll.

## Schritt 3: Umbau nach Weg A (`feature/cluster-training`)

### Task 8: `Environment` für mehrere Prozesse

**Dateien:**
- Ändern: `src/infm_sota2/env.py`, `tests/helpers.py` (`local_env`), `tests/test_trainer.py` (baut die Umgebung über `local_env`)
- Neu: `src/infm_sota2/cluster/distributed.py`
- Test: `tests/test_env.py` (ergänzen)

**Schnittstelle:**
- `WandbMode = Literal["online", "offline", "disabled"]`.
- `Environment` bekommt die Felder `data_dir: Path` (vorher Property), `rank: int`, `local_rank: int`, `world_size: int`, `launched_by_torchrun: bool`. `describe()` bleibt unverändert.
- `detect(smoke_test)`:
  - `torchrun` erkennt es an `RANK`, `LOCAL_RANK` und `WORLD_SIZE`, sonst gilt Rang 0 von 1.
  - Gerät unter `torchrun` mit CUDA: `cuda:<local_rank>`, sonst wie bisher `cuda` bzw. `cpu`.
  - `smoke_test` ergibt `disabled`. Sonst gilt `WANDB_MODE`, falls gesetzt (ein anderer Wert als die drei erlaubten ergibt `ValueError`), sonst `online`.
  - `data_dir` ist `SOTA2_DATA_DIR`, falls gesetzt, sonst `base_dir / "data"`. Das gilt lokal und in Colab.
- `cluster/distributed.py`: `init_distributed(env: Environment) -> None` setzt bei CUDA das Gerät und startet die Prozessgruppe nur bei `world_size > 1` (NCCL auf CUDA, Gloo auf der CPU). `destroy_distributed(env: Environment) -> None` beendet sie.

- [ ] **Schritt 1:** Fehlschlagende Tests schreiben, Umgebungsvariablen mit `unittest.mock.patch.dict(os.environ, ..., clear=False)` setzen und fremde vorher entfernen.
  - `test_detect_reads_torchrun_variables`: `RANK=1`, `LOCAL_RANK=0`, `WORLD_SIZE=2` ergeben Rang 1 von 2 und `launched_by_torchrun is True`.
  - `test_detect_without_torchrun_is_single_process`.
  - `test_wandb_mode_from_environment`: `offline` gilt, mit `smoke_test=True` bleibt es `disabled`.
  - `test_invalid_wandb_mode_is_an_error`.
  - `test_data_dir_from_environment` und `test_data_dir_defaults_to_working_directory`.
  - Die bisherigen Tests in `test_env.py` bleiben unverändert grün.
- [ ] **Schritt 2:** Tests ausführen. Erwartet: FAIL.
- [ ] **Schritt 3:** Umsetzen.
- [ ] **Schritt 4:** Tests und alle Checks. Erwartet: alles grün.
- [ ] **Schritt 5:** Commit `feat: Environment kennt Ränge, W&B-Modus und Datenordner aus der Umgebung`, pushen.

### Task 9: Gleiche Batches bei jeder Knotenzahl

**Dateien:**
- Ändern: `src/infm_sota2/data/splits.py`
- Test: `tests/test_splits.py` (ergänzen)

**Schnittstelle:**
- `TextSplits.distributed_train_loader(batch_size: int, seed: int, rank: int, world_size: int) -> DataLoader`: `batch_size` meint Fenster pro Prozess. `DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed, drop_last=True)` und DataLoader mit `drop_last=True`. Braucht keine Prozessgruppe.

- [ ] **Schritt 1:** Fehlschlagende Tests schreiben.
  - `test_same_windows_per_step_for_any_world_size`: Mini-Buch mit `seq_len` 8, Gesamt-Batch 8. Für p = 1, 2 und 4 ergibt in jedem Schritt die Vereinigung der Eingaben aller Ränge (als sortierte Zeilen) dasselbe wie bei p = 1.
  - `test_set_epoch_changes_order`: `sampler.set_epoch(1)` ergibt einen anderen ersten Batch als Epoche 0.
- [ ] **Schritt 2:** Tests ausführen. Erwartet: FAIL.
- [ ] **Schritt 3:** Umsetzen.
- [ ] **Schritt 4:** Tests und alle Checks. Erwartet: alles grün.
- [ ] **Schritt 5:** Commit `feat: Train-Loader für mehrere Prozesse mit DistributedSampler`, pushen.

### Task 10: `train()` mit Präzision, Durchsatz und Verlauf

**Dateien:**
- Ändern: `src/infm_sota2/training/loop.py`, `src/infm_sota2/training/__init__.py` (Export `TrainPrecision`), `src/infm_sota2/training/trainer.py` (ruft `train` vorerst mit `precision="fp16-mixed" if config.use_amp else "fp32"` und `eval_model=model` auf)
- Test: `tests/test_training.py`

**Schnittstelle:**
- `TrainPrecision = Literal["fp32", "fp16-mixed", "bf16-mixed"]`.
- `train(model, optimizer, train_loader, val_loader, device, epochs, eval_every, eval_batches, grad_clip, precision: TrainPrecision, checkpoint_path, checkpoint_metadata, log, eval_model: nn.Module) -> TrainResult`: `precision` steht an der Stelle von `use_amp`, `eval_model` ist neu am Ende.
  - `fp16-mixed` ohne CUDA: `ValueError` mit „CUDA“. Unbekannte Präzision: `ValueError`.
  - autocast mit float16 und GradScaler bei `fp16-mixed`, mit bfloat16 ohne Scaler bei `bf16-mixed`, ohne autocast bei `fp32`. `fp32` und `fp16-mixed` rechnen genau wie bisher `use_amp=False` und `use_amp=True`.
  - `evaluate()` und `save_checkpoint()` nutzen `eval_model`.
  - Zu Beginn jeder Epoche `set_epoch(epoch)`, wenn `train_loader.sampler` ein `DistributedSampler` ist.
  - Zeitmessung mit `time.perf_counter()` von Prüfung zu Prüfung. Vor dem Stoppen `torch.cuda.synchronize(device)` auf CUDA, die Prüfung selbst zählt nicht mit. `log` bekommt zusätzlich `throughput/tokens_per_second` (Tokens dieses Prozesses).
- `TrainResult` bekommt `tokens_per_second: float | None` (Median aller Intervalle außer dem ersten, `None` bei weniger als zwei) und `val_losses: tuple[tuple[int, float], ...]` (Schritt und Validierungs-Loss jeder Prüfung).

- [ ] **Schritt 1:** `run_train` in `tests/test_training.py` auf `precision="fp32"` und `eval_model=model` umstellen. Dann fehlschlagende Tests schreiben.
  - `test_bf16_mixed_trains_on_cpu`: Der beste Validierungs-Loss liegt mehr als 1,0 unter dem Wert vorher (wie `test_training_lowers_validation_loss`).
  - `test_fp16_mixed_requires_cuda` (ersetzt `test_amp_requires_cuda`) und `test_unknown_precision_is_an_error`.
  - `test_checkpoint_from_compiled_model_loads_into_plain_model`: Training mit `torch.compile(model, backend="eager")` und `eval_model=model`; der Checkpoint lädt strikt in `tiny_model(seed=999)`, kein Schlüssel beginnt mit `_orig_mod.`.
  - `test_distributed_sampler_gets_each_epoch`: Ein `DistributedSampler` (`num_replicas=1`, `rank=0`), dessen `set_epoch` die Aufrufe mitschreibt, sieht bei 3 Epochen `[0, 1, 2]`.
  - `test_tokens_per_second_needs_two_intervals`: Mit nur der Prüfung am Epochenende `None`, mit `eval_every=1` eine Zahl über 0.
  - `test_val_losses_match_logged_losses`: Schritte und Werte von `val_losses` gleichen den geloggten.
  - `test_logs_train_and_val_loss` erwartet zusätzlich den Schlüssel `throughput/tokens_per_second`.
- [ ] **Schritt 2:** Tests ausführen. Erwartet: FAIL.
- [ ] **Schritt 3:** Umsetzen.
- [ ] **Schritt 4:** Tests und alle Checks. `test_fit_gives_same_result_as_train` vergleicht ab jetzt alle Felder außer `tokens_per_second`. Erwartet: alles grün.
- [ ] **Schritt 5:** Commit `feat: train() mit bf16, Durchsatz, Verlauf und Modell für die Auswertung`, pushen.

### Task 11: `Trainer.fit` mit DDP, `compile` und fused AdamW

**Dateien:**
- Ändern: `src/infm_sota2/training/trainer.py`, `notebooks/workbook_timo_lotr.ipynb` (per `nbformat`-Skript im Scratchpad), `docs/lotr-training.md` (Abschnitt „Code“: `precision` statt `use_amp`)
- Test: `tests/test_trainer.py`, neu `tests/test_ddp.py`

**Schnittstelle:**
- `TrainConfig(weight_decay, batch_size, eval_every, eval_batches, grad_clip, precision: TrainPrecision, compile_backend: str | None, fused_adamw: bool, seed)`; `batch_size` ist die Gesamt-Batchgröße.
- `fit` (Signatur unverändert):
  - `batch_size % env.world_size != 0`: `ValueError` vor allem anderen. `fused_adamw` ohne CUDA: `ValueError`.
  - Unter `torchrun` `data.distributed_train_loader(batch_size // world_size, seed, rank, world_size)`, sonst `data.train_loader(batch_size, seed)` wie bisher.
  - Ab zwei Prozessen `DistributedDataParallel(model, device_ids=[env.device.index] bei CUDA, sonst None, broadcast_buffers=False)`, danach `torch.compile(..., backend=compile_backend)`, falls gesetzt.
  - `AdamW(model.parameters(), lr, weight_decay, fused=True)` mit `fused_adamw`, sonst ohne `fused` wie bisher, damit das Notebook bitgleich bleibt.
  - `train(step_model, ..., precision=config.precision, eval_model=model)`.
  - Nur Rang 0: W&B im Modus `env.wandb_mode` (sonst `disabled`), Checkpoint, Ausgaben, Zurückladen und Kopie des Checkpoints. Die W&B-Konfiguration bekommt `world_size` und die Batchgröße pro Prozess.
  - Geloggte `tokens_seen` und `throughput/tokens_per_second` und das zurückgegebene `TrainResult` (`tokens_seen`, `tokens_per_second`) sind mit `world_size` hochgerechnet.
- LOTR-Notebook: `USE_AMP = ...` wird `PRECISION = "fp16-mixed" if env.device.type == "cuda" else "fp32"`, `CONFIG["environment"]` behält den Schlüssel `"mixed_precision"` mit `PRECISION != "fp32"`, `TrainConfig(..., precision=PRECISION, compile_backend=None, fused_adamw=False, ...)`. Keine Ausgabe ändert sich.

- [ ] **Schritt 1:** `CONFIG` in `tests/test_trainer.py` auf die neuen Felder umstellen. Fehlschlagende Tests schreiben.
  - `test_fit_under_torchrun_uses_distributed_loader`: Mit `launched_by_torchrun=True` und `world_size=1` ruft `fit` `distributed_train_loader(4, 1, 0, 1)` auf (per `patch.object(..., wraps=...)`).
  - `test_batch_size_must_divide_world_size`: `world_size=3`, Batch 4: `ValueError` vor dem Training.
  - `test_fused_adamw_needs_cuda`.
  - `test_compile_and_bf16_on_cpu`: `compile_backend="eager"` und `bf16-mixed` liefern einen endlichen besten Validierungs-Loss.
  - `tests/test_ddp.py`, übersprungen, wenn `torch.distributed.is_gloo_available()` falsch ist:
    - `test_two_processes_match_one_process`: `torch.multiprocessing.spawn` startet 2 Prozesse mit Gloo (`init_method` als `file://` in einem temporären Ordner), jeder mit `world_size=2` und `launched_by_torchrun=True`. Sie trainieren `tiny_model(seed=1, vocab_size=50257)` 1 Epoche auf dem Mini-Buch mit Gesamt-Batch 8, Lernrate 1e-2 und `fp32`. Ein Prozess mit `world_size=1` und `launched_by_torchrun=True` trainiert dasselbe. Erwartet: `torch.allclose` für alle Gewichte mit `atol=1e-5`.
    - `test_only_rank_zero_writes_checkpoint`: Nach dem Lauf mit 2 Prozessen und `checkpoint_name="best.pth"` liegt genau eine Datei im Checkpoint-Ordner.
- [ ] **Schritt 2:** Tests ausführen. Erwartet: FAIL.
- [ ] **Schritt 3:** `fit` umsetzen und das Notebook per `nbformat`-Skript ändern.
- [ ] **Schritt 4:** `uv run --with nbclient python scripts/smoke_notebook.py notebooks/workbook_timo_lotr.ipynb <scratchpad>/lotr-after.txt` und `diff <scratchpad>/lotr-before.txt <scratchpad>/lotr-after.txt`. Erwartet: keine Unterschiede.
- [ ] **Schritt 5:** Tests und alle Checks. Erwartet: alles grün.
- [ ] **Schritt 6:** Commit `feat: Trainer mit DDP, torch.compile und fused AdamW, LOTR-Notebook rechnet unverändert`, pushen.

### Task 12: Volle Fassung von `cluster.train`, Konfigurationen und Suche

**Dateien:**
- Ändern: `src/infm_sota2/cluster/train.py`, `src/infm_sota2/model/pretrained.py`, `src/infm_sota2/model/__init__.py`
- Neu: `configs/lotr-small.toml`, `configs/lotr-small-search.toml`, `configs/lotr-small-scaling.toml`, `configs/lotr-small-main.toml`, `configs/lotr-medium.toml`, `configs/lotr-large.toml`, `slurm/search.sbatch`
- Test: `tests/test_cluster_train.py`, `tests/test_cluster_config.py`, `tests/test_scripts.py`, `tests/test_pretrained.py`

**Schnittstelle:**
- `GPT2_MEDIUM_CONFIG` (24 Schichten, 1024 Dimensionen, 16 Köpfe) und `GPT2_LARGE_CONFIG` (36, 1280, 20), sonst wie `GPT2_SMALL_CONFIG`.
- `TrainSettings.precision` bekommt den Typ `TrainPrecision` aus Task 10, `load_config` prüft den Wert wie bisher.
- `train`: Die Schritt-2-Sperren fallen weg. `main` ruft vor `run` `init_distributed` und danach `destroy_distributed` auf. `TrainConfig` mit `precision`, `compile_backend` und `fused_adamw` aus der Konfiguration. Nur Rang 0 misst, erzeugt Texte und schreibt die Datei, die anderen Ränge geben `{}` zurück. Neu im Datensatz: `result.val_losses`, `mfu` (über `flops_per_token` und `peak_tflops` aus `benchmark.py`, `None` ohne bekannte GPU) und `environment.world_size`.
- Konfigurationen mit den Werten aus Teil 3 der Spezifikation:
  - `lotr-small.toml`: Nr. 3, Gruppe `week6-optimized`.
  - `lotr-small-search.toml`: Nr. 4, Gruppe `week6-search`, ohne Checkpoint und Report, mit dem Gitter.
  - `lotr-small-scaling.toml`: Nr. 5, Gruppe `week6-scaling`, Batch 32, 2 Epochen, Prüfung alle 10 Schritte, ohne Checkpoint und Report.
  - `lotr-small-main.toml`: Nr. 6, Gruppe `week6-main`, 20 Epochen.
  - `lotr-medium.toml` und `lotr-large.toml`: Nr. 7 und 8.
  - Lernrate und Batchgröße von Nr. 5 bis 8 stehen bis zur Auswertung der Suche auf 4e-4 und 8, mit dem Kommentar `# nach der Suche eintragen`.
- `slurm/search.sbatch`: `--array=0-11%4`, Log `logs/%x-%A_%a.out`, `torchrun --standalone --nproc-per-node=1 -m infm_sota2.cluster.train --config configs/lotr-small-search.toml --grid-index "$SLURM_ARRAY_TASK_ID"`.

- [ ] **Schritt 1:** Fehlschlagende Tests schreiben.
  - `test_smoke_run_with_bf16_and_compile`: `smoke-cpu.toml` unverändert. Erwartet: `result.val_losses` nicht leer, der letzte Wert kleiner als der erste, `error is None`.
  - `test_out_of_memory_writes_error_record`: wie der Test für `FloatingPointError` aus Task 7, mit `torch.OutOfMemoryError`.
  - `test_grid_run_uses_search_settings`: Eine Testkonfiguration mit `[search]` und `grid_index=1` schreibt eine Datei, deren Name `-lr` und `-bs` enthält, mit Epochen aus `[search]`.
  - `test_two_process_smoke_run_writes_one_file` in `tests/test_ddp.py`: `run` in 2 Gloo-Prozessen mit `smoke-cpu.toml` schreibt genau eine Ergebnisdatei.
  - `test_model_configs_match_constants`: Die Modelle in `lotr-medium.toml` und `lotr-large.toml` gleichen `GPT2_MEDIUM_CONFIG` und `GPT2_LARGE_CONFIG` bis auf `context_length` 256.
  - `test_search_array_matches_grid`: Die Array-Grenze in `search.sbatch` ist die Zahl der Kombinationen aus `lotr-small-search.toml` minus 1.
- [ ] **Schritt 2:** Tests ausführen. Erwartet: FAIL.
- [ ] **Schritt 3:** Umsetzen.
- [ ] **Schritt 4:** Lokal mit zwei Prozessen: `uv run torchrun --standalone --nproc-per-node=2 -m infm_sota2.cluster.train --config configs/smoke-cpu.toml` in einem temporären Klon-Ordner mit Mini-Buch. Erwartet: beide Ränge enden ohne Fehler, genau eine JSON-Datei.
- [ ] **Schritt 5:** Tests und alle Checks. Erwartet: alles grün.
- [ ] **Schritt 6:** Commit `feat: Läufe mit DDP auf mehreren Knoten, Suche als Job-Array`, pushen. Danach startet Timo den Smoke-Lauf auf 2 Knoten, dann Nr. 3 bis 8.

## Schritt 4: Auswertung (`feature/cluster-training`)

### Task 13: Notebook für die Vorlesung und Doku

**Dateien:**
- Neu: `notebooks/workbook_timo_cluster.ipynb` (per `nbformat`-Skript im Scratchpad)
- Ändern: `tests/test_notebooks.py` (`TIMO_NOTEBOOKS` ergänzen), `README.md` (Projektstruktur mit `infm_sota2/cluster/`), `docs/cluster-training.md` (Stand: umgesetzt, Ergebnisse folgen)

**Schnittstelle:**
- Erste Zelle mit Colab-Link auf `blob/main`, eine Setup-Zelle mit `BRANCH = "main"` wie in den anderen Notebooks von Timo.
- Das Notebook liest `results/week06/` mit `load_results` und zeigt pro Abschnitt Tabelle oder Diagramm: Job 1, Benchmark neben den A100-Zahlen aus `docs/attention-v2.md`, Nr. 2 gegen Nr. 3, `search_table`, `scaling_table` als Diagramm mit idealer Geraden, Nr. 6 bis 8 mit Test-Loss, Verlauf und Texten, Fazit zum Ausfüllen. Gibt es noch keine Dateien, sagt jede Zelle das, statt abzubrechen. Diagramme nach dem Skill dataviz.

- [ ] **Schritt 1:** `"workbook_timo_cluster.ipynb"` in `TIMO_NOTEBOOKS` eintragen, `uv run python -m unittest tests.test_notebooks`. Erwartet: FAIL, die Datei fehlt.
- [ ] **Schritt 2:** Notebook per `nbformat`-Skript erzeugen.
- [ ] **Schritt 3:** `uv run --with nbclient python scripts/smoke_notebook.py notebooks/workbook_timo_cluster.ipynb <scratchpad>/cluster-smoke.txt`. Erwartet: läuft durch, auch ohne Ergebnisdateien.
- [ ] **Schritt 4:** README und Spezifikation nachziehen, Texte mit dem Skill humanizer überarbeiten.
- [ ] **Schritt 5:** Alle Checks. Erwartet: alles grün.
- [ ] **Schritt 6:** Commit `feat: Notebook für die Ergebnisse von Woche 6`, pushen. Timo fragen, ob der Pull Request nach `week/06` erstellt werden soll.

## Nach den Läufen

Sobald die Läufe auf dem Cluster fertig sind und die JSON-Dateien im Repo liegen: beste Kombination der Suche in `lotr-small-scaling.toml`, `lotr-small-main.toml`, `lotr-medium.toml` und `lotr-large.toml` eintragen (vor Nr. 5 bis 8), Notebook in Colab oder lokal ausführen, Ergebnisse und Fazit in `docs/cluster-training.md` und ins Notebook schreiben.
