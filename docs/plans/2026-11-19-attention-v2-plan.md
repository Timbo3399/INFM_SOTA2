# Notebook-Klassen und Attention V2: Implementierungsplan

> **Für Agenten:** Pflicht-Skill zum Umsetzen: superpowers:subagent-driven-development (empfohlen) oder superpowers:executing-plans. Die Schritte sind Checkboxen (`- [ ]`).

**Ziel:** Der Code aus `workbook_timo_lotr.ipynb` liegt in Klassen unter `src/infm_sota2/`, ohne dass sich eine Zahl ändert. Danach misst ein neues Notebook die Schritte aus Woche 5 mit `GPTModelV2` und einer Benchmark-Klasse.

**Architektur:** Bausteine (`Environment`, `TextSplits`, `Trainer`, `ModelReport`) mit einer Zeile pro Notebook-Schritt. `train()` bleibt unverändert und frei von W&B. `GPTModelV2` steht neben `GPTModel`, und der Benchmark misst beide.

**Technik:** Python 3.13 (Syntax bis 3.12), torch 2.14.1, tiktoken 0.14.0, wandb 0.28.1, unittest, nbformat und nbclient für die Notebooks.

**Spezifikation:** [`docs/attention-v2.md`](../attention-v2.md)

## Globale Vorgaben

- Keine neuen Laufzeit-Abhängigkeiten. pandas und matplotlib nur im Notebook, `src/` importiert sie nicht.
- Keine Default-Werte für Hyperparameter in Signaturen.
- `train()`, `evaluate()`, `batch_loss()`, `generate()` und `complete()` bleiben unverändert.
- Nicht anfassen: `GPTModel`, `transformer/*`, `create_dataloader`, `BPEDataset`, `Embeddings`, alle Notebooks außer `workbook_timo_lotr.ipynb` und dem neuen `workbook_timo_attention_v2.ipynb`.
- Tests mit `unittest` auf der CPU, in Sekunden, auch unter Windows. Gemeinsame Test-Bausteine stehen in `tests/helpers.py`.
- ruff (`E, F, I, UP, B, SIM`), `ruff format`, pyright `standard` und alle Tests grün. Der Pre-commit-Hook prüft das.
- Kein Buchtext in Tests, Repo oder Notebook-Ausgaben. Tests nutzen selbst geschriebene Mini-Texte.
- Notebooks nur per `nbformat`-Skript ändern, nie das JSON von Hand.
- Fehler laut melden: `ValueError` bei falschen Eingaben, kein breites `except`.
- Branches: Teil 1 auf `feature/notebook-classes` von `origin/week/05`, Teil 2 auf `feature/attention-v2` vom Stand nach Teil 1.
- Vor dem ersten Commit fragt der Umsetzer Timo. Danach gibt es einen Commit pro Aufgabe. Commit-Messages haben ein Präfix und eine deutsche Beschreibung, z. B. `feat: …`, und keine Co-Autor-Zeile.

## Review-Fokus

1. Colab installiert das Paket aus einem gelöschten Branch. Das ist mit `feature/lotr-training` passiert, `BRANCH` muss `"main"` sein. Test in Task 6 und 9.
2. Der Benchmark bekommt eine Sequenzlänge über der Kontextlänge des Modells und soll sofort mit `ValueError` abbrechen, nicht mitten im Lauf. Test in Task 8.
3. Die OpenAI-Gewichte gibt es mit `qkv_bias=True`, Raschkas eigene Läufe nutzen `False`. Die Umrechnung muss beides können. Test in Task 7.
4. Eine Variante bekommt zu wenig Speicher. Die nächste soll trotzdem laufen. Test in Task 8.
5. `Trainer.fit` ohne Checkpoint darf nichts in den Checkpoint-Ordner schreiben. Test in Task 4.

---

## Teil 1: Notebook-Code in Klassen

### Task 1: `Environment`

**Dateien:**
- Neu: `src/infm_sota2/env.py`
- Test: `tests/test_env.py`

**Schnittstelle:**
- Liefert:
  - `WANDB_ENTITY = "hochschule-offenburg"`, `WANDB_PROJECT = "SOTA_2"`.
  - `version_mismatches(requirements: list[str], installed: Callable[[str], str]) -> list[str]`.
  - `Environment` als `@dataclass(frozen=True)` mit `in_colab: bool`, `base_dir: Path`, `run_dir: Path`, `device: torch.device`, `wandb_mode: str`.
    - Properties: `data_dir` (`base_dir / "data"`), `checkpoint_dir` (`base_dir / "checkpoints"`), `gpu_name: str | None`.
    - Methoden: `detect(cls, smoke_test: bool) -> Environment` als Klassenmethode, `login_wandb() -> None`, `check_versions() -> list[str]`, `describe() -> dict[str, object]`.

- [ ] **Schritt 1:** Branch anlegen.
  `git fetch origin`, dann `git switch -c feature/notebook-classes origin/week/05`.
  Die ungetrackten Dateien `docs/attention-v2.md` und dieser Plan kommen mit.
- [ ] **Schritt 2:** Fehlschlagende Tests schreiben, in `tests/test_env.py`.
  - `test_detect_local_uses_working_directory`: `detect(smoke_test=True)` ergibt `in_colab is False`, `base_dir == Path.cwd()`, `run_dir == Path.cwd() / "checkpoints" / "runs"`, `data_dir == Path.cwd() / "data"` und `checkpoint_dir == Path.cwd() / "checkpoints"`.
  - `test_smoke_test_disables_wandb`: `detect(True).wandb_mode == "disabled"`, `detect(False).wandb_mode == "online"`.
  - `test_describe_names_device_and_torch`: Die Schlüssel sind `{"device", "gpu", "torch", "colab"}`, `["device"] == env.device.type` und `["torch"] == torch.__version__`.
  - `test_version_mismatch_is_reported`: `version_mismatches(["torch==2.14.1", "wandb==0.28.1"], {"torch": "2.14.1+cu130", "wandb": "0.27.0"}.__getitem__)` ergibt `["Achtung: wandb 0.27.0 installiert, pyproject.toml erwartet 0.28.1"]`.
  - `test_matching_versions_give_no_message`: Ein lokales Suffix wie `+cpu` zählt nicht als Abweichung.
- [ ] **Schritt 3:** `uv run python -m unittest tests.test_env` ausführen. Erwartet: FAIL (`No module named 'infm_sota2.env'`).
- [ ] **Schritt 4:** `env.py` umsetzen. Die Werte stammen aus den alten Zellen 3, 4, 6 und 7 des LOTR-Notebooks.
  - Colab erkennt `"google.colab" in sys.modules`. Dann `drive.mount("/content/drive")`, `base_dir = /content/drive/MyDrive/INFM_SOTA2` und `run_dir = /content/runs`.
  - Lokal: `base_dir = Path.cwd()`.
  - Gerät: `cuda`, wenn verfügbar.
  - `login_wandb()` nimmt in Colab `userdata.get("WANDB_API_KEY")`, lokal reicht `wandb.login()`.
  - `google.colab` wird nur innerhalb der Funktionen importiert, mit `# type: ignore` für pyright.
  - `check_versions()` ruft `version_mismatches(requires("infm-sota2") or [], importlib.metadata.version)` auf.
- [ ] **Schritt 5:** Tests erneut ausführen. Erwartet: PASS.
- [ ] **Schritt 6:** Commit: `feat: Environment erkennt Colab oder lokal`.

### Task 2: `TextSplits`

**Dateien:**
- Neu: `src/infm_sota2/data/splits.py`
- Export in `src/infm_sota2/data/__init__.py`
- Hilfsfunktion `write_mini_book` in `tests/helpers.py`
- Test: `tests/test_splits.py`

**Schnittstelle:**
- Nutzt `clean_book_text`, `split_text`, `CleaningStats` und `BPEDataset`.
- Liefert die Klasse `TextSplits` mit:
  - Attributen `train: str`, `val: str`, `test: str`, `stats: CleaningStats`, `seq_len: int`, `max_chars: dict[str, int] | None` und `tokenizer: tiktoken.Encoding`.
  - der Klassenmethode `from_book(cls, data_dir: Path, file: str, start_line: str, end_line: str, book_title: str, train_ratio: float, val_ratio: float, max_chars: dict[str, int] | None, seq_len: int) -> TextSplits`. Die Schlüssel passen zu `CONFIG["data"]`, das Notebook ruft also `from_book(env.data_dir, seq_len=…, **CONFIG["data"])` auf.
  - `summary() -> str` mit dem Text der alten Zelle 9.
  - `split_summary(batch_size: int) -> str` mit dem Text der alten Zellen 10 und 11.
  - `train_loader(batch_size: int, seed: int) -> DataLoader` mit `shuffle=True`, `drop_last=True` und einem Generator mit `seed`.
  - `loader(split: Literal["val", "test"], batch_size: int) -> DataLoader` ohne Mischen.

- [ ] **Schritt 1:** `write_mini_book(directory: Path) -> str` in `tests/helpers.py` schreiben.
  - Die Funktion legt `mini-book.txt` an: Vorspann, `PROLOGUE`, 200 Absätze `f"Sentence number {i} tells a short story about tests."` mit Leerzeilen dazwischen, dann `APPENDIX A`.
  - Sie gibt den Dateinamen zurück.
- [ ] **Schritt 2:** Fehlschlagende Tests schreiben (`seq_len=8`).
  - `test_splits_cover_the_cleaned_text`: `train + val + test` ist gleich `clean_book_text(...)[0]`.
  - `test_max_chars_cuts_each_split`: Bei `{"train": 300, "val": 100, "test": 100}` sind die Längen 300, 100 und 100.
  - `test_same_seed_gives_same_train_order`: Gleicher Seed ergibt den gleichen ersten Batch, ein anderer Seed einen anderen.
  - `test_eval_loader_keeps_order`: Der erste Batch von `loader("val", 4)` sind die Fenster 0 bis 3.
  - `test_summary_lists_cleaning_counts`: Der Text enthält `f"Absätze:                       {stats.paragraphs:,}"`.
  - `test_split_summary_names_steps_per_epoch`: Der Text endet mit `f"Schritte pro Epoche: {len(train_loader(4, 1))}"`.
- [ ] **Schritt 3:** `uv run python -m unittest tests.test_splits` ausführen. Erwartet: FAIL.
- [ ] **Schritt 4:** `splits.py` umsetzen.
  - Die Texte von `summary()` und `split_summary()` übernehmen die `print`-Zeilen der alten Zellen 9 bis 11 Zeichen für Zeichen, damit der Vergleich in Task 6 nichts findet.
  - Die Datasets entstehen einmal in `from_book`.
- [ ] **Schritt 5:** Tests erneut ausführen. Erwartet: PASS.
- [ ] **Schritt 6:** Commit: `feat: TextSplits bereitet Buchtext und Loader auf`.

### Task 3: `build_model` und `load_pretrained`

**Dateien:**
- Neu: `src/infm_sota2/model/build.py`
- Ändern: `src/infm_sota2/model/pretrained.py`, Exporte in `src/infm_sota2/model/__init__.py`
- Test: `tests/test_build.py`

**Schnittstelle:**
- `build_model[M: nn.Module](model_cls: Callable[[dict[str, Any]], M], config: dict[str, Any], seed: int, device: torch.device) -> M` setzt `torch.manual_seed(seed)`, baut das Modell und schiebt es aufs Gerät.
- `load_pretrained[M: nn.Module](model_cls: Callable[[dict[str, Any]], M], config: dict[str, Any], weights_path: Path, device: torch.device) -> M` baut das Modell ohne neuen Seed, lädt die Gewichte mit `map_location="cpu"`, `weights_only=True` und strikt, dann `.to(device)`.
  Ohne neuen Seed verbraucht es dieselben Zufallszahlen wie der alte Code, und das Nachtrainieren bleibt gleich.

- [ ] **Schritt 1:** Fehlschlagende Tests schreiben.
  - `test_build_model_is_seeded`: Zwei Aufrufe mit Seed 1 liefern gleiche Gewichte.
  - `test_load_pretrained_restores_weights`: Ein gespeichertes `tiny_model(seed=1).state_dict()` lädt in ein neues Modell, alle Tensoren sind gleich.
  - `test_load_pretrained_is_strict`: Fehlt ein Schlüssel, gibt es einen `RuntimeError`.
  - `test_load_pretrained_uses_rng_like_plain_construction`: Nach `torch.manual_seed(5)` und `load_pretrained(...)` ergibt `torch.rand(1)` denselben Wert wie nach `torch.manual_seed(5)` und `GPTModel(TINY_CONFIG)`.
- [ ] **Schritt 2:** `uv run python -m unittest tests.test_build` ausführen. Erwartet: FAIL.
- [ ] **Schritt 3:** Beide Funktionen umsetzen.
- [ ] **Schritt 4:** Tests erneut ausführen. Erwartet: PASS.
- [ ] **Schritt 5:** Commit: `feat: Modelle mit Seed bauen und OpenAI-Gewichte laden`.

### Task 4: `TrainConfig` und `Trainer`

**Dateien:**
- Neu: `src/infm_sota2/training/trainer.py`
- Exporte in `src/infm_sota2/training/__init__.py`
- Test: `tests/test_trainer.py`

**Schnittstelle:**
- Nutzt `Environment`, `WANDB_ENTITY`, `WANDB_PROJECT`, `TextSplits`, `train`, `load_checkpoint` und `TrainResult`.
- Liefert:
  - `TrainConfig` als `@dataclass(frozen=True)` mit `weight_decay: float`, `batch_size: int`, `eval_every: int`, `eval_batches: int | None`, `grad_clip: float`, `use_amp: bool` und `seed: int`.
  - `Trainer(env: Environment, wandb_config: dict[str, object])`.
  - `Trainer.fit(model: nn.Module, data: TextSplits, config: TrainConfig, learning_rate: float, epochs: int, name: str, group: str, checkpoint_name: str | None) -> TrainResult`.

`fit` macht dasselbe wie das alte `run_training`, plus `keep_checkpoint` und `load_checkpoint` aus den alten Zellen 19 und 22, in dieser Reihenfolge:

1. `AdamW(lr=learning_rate, weight_decay=…)`.
2. `wandb.init(entity, project, group, name, mode=env.wandb_mode, config={**wandb_config, "run": {"learning_rate": …, "epochs": …}})`.
3. `train(...)` mit `data.train_loader(batch_size, seed)` und `data.loader("val", batch_size)`.
4. Ins Summary kommen `best_val_loss` und `best_step`, dann `finish()`.
5. Gedruckt wird dieselbe Zeile wie bisher: `"{name}: bester Validierungs-Loss … bei Schritt … von …"`.
6. Nur mit Checkpoint: zurückladen, nach `env.checkpoint_dir` kopieren und `"Checkpoint: {ziel}"` drucken.

- [ ] **Schritt 1:** Fehlschlagende Tests schreiben.
  - Aufbau für alle Tests:
    - `Environment(False, tmp, tmp / "runs", cpu, "disabled")`.
    - Die Daten kommen aus `write_mini_book` mit `seq_len=8`.
    - Das Modell ist `tiny_model(seed=1, vocab_size=50257)`.
    - `TrainConfig(0.1, 4, 5, None, 1.0, False, 1)`.
  - `test_fit_gives_same_result_as_train`: `fit` liefert dasselbe `TrainResult` wie ein direkter Aufruf von `train()` mit denselben Werten und Seeds.
  - `test_fit_keeps_best_checkpoint`: Mit `checkpoint_name="best.pth"` gibt es `checkpoint_dir / "best.pth"`. Nach `fit` ergibt `evaluate` auf `val` den Wert `result.best_val_loss` (`places=5`).
  - `test_fit_without_checkpoint_writes_nothing`: Bei `checkpoint_name=None` bleibt `checkpoint_dir` leer oder fehlt.
- [ ] **Schritt 2:** `uv run python -m unittest tests.test_trainer` ausführen. Erwartet: FAIL.
- [ ] **Schritt 3:** `trainer.py` umsetzen.
- [ ] **Schritt 4:** Tests erneut ausführen. Erwartet: PASS.
- [ ] **Schritt 5:** Commit: `feat: Trainer bündelt Optimizer, W&B und Checkpoints`.

### Task 5: `ModelReport`

**Dateien:**
- Neu: `src/infm_sota2/training/report.py`
- Export in `src/infm_sota2/training/__init__.py`
- Test: `tests/test_report.py`

**Schnittstelle:**
- Nutzt `TextSplits`, `evaluate`, `complete` und `longest_copied_span`.
- Liefert `ModelReport(data: TextSplits, device: torch.device, batch_size: int, seed: int, prompt: str, samples: int, max_new_tokens: int, temperature: float, top_k: int | None)`. Die letzten fünf Namen passen zu `CONFIG["generation"]`.
- Methoden:
  - `add(name, model, context_size) -> None`: druckt wie das alte `record`.
  - `add_texts(name, texts) -> None`.
  - `test_loss(name) -> float`.
  - `losses() -> list[dict[str, object]]` mit den Schlüsseln `Modell`, `Test-Loss`, `Perplexity`.
  - `completions() -> list[dict[str, object]]` mit `Modell`, `Nr.`, `Text`, `Kopiert (Wörter)` und `Längste kopierte Folge`. Verglichen wird mit `data.train`.
- Text Nr. `i` (ab 0) erzeugt `complete` mit `torch.Generator(device=device).manual_seed(seed + i)`, wie bisher.

- [ ] **Schritt 1:** Fehlschlagende Tests schreiben. Aufbau wie in Task 4, mit `samples=2`, `max_new_tokens=3`, `temperature=0.8` und `top_k=5`.
  - `test_losses_match_evaluate`: `losses() == [{"Modell": "A", "Test-Loss": x, "Perplexity": math.exp(x)}]` mit `x = evaluate(model, data.loader("test", 4), cpu, None)`.
  - `test_same_seed_gives_same_texts`: Zwei Reports mit gleichem Seed liefern gleiche `completions()`.
  - `test_external_texts_appear_only_in_completions`: `add_texts("Claude", ["x", "y"])` gibt zwei Zeilen in `completions()` und keine in `losses()`.
  - `test_completion_row_counts_copied_words`: `Kopiert (Wörter)` ist gleich `len(longest_copied_span(text, data.train).split())`.
- [ ] **Schritt 2:** `uv run python -m unittest tests.test_report` ausführen. Erwartet: FAIL.
- [ ] **Schritt 3:** `report.py` umsetzen.
- [ ] **Schritt 4:** Tests erneut ausführen. Erwartet: PASS.
- [ ] **Schritt 5:** Commit: `feat: ModelReport misst Test-Loss und erzeugt Texte`.

### Task 6: LOTR-Notebook umstellen

**Dateien:**
- Ändern: `notebooks/workbook_timo_lotr.ipynb`, `docs/lotr-training.md`, `README.md`
- Neu: `tests/test_notebooks.py`

Ein Hilfsskript im Scratchpad (`run_smoke.py`, kommt nicht ins Repo) setzt `SMOKE_TEST = True`, führt das Notebook aus und schreibt alle stdout-Ausgaben in eine Datei:

```python
import sys, nbformat
from nbclient import NotebookClient

nb = nbformat.read(sys.argv[1], as_version=4)
for cell in nb.cells:
    if cell.cell_type == "code" and cell.source.strip() == "SMOKE_TEST = False":
        cell.source = "SMOKE_TEST = True"
NotebookClient(nb, timeout=900, resources={"metadata": {"path": "notebooks"}}).execute()
with open(sys.argv[2], "w", encoding="utf-8") as out:
    for cell in nb.cells:
        for o in cell.get("outputs", []):
            if o.output_type == "stream" and o.name == "stdout":
                out.write(o.text)
```

Neue Zellen. Die Ausgaben der alten Zellen werden übernommen:

| Neu | Inhalt | Ausgabe von alter Zelle |
|---|---|---|
| 0, 1, 2 | unverändert, nur der Colab-Link in Zelle 0 zeigt auf `blob/main` | 0, 1, 2 |
| 3 | `BRANCH = "main"`, `%pip install` (nur Colab), `env = Environment.detect(smoke_test=SMOKE_TEST)`, Versionsmeldungen, `DATA_PATH`, Ausgabe von Text und Checkpoints wie bisher | 3 + 4 |
| 4, 5 | Markdown und `CONFIG` unverändert, `CONFIG["environment"] = {**env.describe(), "mixed_precision": USE_AMP, "branch": BRANCH}` | 5, 6 |
| 6 | `if not SMOKE_TEST: env.login_wandb()` | 7 |
| 7, 8 | Markdown, `data = TextSplits.from_book(env.data_dir, seq_len=…, **CONFIG["data"])` und `print(data.summary())` | 8, 9 + 10 |
| 9 | `print(data.split_summary(CONFIG["batch_size"]))` | 11 |
| 10, 11 | Markdown zu den Bausteinen (neuer Text), dann `TRAIN = TrainConfig(...)`, `trainer = Trainer(env, CONFIG)` und `report = ModelReport(...)` | 12, 13 |
| ab 12 | die alten Zellen 14 bis 29 mit `build_model`, `trainer.fit`, `report.add`, `load_pretrained`, `report.add_texts`, `report.losses()` und `report.completions()`. Markdown, Claude-Zellen und Fazit bleiben unverändert. | 14 bis 29 |

- [ ] **Schritt 1:** Ausgangswerte vor dem Umbau festhalten.
  `uv run --with nbclient python <scratchpad>/run_smoke.py notebooks/workbook_timo_lotr.ipynb <scratchpad>/lotr-before.txt` ausführen.
  Erwartet: Das Notebook läuft durch, und die Datei enthält Losses und Texte.
- [ ] **Schritt 2:** Fehlschlagenden Test schreiben, in `tests/test_notebooks.py`.
  `test_timo_notebooks_install_from_main`: Für jedes Notebook in `TIMO_NOTEBOOKS = ["workbook_timo_lotr.ipynb"]` gibt es eine Code-Zeile `BRANCH = "main"`, und der Colab-Link enthält `blob/main/notebooks/<name>`.
- [ ] **Schritt 3:** `uv run python -m unittest tests.test_notebooks` ausführen. Erwartet: FAIL, weil `BRANCH` noch `"feature/lotr-training"` ist.
- [ ] **Schritt 4:** Das Notebook per `nbformat`-Skript nach der Tabelle umbauen.
- [ ] **Schritt 5:** Das Notebook nach dem Umbau laufen lassen, in `lotr-after.txt`, dann `diff lotr-before.txt lotr-after.txt` ausführen.
  Erwartet: Nur die Zeile mit dem `environment`-Dict unterscheidet sich, weil `branch` jetzt `main` ist und die Schlüssel anders sortiert sind. Alle Losses, Lernraten und Texte sind gleich.
- [ ] **Schritt 6:** Die Ausgaben des Smoke-Laufs verwerfen.
  Im Repo bleibt das Notebook mit `SMOKE_TEST = False` und den übernommenen Ausgaben vom 12.11.
- [ ] **Schritt 7:** Doku nachziehen.
  - In `docs/lotr-training.md`: Abschnitte „Aufbau“ und „Code“ mit den neuen Klassen, Anleitung Schritt 2 auf Branch `main`.
  - In der README unter Projektstruktur: `env.py`, `data/splits.py`, `training/trainer.py`, `training/report.py`, `model/build.py`.
- [ ] **Schritt 8:** Alle Checks ausführen: `uv run ruff check`, `uv run ruff format --check`, `uv run pyright` und `uv run python -m unittest discover -s tests -p 'test*.py'`. Dann `uv run coverage run -m unittest discover -s tests -p 'test*.py'` und `uv run coverage report`.
  Erwartet: alles grün.
- [ ] **Schritt 9:** Commit `refactor: LOTR-Notebook nutzt die neuen Klassen`. Dann Timo fragen, ob der Pull Request nach `week/05` erstellt werden soll.

---

## Teil 2: Woche 5

### Task 7: `GPTModelV2`

**Dateien:**
- Neu: `src/infm_sota2/model/gpt_v2.py`
- Ändern: `src/infm_sota2/model/pretrained.py` (`load_pretrained`), Exporte in `model/__init__.py`
- Test: `tests/test_gpt_v2.py`

**Schnittstelle:**
- `GPTModelV2(cfg: dict[str, Any])` nimmt dieselben Schlüssel wie `GPTModel`. `forward(in_idx: Tensor) -> Tensor` liefert die Logits.
  - Parameternamen: `tok_emb`, `pos_emb`, `trf_blocks.{i}.att.qkv`, `trf_blocks.{i}.att.out_proj`, `trf_blocks.{i}.ff.{0,2}`, `trf_blocks.{i}.norm1`, `trf_blocks.{i}.norm2`, `final_norm` und `out_head`.
  - Es gibt keine Buffer.
- `convert_v1_state_dict(state: dict[str, Tensor]) -> dict[str, Tensor]`:
  - Schlüssel auf `.att.mask` fallen weg.
  - `W_query`, `W_key` und `W_value` werden in dieser Reihenfolge mit `torch.cat(dim=0)` zu `att.qkv` zusammengefügt, für `weight` und, falls vorhanden, `bias`.
  - `.ff.layers.` wird zu `.ff.`.
  - Bei `norm1`, `norm2` und `final_norm` wird `.scale` zu `.weight` und `.shift` zu `.bias`.
- `load_pretrained` rechnet um, wenn `model_cls` `GPTModelV2` ist.
- Aufbau laut Spezifikation, Abschnitt „GPTModelV2“:
  - `qkv(x).view(b, t, 3, n_heads, head_dim).permute(2, 0, 3, 1, 4)`.
  - `F.scaled_dot_product_attention(q, k, v, dropout_p=drop_rate if self.training else 0.0, is_causal=True)`.
  - `nn.GELU(approximate="tanh")` und `nn.LayerNorm(emb_dim, eps=1e-5)`.

- [ ] **Schritt 1:** Branch anlegen: `git switch -c feature/attention-v2`, vom Stand nach Task 6.
- [ ] **Schritt 2:** Fehlschlagende Tests schreiben.
  - `test_matches_v1_with_converted_weights`:
    - Aufbau: `v1 = tiny_model(seed=1).eval()`. In jedem Block wird `ff.layers[1]` durch `nn.GELU(approximate="tanh")` ersetzt. `v2 = GPTModelV2(TINY_CONFIG).eval()` lädt `convert_v1_state_dict(v1.state_dict())` strikt.
    - Erwartet: `torch.allclose(v1(x), v2(x), atol=1e-5)` für `x` mit der Form `(2, 6)`.
  - `test_ignores_future_tokens`: wie der Test in `tests/test_model.py`, nur für V2.
  - `test_converts_without_qkv_bias`: dasselbe mit `qkv_bias=False` bei beiden Modellen.
  - `test_math_kernel_gives_same_logits`: Mit `sdpa_kernel(SDPBackend.MATH)` sind die Logits gleich wie ohne (`atol=1e-6`).
  - `test_has_no_mask_buffer`: Kein Schlüssel im `state_dict` enthält `mask`.
  - `test_load_pretrained_converts_v1_file`: Eine V1-Datei lädt per `load_pretrained(GPTModelV2, ...)`, die Logits sind wie im ersten Test.
- [ ] **Schritt 3:** `uv run python -m unittest tests.test_gpt_v2` ausführen. Erwartet: FAIL.
- [ ] **Schritt 4:** `gpt_v2.py` umsetzen und `load_pretrained` ergänzen.
- [ ] **Schritt 5:** Tests erneut ausführen. Erwartet: PASS.
- [ ] **Schritt 6:** Commit: `feat: GPTModelV2 aus torch.nn mit scaled_dot_product_attention`.

### Task 8: Benchmark

**Dateien:**
- Neu: `src/infm_sota2/benchmark.py`
- Test: `tests/test_benchmark.py`

**Schnittstelle:**
- Nutzt `batch_loss` aus `training/loop.py`.
- Typen:
  - `Precision = Literal["fp32", "fp16-mixed", "bf16-mixed", "bf16"]`.
  - `Attention = Literal["auto", "math"]`.
- `Variant` als `@dataclass(frozen=True)` mit `name: str`, `model_cls: Callable[[dict[str, Any]], nn.Module]`, `precision: Precision`, `attention: Attention` und `compile_backend: str | None`.
  - `None` heißt: nicht kompilieren.
  - Das Notebook nimmt `"inductor"`, im Smoke-Test und in den Tests `"eager"`, weil Windows für `inductor` einen C++-Compiler braucht.
- `BenchmarkResult` als `@dataclass(frozen=True)` mit `variant: str`, `tokens_per_second: float | None`, `ms_per_step: float | None`, `peak_memory_gb: float | None`, `flops_per_token: int`, `mfu: float | None` und `error: str | None`.
- `flops_per_token(model: nn.Module, config: dict[str, Any], seq_len: int) -> int` rechnet `6 · Σ nn.Linear.weight.numel() + 12 · n_layers · seq_len · emb_dim`.
- `peak_tflops(gpu_name: str | None, precision: Precision) -> float | None` nutzt die Tabelle aus der Spezifikation:
  - Geprüft wird in der Reihenfolge `L40S`, `L4`, `A100`, `H100 PCIe`, `H100`, `T4`.
  - Bei der T4 ist bf16 `None`.
  - Die Werte vor dem Eintragen gegen die NVIDIA-Datenblätter prüfen und die Quelle als Kommentar notieren.
- `Benchmark(model_config: dict[str, Any], batch_size: int, seq_len: int, warmup_steps: int, measure_steps: int, device: torch.device, seed: int)`:
  - Der Konstruktor gibt `ValueError`, wenn `seq_len > model_config["context_length"]`.
  - `run(variant: Variant) -> BenchmarkResult` läuft wie in der Spezifikation, Abschnitt „Benchmark“, beschrieben.
  - Die Lernrate ist die Konstante `1e-4`, weil sie die Geschwindigkeit nicht beeinflusst.
  - Bei `"math"` wird mit `sdpa_kernel(SDPBackend.MATH)` gerechnet.
  - Bei `fp16-mixed` auf der CPU gibt es `ValueError`.
  - `torch.OutOfMemoryError` wird zum Ergebnis mit `error="zu wenig Speicher"`.
  - Zum Schluss: `del`, `gc.collect()` und auf CUDA `empty_cache()`.

- [ ] **Schritt 1:** Fehlschlagende Tests schreiben. Aufbau: `Benchmark(TINY_CONFIG, 2, 8, 1, 2, cpu, 1)`.
  - `test_flops_per_token_for_tiny_models`: `flops_per_token(tiny_model(1), TINY_CONFIG, 8) == 41_856`, ebenso für `GPTModelV2(TINY_CONFIG)`.
    Rechnung: Die Linear-Gewichte sind `2 · (768 + 256 + 2048) + 320 = 6464`, also `6 · 6464 + 12 · 2 · 8 · 16 = 41_856`.
  - `test_peak_tflops_by_gpu_name`:
    - `("NVIDIA L40S", "bf16-mixed")` ergibt 362.0.
    - `("NVIDIA L4", "bf16")` ergibt 121.0.
    - `("NVIDIA H100 PCIe", "fp32")` ergibt 51.0.
    - `("NVIDIA H100 80GB HBM3", "bf16-mixed")` ergibt 989.0.
    - `("Tesla T4", "fp16-mixed")` ergibt 65.0.
    - `("Tesla T4", "bf16-mixed")`, `(None, "fp32")` und `("Unknown", "fp32")` ergeben `None`.
  - `test_run_on_cpu_reports_speed`: V1 mit `fp32` liefert endliche Tokens/s über 0, `peak_memory_gb is None`, `mfu is None`, `error is None` und `flops_per_token == 41_856`.
  - `test_bf16_and_math_kernel_run_on_cpu`: V2 mit `bf16-mixed` und `math` sowie V1 mit `bf16` laufen ohne Fehler.
  - `test_compiled_variant_runs`: V2 mit `compile_backend="eager"` läuft ohne Fehler.
  - `test_out_of_memory_is_recorded_and_next_run_works`:
    - Das Modell ist eine Klasse im Test, deren `forward` einen `torch.OutOfMemoryError` wirft.
    - Erwartet: `error == "zu wenig Speicher"` und `tokens_per_second is None`. Danach läuft eine V1-Variante normal.
  - `test_rejects_seq_len_longer_than_context`: Bei `seq_len=9` gibt es `ValueError`.
  - `test_fp16_needs_cuda`: Bei `fp16-mixed` auf der CPU gibt es `ValueError`.
- [ ] **Schritt 2:** `uv run python -m unittest tests.test_benchmark` ausführen. Erwartet: FAIL.
- [ ] **Schritt 3:** `benchmark.py` umsetzen.
  - Gemessen wird nach den Aufwärmschritten mit `torch.cuda.Event(enable_timing=True)` und `synchronize()`, auf der CPU mit `time.perf_counter()`.
  - `torch.cuda.reset_peak_memory_stats()` läuft vor den Messschritten.
- [ ] **Schritt 4:** Tests erneut ausführen. Erwartet: PASS.
- [ ] **Schritt 5:** Commit: `feat: Benchmark misst Tokens/s, Speicher und MFU`.

### Task 9: Notebook für Woche 5

**Dateien:**
- Neu: `notebooks/workbook_timo_attention_v2.ipynb`, per `nbformat`-Skript erzeugt
- Ändern: `tests/test_notebooks.py` (`TIMO_NOTEBOOKS` um das neue Notebook ergänzen), `docs/attention-v2.md`, `README.md`

Aufbau des Notebooks laut Spezifikation, Abschnitt „Notebook“:

- Colab-Link auf `blob/main`.
- `SMOKE_TEST = False`.
- Eine Setup-Zelle mit `BRANCH = "main"`, `%pip install`, `Environment.detect` und der GPU-Ausgabe.
- Einstellungen:
  - Normaler Lauf: `GPT2_SMALL_CONFIG` mit `batch_size=4`, `seq_len=1024`, `warmup_steps=3`, `measure_steps=10` und `seed=123`.
  - Smoke-Test: winzige Konfiguration (Embedding 32, 2 Köpfe, 2 Schichten, Kontext 32) mit `batch_size=2`, `seq_len=32`, `warmup_steps=1` und `measure_steps=2`.
- Die sechs Varianten aus der Spezifikation in einer Zelle. Auf einer GPU ohne bf16 (T4, Compute Capability unter 8) nimmt Nr. 2 `fp16-mixed`.
- Pro Aufgabe der Folie ein Markdown-Abschnitt und eine Zeile `results.append(bench.run(VARIANTS[i]))`. Danach `pd.DataFrame(map(asdict, results))`.
- Ein Abschnitt V1 gegen V2 mit den OpenAI-Gewichten, ohne Zufall (`temperature=0`). Im Smoke-Test wird er übersprungen und sagt das.
- Die Tabelle aller Varianten und zwei Diagramme mit `df.plot.barh` (Tokens/s und Speicher).
- Ein leeres Fazit zum Ausfüllen nach dem Lauf.

- [ ] **Schritt 1:** `"workbook_timo_attention_v2.ipynb"` in `TIMO_NOTEBOOKS` eintragen.
  `uv run python -m unittest tests.test_notebooks` ausführen. Erwartet: FAIL, weil die Datei fehlt.
- [ ] **Schritt 2:** Das Notebook per `nbformat`-Skript erzeugen.
- [ ] **Schritt 3:** `uv run --with nbclient python <scratchpad>/run_smoke.py notebooks/workbook_timo_attention_v2.ipynb <scratchpad>/bench-smoke.txt` ausführen.
  Erwartet: Das Notebook läuft durch, und alle sechs Varianten haben `error = None`.
  Das Skript speichert das ausgeführte Notebook nicht, im Repo bleibt es also ohne Ausgaben, bis Timo es auf der GPU laufen lässt.
- [ ] **Schritt 4:** `test_notebooks` ausführen. Erwartet: PASS.
- [ ] **Schritt 5:** Doku nachziehen.
  - In `docs/attention-v2.md`: Statt „Design, noch nicht umgesetzt“ steht „Umgesetzt, die Ergebnisse vom GPU-Lauf fehlen noch“.
  - In der README unter Projektstruktur: `model/gpt_v2.py`, `benchmark.py` und `docs/attention-v2.md`.
- [ ] **Schritt 6:** Alle Checks wie in Task 6, Schritt 8. Erwartet: alles grün.
- [ ] **Schritt 7:** Commit `feat: Notebook für Woche 5 mit Benchmark aller Schritte`. Dann Timo fragen, ob der Pull Request nach `week/05` erstellt werden soll.

## Nicht in diesem Plan

Teil 3 aus der Spezifikation, der KV-Cache, bekommt einen eigenen Plan, wenn Teil 2 fertig ist.
