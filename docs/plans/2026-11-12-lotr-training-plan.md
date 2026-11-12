# LOTR-Training: Implementierungsplan

> **Ausführung:** Aufgabe für Aufgabe, testgetrieben (erst roter Test, dann Code).

**Ziel:** Trainings-, Mess- und Generierungscode in `src/infm_sota2/`, mit dem das Notebook `workbook_timo_lotr.ipynb` GPT-2 Small auf dem bereinigten Herr-der-Ringe-Text trainiert und vier GPT-2-Varianten plus Claude vergleicht.

**Architektur:** Reine Funktionen ohne Default-Werte für Hyperparameter. W&B wird nur über eine übergebene `log`-Funktion angebunden. Das Notebook hält alle Werte in einem `CONFIG`.

**Technik:** Python 3.13, torch 2.14.1, tiktoken 0.14.0, wandb 0.28.1, unittest

**Spezifikation:** [`docs/lotr-training.md`](../lotr-training.md)

## Globale Vorgaben

- Keine neuen Laufzeit-Abhängigkeiten; Download über `urllib`, Prüfsumme über `hashlib`.
- Hyperparameter ohne Default-Werte in Funktionssignaturen.
- Fehler laut melden: `ValueError` bei falschen Eingaben, `FloatingPointError` bei NaN/Inf-Loss, kein breites `except`.
- `torch.load(..., weights_only=True)` überall.
- Gewichte: `https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/402728fa68de0d3bb91e3e148ae351511eb7f4db/gpt2-small-124M.pth`, SHA-256 `24a9078c5b27137fb2706d2206349c759952a7de8f72ef8e305dc02511bcabf8`.
- ruff, `ruff format`, pyright (standard) und alle Unittests grün; Tests laufen auf der CPU in Sekunden.
- Keine Ausschnitte aus dem Buch in Tests oder Repo. Tests nutzen selbst geschriebene Mini-Texte.

## Review-Fokus

1. Marker-Zeile fehlt oder kommt doppelt vor (andere Ausgabe des Buchs) → `ValueError`, nicht stilles Abschneiden. Test in Aufgabe 1.
2. Abgebrochener Download hinterlässt eine Teildatei → nächster Lauf lädt neu, eine fertige Datei mit falscher Prüfsumme bricht ab. Tests in Aufgabe 2.
3. Prompt länger als der Kontext → `generate` schneidet auf die letzten `context_size` Tokens zu. Test in Aufgabe 3.
4. Mixed Precision auf der CPU angefordert → `ValueError`. Test in Aufgabe 4.
5. Ratios ohne Platz für ein Testset (`train + val >= 1`) → `ValueError`. Test in Aufgabe 1.

---

### Aufgabe 1: Textbereinigung und Aufteilung

**Dateien:** neu `src/infm_sota2/data/cleaning.py`, Export in `src/infm_sota2/data/__init__.py`, Test `tests/test_cleaning.py`

**Schnittstelle:**
- `CleaningStats` (frozen dataclass): `narrative_lines: int`, `headers_removed: int`, `header_titles: dict[str, int]`, `page_breaks_joined: int`, `page_breaks_paragraph: int`, `hyphens_joined: int`, `hyphens_kept: int`, `paragraphs: int`
- `clean_book_text(text: str, start_line: str, end_line: str, book_title: str) -> tuple[str, CleaningStats]`
- `split_text(text: str, train_ratio: float, val_ratio: float) -> tuple[str, str, str]`

**Regeln (nicht aus der Signatur ableitbar):**
- Zeilen werden gestrippt. Erzählteil = von der Zeile, die genau `start_line` ist (inklusive), bis zur Zeile, die genau `end_line` ist (exklusive). Jede der beiden muss genau einmal vorkommen.
- Kopfzeile = `^\S{1,6} {book_title}$` (gerade Seite), `^[A-Z][A-Z ,'\-]+ [0-9IOQSlZ]{1,5}$` (ungerade Seite, Kapitelname plus OCR-Seitenzahl) oder `^[0-9]{1,4}$`. `header_titles` zählt den Kapitelnamen ohne Seitenzahl bzw. den Buchtitel.
- Seitenwechsel = Kopfzeile(n) samt umgebenden Leerzeilen. Fortsetzung, wenn die vorige Textzeile nicht auf `. ! ? " ” ’ '` endet oder die nächste kleingeschrieben beginnt, sonst neuer Absatz.
- Absätze: Zeilen eines Blocks werden mit Leerzeichen verbunden, Absätze mit `"\n\n"`. Endet eine Zeile auf `-` und beginnt die nächste kleingeschrieben: zusammenziehen ohne Bindestrich, wenn das Wort (kleingeschrieben) sonst im Erzählteil vorkommt, sonst Bindestrich behalten und ohne Leerzeichen verbinden.
- `split_text`: Schnitt jeweils direkt hinter dem ersten `"\n\n"` ab der Ziel-Zeichenposition. Es gilt `train + val + test == text`.

**Tests:**
- `test_keeps_only_text_between_markers`, `test_rejects_missing_marker`, `test_rejects_duplicate_marker`
- `test_removes_even_and_odd_page_headers` (zählt `headers_removed == 2`, `header_titles`)
- `test_joins_sentence_across_page_break` (`page_breaks_joined == 1`, Satz ohne `"\n\n"` in der Mitte)
- `test_page_break_after_full_sentence_starts_paragraph`
- `test_unwraps_lines_into_paragraphs`
- `test_dehyphenates_known_word` („some-/thing“, wenn „something“ vorkommt) und `test_keeps_hyphen_of_compound` („long-/expected“)
- `test_split_preserves_all_text`, `test_split_cuts_at_paragraph_boundaries`, `test_split_rejects_ratios_without_test_part`

### Aufgabe 2: GPT-2-Konfiguration, Gewichte, Modell-Shape

**Dateien:** neu `src/infm_sota2/model/pretrained.py`, Export in `src/infm_sota2/model/__init__.py`, Tests `tests/test_pretrained.py`, `tests/test_model.py`

**Schnittstelle:**
- `GPT2_SMALL_CONFIG: dict[str, int | float | bool]` = vocab 50257, context 1024, emb 768, heads 12, layers 12, drop 0.1, qkv_bias True
- `GPT2_SMALL_WEIGHTS_URL: str`, `GPT2_SMALL_WEIGHTS_SHA256: str` (Werte aus den globalen Vorgaben)
- `sha256_of(path: Path) -> str`
- `download_weights(url: str, sha256: str, dest: Path) -> Path`: Existiert `dest`, wird nur geprüft, bei falscher Summe `ValueError`. Sonst Download nach `dest` + `.part`, prüfen, umbenennen; bei falscher Summe `.part` löschen und `ValueError`.

**Tests** (Download über `file://`-URLs auf temporäre Dateien):
- `test_sha256_of_matches_hashlib`
- `test_download_copies_file_with_matching_hash`
- `test_download_rejects_wrong_hash_and_leaves_no_file`
- `test_existing_file_is_verified_not_downloaded` (URL zeigt ins Leere)
- `test_existing_file_with_wrong_hash_raises`
- `test_gpt_model_returns_logits_per_token` (winzige Konfiguration: Shape `(batch, tokens, vocab)`)

### Aufgabe 3: Text erzeugen

**Dateien:** neu `src/infm_sota2/training/generate.py`, `src/infm_sota2/training/__init__.py`, Test `tests/test_generate.py`

**Schnittstelle:**
- `generate(model: nn.Module, idx: Tensor, max_new_tokens: int, context_size: int, temperature: float, top_k: int | None, generator: torch.Generator | None) -> Tensor`; `temperature == 0` heißt greedy, `< 0` ist ein `ValueError`. Läuft unter `torch.no_grad()` im eval-Modus.
- `complete(model: nn.Module, tokenizer: tiktoken.Encoding, prompt: str, max_new_tokens: int, context_size: int, temperature: float, top_k: int | None, generator: torch.Generator | None) -> str`: gibt nur die Fortsetzung zurück, Gerät aus den Modellparametern.

**Tests** (winziges `GPTModel`, vocab 20):
- `test_appends_max_new_tokens`
- `test_greedy_picks_argmax`
- `test_top_k_one_equals_greedy`
- `test_same_seed_gives_same_sample`
- `test_crops_prompt_longer_than_context`
- `test_rejects_negative_temperature`
- `test_complete_returns_only_continuation`

### Aufgabe 4: Trainingsschleife und Checkpoints

**Dateien:** neu `src/infm_sota2/training/loop.py`, Test `tests/test_training.py`

**Schnittstelle:**
- `batch_loss(model: nn.Module, inputs: Tensor, targets: Tensor) -> Tensor`: Cross-Entropy über alle Positionen
- `evaluate(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int | None) -> float`
- `TrainResult` (dataclass): `best_val_loss: float`, `best_step: int`, `steps: int`, `tokens_seen: int`
- `train(model, optimizer, train_loader, val_loader, device, epochs: int, eval_every: int, eval_batches: int | None, grad_clip: float, use_amp: bool, checkpoint_path: Path | None, checkpoint_metadata: dict[str, object], log: Callable[[dict[str, float], int], None]) -> TrainResult`
  - prüft alle `eval_every` Schritte und am Ende jeder Epoche; loggt `train/loss` (Mittel seit der letzten Prüfung), `val/loss`, `epoch`, `tokens_seen`
  - speichert bei neuem Bestwert nach `checkpoint_path` (falls nicht `None`)
  - NaN/Inf-Loss: `FloatingPointError`; `use_amp` ohne CUDA: `ValueError`
- `save_checkpoint(path: Path, model: nn.Module, metadata: dict[str, object]) -> None`, `load_checkpoint(path: Path, model: nn.Module) -> dict[str, object]`

**Tests** (winziges Modell, Tokens aus einem sich wiederholenden Muster):
- `test_untrained_loss_is_close_to_log_vocab` (|Loss − ln 20| < 0.5)
- `test_evaluate_respects_max_batches`
- `test_training_lowers_validation_loss`
- `test_logs_train_and_val_loss`
- `test_saves_checkpoint_only_on_improvement` (Metadaten enthalten `step` und `val_loss`)
- `test_checkpoint_round_trip_restores_weights`
- `test_non_finite_loss_raises`
- `test_amp_requires_cuda`

### Aufgabe 5: Auswendiglern-Check

**Dateien:** neu `src/infm_sota2/training/analysis.py`, Test `tests/test_analysis.py`

**Schnittstelle:** `longest_copied_span(text: str, reference: str) -> str`: längste zusammenhängende Wortfolge aus `text`, die mit Wortgrenzen in `reference` vorkommt; Leerraum wird normalisiert, `""` wenn kein Wort passt.

**Tests:** `test_finds_longest_shared_word_sequence`, `test_respects_word_boundaries` („the“ passt nicht in „other“), `test_returns_empty_when_nothing_shared`, `test_ignores_whitespace_differences`

### Aufgabe 6: Abhängigkeiten und README

- `datasets` aus `dependencies` in die dev-Gruppe verschieben, `uv lock`.
- Prüfen: Die Schleife aus der README läuft über `requires("infm-sota2")` ohne Fehler.
- README: Projektstruktur um `data/cleaning.py`, `model/`, `training/`, `docs/` ergänzen und auf `docs/lotr-training.md` verweisen.

### Aufgabe 7: Notebook

`notebooks/workbook_timo_lotr.ipynb`, erzeugt per `nbformat`-Skript, Abschnitte wie in der Spezifikation. `SMOKE_TEST` schaltet auf ein winziges Modell, wenige Schritte, W&B aus und lässt das Laden der OpenAI-Gewichte aus. Die übersprungenen Zellen sagen das selbst.

**Prüfung:** Das Notebook lokal mit `SMOKE_TEST = True` komplett ausführen (`uv run --with nbclient`), danach Ausgaben löschen und den Colab-Link auf `blob/main` setzen.

### Aufgabe 8: Abschluss

`ruff check`, `ruff format --check`, `pyright`, Unittests, Coverage; Doku mit den gemessenen Zahlen aus der Bereinigung aktualisieren; Review des ganzen Branches.
