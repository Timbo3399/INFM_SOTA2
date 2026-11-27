# old/

Hier liegt, was seit dem Aufräumen am 26.11.2026 nicht mehr gebraucht wird. Gelöscht wurde nichts. Wie der Code heute aufgebaut ist, steht in [docs/code.md](../docs/code.md). Die Ordner hier spiegeln die alten Pfade, `old/notebooks/workbook_timo_lotr.ipynb` lag also vorher unter `notebooks/workbook_timo_lotr.ipynb`.

Nichts in `old/` wird importiert, geprüft oder getestet. Ruff, Pyright und die Tests schauen nur in `src/`, `tests/` und `scripts/`. Die Notebooks hier laufen mit dem heutigen Code nicht mehr, ihre gespeicherten Ausgaben kann man aber lesen. Ihre Colab-Links zeigen noch auf `notebooks/` in `main` und funktionieren nicht mehr.

Zuletzt lief alles zusammen auf Commit `9bbd922` von `week/06`. Wer dort etwas ausführen will, holt sich diesen Stand:

```bash
git switch --detach 9bbd922
```

Zurück geht es mit `git switch -`.

| Datei | Warum hier |
|---|---|
| `notebooks/workbook_artur.ipynb`, `workbook_marvin.ipynb`, `workbook_timo.ipynb`, `workbook_valentin.ipynb` | Übungen aus Woche 1 bis 3: Tokenizer, Embeddings, Attention, DataLoader |
| `notebooks/workbook_marvin_training_v1.ipynb` | Training mit Wikitext aus Woche 4 |
| `notebooks/workbook_timo_lotr.ipynb` | Woche 4: GPT-2 auf „Der Herr der Ringe“ in Colab, die Ergebnisse stehen im Notebook und in `docs/lotr-training.md` |
| `notebooks/workbook_timo_attention_v2.ipynb` | Woche 5: Benchmark der Schritte von Folie 20 auf einer A100, die Ergebnisse stehen im Notebook und in `docs/attention-v2.md` |
| `docs/lotr-training.md` | Entwurf und Ergebnisse von Woche 4, dazu die Regeln für die Bereinigung des Buchtexts, nach denen der Code bis heute bereinigt |
| `docs/attention-v2.md` | Entwurf und Ergebnisse von Woche 5, dazu die Herleitung von FLOPs und MFU im Benchmark |
| `docs/plans/` | die Umsetzungspläne von Woche 4 bis 6 |
| `src/infm_sota2/data/dataloader.py`, `embedding.py` | `create_dataloader` und `Embeddings` aus Woche 2. Das Training nutzt `TextSplits`, und die Embeddings stecken im Modell. |
| `src/infm_sota2/data/dataset.py` | alter Stand mit `BPEDataset.decode` und `vocab_size`, die nur die Notebooks von Woche 2 brauchten |
| `src/infm_sota2/transformer/attention.py` | alter Stand mit `SelfAttention` und `CausalAttention`, den Vorstufen von `MultiHeadAttention` aus Woche 3 |
| `tests/test_dataloader.py`, `test_attention.py` | alter Stand mit den Tests für diese Bausteine |
| `src/infm_sota2/training/loop.py` | alter Stand, in dem `evaluate` mit `max_batches` und `train` mit `eval_batches` nur einen Teil der Validierung prüfen konnten. Woche 6 prüft immer den ganzen Teil. |
| `src/infm_sota2/training/report.py` | alter Stand mit `ModelReport.add_texts` für die Texte von Claude und `losses` für die Tabelle im LOTR-Notebook |
| `tests/test_training.py`, `test_report.py` | alter Stand mit den Tests dafür |
| `src/infm_sota2/env.py` | alter Stand mit dem Teil für Colab (Google Drive, schnelle Platte `run_dir`, W&B-Login mit Colab-Secret, Versionsvergleich) und den Feldern `local_rank` und `launched_by_torchrun` |
| `src/infm_sota2/data/splits.py` | alter Stand mit dem Train-Loader ohne `DistributedSampler` und mit `summary` und `split_summary` für das LOTR-Notebook |
| `src/infm_sota2/training/trainer.py` | `Trainer` und `TrainConfig`. Heute macht `fit` dasselbe und nimmt die Werte direkt aus der TOML-Datei. |
| `tests/test_env.py`, `test_splits.py`, `test_trainer.py` | alter Stand mit den Tests dafür |
