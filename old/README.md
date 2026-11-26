# old/

Hier liegt, was seit dem Aufräumen am 26.11.2026 nicht mehr gebraucht wird. Gelöscht wurde nichts. Die Ordner spiegeln die alten Pfade, `old/notebooks/workbook_timo_lotr.ipynb` lag also vorher unter `notebooks/workbook_timo_lotr.ipynb`.

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
