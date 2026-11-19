# Woche 5: Attention V2

Woche 5, Aufgabe *Attention V2: increasing performance*. Neues Notebook `notebooks/workbook_timo_attention_v2.ipynb`.

Vorher ziehen wir den Code aus [`workbook_timo_lotr.ipynb`](../notebooks/workbook_timo_lotr.ipynb) in Klassen unter `src/infm_sota2/` um, damit beide Notebooks nur noch wenige Methoden aufrufen.

Stand: Teil 1 und 2 sind umgesetzt und auf einer A100 gemessen (Ergebnisse unten). Der KV-Cache (Teil 3) ist offen.

## Aufgabe

Laut Folie 20 aus Woche 5:

1. Die aktuelle Implementierung mit dem Skript aus Moodle benchmarken.
2. 16 Bit einbauen und erneut benchmarken.
3. GPT aus `torch.nn`-Bausteinen bauen, damit Tensor Cores und FlashAttention automatisch greifen, und erneut benchmarken.
4. Das Modell mit `torch.compile` kompilieren und erneut benchmarken.
5. Optional einen KV-Cache bauen.
6. Ein Paper vorstellen. Wir nehmen FlashAttention, das kommt später.

Das Notebook wird in der Vorlesung gezeigt. Deshalb hat jeder Schritt eine kurze Erklärung, einen Aufruf und sein Ergebnis. Die Logik liegt getestet in `src/`.

## Ergebnisse (Lauf vom 19.11.2026, NVIDIA A100-SXM4-40GB in Colab)

GPT-2 Small, Batchgröße 4, 1024 Tokens, 3 Aufwärm- und 10 Messschritte. Ein Trainingsschritt kostet 854 Mio. FLOPs pro Token.

| Nr. | Schritt | Tokens/s | ms pro Schritt | Speicher (GiB) | MFU | gegenüber Nr. 1 |
|---|---|---|---|---|---|---|
| 1 | Ausgangsstand, fp32 | 13.760 | 297,7 | 12,2 | 60,3 % (gegen fp32) | 1,0× |
| 2 | 16 Bit mit `autocast` | 35.447 | 115,6 | 10,5 | 9,7 % | 2,6× |
| 3 | 16 Bit, ganzes Modell | 50.702 | 80,8 | 7,6 | 13,9 % | 3,7× |
| 4 | torch.nn ohne Flash | 50.457 | 81,2 | 8,6 | 13,8 % | 3,7× |
| 5 | torch.nn mit Flash | 75.621 | 54,2 | 5,8 | 20,7 % | 5,5× |
| 6 | `torch.compile` | 92.586 | 44,2 | 5,8 | 25,4 % | 6,7× |

- **Zusammen sind die Schritte 6,7-mal so schnell wie der Ausgangsstand** und brauchen knapp die Hälfte des Speichers (5,8 statt 12,2 GiB).
- **16 Bit bringt den größten einzelnen Sprung.** Mit `autocast` laufen die Matrixprodukte auf den Tensor Cores, das ist 2,6-mal so schnell wie fp32.
- **Das ganze Modell in bf16 (Nr. 3)** ist noch einmal 1,4-mal so schnell wie `autocast` und braucht 2,9 GiB weniger. Ein Grund: `autocast` wandelt in jedem Schritt die fp32-Gewichte in bf16 um, und AdamW rechnet auf fp32-Kopien. Beides entfällt hier. Fürs Training bleibt es trotzdem bei `autocast`, weil AdamW in reinem bf16 kleine Updates verliert.
- **Die `torch.nn`-Bausteine** (Nr. 2 gegen Nr. 4, gleiche Präzision) sind 1,4-mal so schnell und sparen 1,9 GiB: fertige Kernel für LayerNorm und GELU und ein Linear für Q, K und V statt drei.
- **FlashAttention** (Nr. 4 gegen Nr. 5, sonst gleich) ist 1,5-mal so schnell und spart 2,8 GiB, ein Drittel des Speichers. Die T×T-Matrix der Attention wird nie im GPU-Speicher abgelegt.
- **`torch.compile`** (Nr. 5 gegen Nr. 6) bringt noch einmal den Faktor 1,2, der Speicher bleibt gleich.
- **Die MFU lässt sich zwischen fp32 und 16 Bit nicht vergleichen.** Nr. 1 erreicht 60 % der fp32-Spitze von 19,5 TFLOPS, alle anderen rechnen gegen die bf16-Spitze von 312 TFLOPS. Vergleichbar sind die Tokens/s. Am Ende nutzen wir ein Viertel der bf16-Spitze. Mit Batchgröße 4 ist die GPU noch nicht voll ausgelastet.
- **`GPTModelV2` ist dasselbe Modell:** Mit den OpenAI-Gewichten setzen V1 und V2 „The Elves are“ wörtlich gleich fort. Die größte Abweichung der Logits ist 0,075 und kommt von der GELU-Näherung.

Grenzen: ein Lauf pro Variante und 10 Messschritte, die Streuung zwischen Läufen ist nicht gemessen. Die Batchgröße ist mit 4 bewusst klein, damit auch fp32 auf kleinere GPUs passt.

## Ziele und Grenzen

Ziele:

- Die Logik aus `workbook_timo_lotr.ipynb` steckt in Klassen, und das Notebook rechnet danach genau dieselben Zahlen wie vorher.
- Ein Benchmark misst alle Schritte der Folie gleich: Tokens pro Sekunde, Spitzenspeicher und MFU.
- `GPTModelV2` ist aus `torch.nn`-Bausteinen gebaut, nutzt `scaled_dot_product_attention` und lädt die OpenAI-Gewichte.
- Das neue Notebook zeigt die Ergebnisse als Tabelle und Diagramm.

Nicht dabei:

- Notebooks der anderen und `workbook_timo.ipynb` aus Woche 1.
- Gemeinsamer Code, den andere Notebooks nutzen. `GPTModel`, `MultiHeadAttention`, `TransformerBlock`, `FeedForward`, `LayerNorm`, `create_dataloader`, `BPEDataset` und `Embeddings` bleiben unverändert.
- Training mit V2, bf16 oder `torch.compile` über `train()`. Woche 5 misst nur die Geschwindigkeit. Trainiert wird damit erst auf dem Cluster in Woche 6, zusammen mit DDP und dem Fortsetzen nach Abbruch.
- FlashAttention selbst implementieren. Das gehört zum Paper.
- Was Raschka zusätzlich einbaut, die Folie aber nicht verlangt: TF32, fused AdamW, ein auf 50.304 aufgefülltes Vokabular, die Suche nach der größten Batchgröße.

Als Fehlschlag gilt, auch wenn alles läuft:

- Das LOTR-Notebook rechnet nach dem Umbau andere Zahlen.
- Benchmark-Zahlen lassen sich nicht vergleichen, weil Aufwärmen oder Kompilieren in der Messung steckt, die Batchgrößen verschieden sind oder die MFU gegen falsche Spitzenwerte gerechnet ist.
- Logik steht nur im Notebook und hat keinen Test.
- Das Notebook zeigt mehr Code als Erklärung.

## Rahmenbedingungen

- torch 2.14.1, tiktoken 0.14.0 und wandb 0.28.1 sind fest, neue Laufzeit-Abhängigkeiten kommen nicht dazu. pandas und matplotlib gibt es nur in der dev-Gruppe und damit nicht auf dem Cluster, also nutzt sie nur das Notebook.
- Hyperparameter bekommen keine Default-Werte (Regel aus Woche 4).
- `train()` bleibt frei von W&B, damit die Tests ohne Konto laufen. Im Modus `disabled` läuft `wandb.init` ohne Konto und ohne Netz (lokal geprüft).
- Die Tests laufen in Sekunden auf der CPU, auch unter Windows. Den Flash-Kernel gibt es dort nicht.
- `torch.compile` hängt `_orig_mod.` vor alle Namen im `state_dict` (lokal geprüft). Unter Windows braucht der Standard-Compiler einen C++-Compiler, die Tests nehmen deshalb `backend="eager"`.
- In Colab kommt das Paket per `%pip install` aus einem Branch. Diese Zelle bleibt im Notebook, weil es das Paket vorher noch nicht gibt.
- Kein Buchtext in Repo, Tests oder Notebook-Ausgaben.
- Die T4 in Colab kann bf16 nicht in Hardware, und der Flash-Kernel von PyTorch braucht mindestens sm80 (A100-Generation). Aussagekräftig sind L4 oder A100 in Colab und L40S oder H100 auf DACHS.
- Der LOTR-Text liegt lokal unter `notebooks/data/The-lord-of-the-rings.txt`, so lässt sich der Umbau auf der CPU prüfen.

## Ansätze

### Aufbau der Notebooks

Timo hat Variante B gewählt.

| Ansatz | Vorteil | Nachteil |
|---|---|---|
| A: eine Klasse pro Experiment, `LotrExperiment(CONFIG).run()` | kürzestes Notebook | Zwischenschritte verschwinden in der Klasse, jedes Experiment braucht eine eigene Klasse |
| B (gewählt): Bausteine, eine Zeile pro Schritt | jeder Abschnitt behält seine Zelle und Ausgabe, die Bausteine taugen auch für Woche 6 | ein paar Zeilen mehr im Notebook |

### Modell aus `torch.nn`-Bausteinen

| Ansatz | Vorteil | Nachteil |
|---|---|---|
| Neue Klasse `GPTModelV2` neben `GPTModel` (Vorschlag) | `GPTModel` bleibt Ausgangsstand für den Benchmark und unverändert für die anderen Notebooks | zwei Modellklassen |
| `GPTModel` umbauen | eine Klasse | ändert gemeinsamen Code, und der Ausgangsstand geht verloren |

### Attention in V2

| Ansatz | Vorteil | Nachteil |
|---|---|---|
| Ein `nn.Linear` für Q, K und V, dann `F.scaled_dot_product_attention(is_causal=True)` (Vorschlag) | braucht keine Maske; der Kernel lässt sich auf `math` umschalten, so misst der Benchmark FlashAttention allein; die Gewichtsnamen bleiben nah an V1 | `scaled_dot_product_attention` ist eine Funktion, kein Modul |
| `nn.MultiheadAttention` | wörtlich „torch.nn“, ruft intern dieselbe Funktion | braucht trotz `is_causal` eine Maske mit T×T Einträgen |
| `nn.TransformerEncoderLayer` | am wenigsten eigener Code | Dropout sitzt an anderer Stelle als bei GPT-2, die OpenAI-Gewichte lassen sich schwer zuordnen |

### 16 Bit

Gemessen werden beide Varianten.

| Ansatz | Vorteil | Nachteil |
|---|---|---|
| bf16 per `autocast`, die Gewichte bleiben fp32 (Folie 12) | stabil, so trainiert man üblicherweise | spart weniger Speicher als reines bf16 |
| ganzes Modell in bf16 per `model.to(torch.bfloat16)` (Moodle-Skript, Raschka) | am wenigsten Speicher | AdamW rechnet die Updates in bf16 und verliert kleine Änderungen |

Auf der T4 nimmt der Benchmark statt bf16 mit `autocast` fp16 mit GradScaler, wie `train()` es heute schon kann.

### Benchmark

| Ansatz | Vorteil | Nachteil |
|---|---|---|
| Eigene Klasse mit denselben Messgrößen wie das Moodle-Skript (Vorschlag) | richtige Spitzenwerte, Aufwärmschritte, ganzer Trainingsschritt, Attention in den FLOPs | eigener Code |
| Moodle-Skript unverändert | vorgegeben | MFU etwa 4-mal zu hoch, kennt die L40S nicht, wärmt nicht auf, `thop` zählt die Attention nicht mit, braucht `thop` und `llms_from_scratch` |

## Teil 1: Notebook-Code in Klassen

### Bausteine

| Baustein | Modul | Aufgabe | Ersetzt im LOTR-Notebook |
|---|---|---|---|
| `Environment` | `infm_sota2/env.py` | Erkennt Colab oder lokal und bindet in Colab Drive ein. Kennt die Pfade für Daten, bleibende Checkpoints und laufende Läufe, dazu Gerät, GPU-Name und W&B-Modus. Methoden: `login_wandb()`, `check_versions()` und `describe()` für die W&B-Konfiguration. | Versionsvergleich aus Zelle 3, Zellen 4 und 7, `environment` aus Zelle 6 |
| `TextSplits` | `data/splits.py` | `from_book(...)` bereinigt, teilt und kürzt im Smoke-Test. `summary()` gibt die Bereinigungsstatistik als Text zurück. Dazu Loader für Training (fester Seed pro Lauf) und Auswertung sowie der Tokenizer. | Zellen 9 bis 11 |
| `TrainConfig`, `Trainer` | `training/trainer.py` | `TrainConfig` sammelt die Werte, die `train()` heute einzeln bekommt. `Trainer.fit(...)` baut den Optimizer, startet den W&B-Run, ruft `train()` auf, lädt den besten Checkpoint zurück ins Modell und kopiert ihn an den bleibenden Ort. | `run_training` und `keep_checkpoint` aus Zelle 13 |
| `ModelReport` | `training/report.py` | `add(name, model, context_size)` misst Test-Loss und Perplexity und erzeugt Texte mit festen Seeds. `add_texts(name, texts)` nimmt die Claude-Texte auf. `losses()` und `completions(reference_text)` liefern Tabellenzeilen samt Auswendiglern-Check. | `record` und `sample_completions` aus Zelle 13, Zellen 27 und 28 |
| `build_model(...)`, `load_pretrained(...)` | `model/` | Baut ein neues Modell mit festem Seed auf dem Gerät und lädt die OpenAI-Gewichte. `load_pretrained` setzt keinen Seed, damit das Nachtrainieren dieselben Zufallszahlen bekommt wie bisher. | `new_scratch_model` aus Zelle 13, Zelle 21 |

`losses()` und `completions()` liefern Listen von Dicts, aus denen das Notebook mit `pd.DataFrame(...)` eine Tabelle macht. So braucht `src/` kein pandas.
`CONFIG` bleibt im Notebook ein Dict wie heute und geht unverändert an W&B. Nur `TrainConfig` ist eine Dataclass. Sie ersetzt die lange Argumentliste von `train()`, Lernrate und Epochen bekommt `fit` pro Lauf.

Im Notebook bleiben die `%pip install`-Zelle, `SMOKE_TEST`, `CONFIG`, die Schleife der Lernraten-Suche (wenige Zeilen, damit die Suche sichtbar bleibt), die Plausibilitätsprüfung mit ln(50257), die Claude-Texte, die Erklärungen und das Fazit.

### Skizze

```python
env = Environment.detect(smoke_test=SMOKE_TEST)
data = TextSplits.from_book(env.data_dir, seq_len=CONFIG["seq_len"], **CONFIG["data"])
print(data.summary())

TRAIN = TrainConfig(...)  # feste Werte aus CONFIG
trainer = Trainer(env, wandb_config=CONFIG)
report = ModelReport(data, env.device, batch_size=..., seed=..., **CONFIG["generation"])

report.add("Untrainiert", build_model(GPTModel, CONFIG["scratch_model"], CONFIG["seed"], env.device), 256)

for lr in CONFIG["lr_search"]["learning_rates"]:
    model = build_model(GPTModel, CONFIG["scratch_model"], CONFIG["seed"], env.device)
    results[lr] = trainer.fit(model, data, TRAIN, learning_rate=lr, epochs=1,
                              name=f"lotr-scratch-lr{lr:g}", group="lotr-lr-search", checkpoint_name=None)

pd.DataFrame(report.losses())
```

Die genauen Signaturen legt der Umsetzer fest.

### Was gleich bleiben muss

- Die Zahlen bleiben gleich, weil Seeds und Reihenfolge der Zufallszahlen gleich bleiben: `torch.manual_seed` vor jedem neuen Modell, ein neuer Train-Loader mit festem Seed pro Lauf und ein Generator mit `seed + i` pro Text.
- Die gedruckten Zeilen behalten ihr Format. Die gespeicherten Ausgaben vom 12.11. stimmen dann weiter, bis das Notebook in Colab neu läuft.
- Geprüft wird mit `SMOKE_TEST` lokal auf der CPU, einmal vor und einmal nach dem Umbau. Alle Losses und Texte müssen übereinstimmen. Auf der CPU ist das Training deterministisch.
- [`docs/lotr-training.md`](lotr-training.md) (Abschnitte „Aufbau“ und „Code“) und die Projektstruktur in der README nennen danach die neuen Klassen.

## Teil 2: Woche 5

### `GPTModelV2`

`GPTModelV2` steht in `model/gpt_v2.py` und nimmt dieselbe Konfiguration (Dict) wie `GPTModel`.

- Es besteht aus `nn.Embedding` für Tokens und Positionen, `nn.Dropout` und den Blöcken. Jeder Block hat zwei `nn.LayerNorm` (eps 1e-5), die Attention, ein Feed-Forward aus `nn.Linear`, `nn.GELU(approximate="tanh")` und `nn.Linear` sowie Dropout auf dem Residual-Zweig wie V1. Am Ende kommen `nn.LayerNorm` und `out_head` ohne Bias.
- Die Attention rechnet `qkv = nn.Linear(d, 3d, bias=qkv_bias)`, dann `F.scaled_dot_product_attention(q, k, v, dropout_p=…, is_causal=True)` mit Dropout nur im Training, dann `out_proj = nn.Linear(d, d)`. Einen Masken-Buffer gibt es nicht.
- V1 rechnet die GELU exakt (`nn.GELU()`), das Original-GPT-2 nutzt die tanh-Näherung. V2 nimmt die tanh-Näherung, V1 bleibt, wie es ist.
- `convert_v1_state_dict(state)` macht aus einem `state_dict` im Format von V1 eines für V2. `W_query`, `W_key` und `W_value` (Gewichte und Bias) werden zu `qkv`, `scale` und `shift` heißen `weight` und `bias`, und die 12 Masken fallen weg. `load_pretrained` lädt damit die OpenAI-Gewichte strikt in V2.

### Benchmark

Die Klasse `Benchmark` steht in `infm_sota2/benchmark.py`.

- Sie bekommt Modellkonfiguration (GPT-2 Small), Batchgröße, Sequenzlänge, Aufwärmschritte, Messschritte und Gerät. Alle Varianten laufen mit denselben Werten.
- Eine `Variant` hat einen Namen, die Modellklasse, die Präzision (`fp32`, `fp16-mixed`, `bf16-mixed` oder `bf16`), den Attention-Kernel (`auto` oder `math`, nur bei V2) und das Backend für `torch.compile` (`None` heißt nicht kompilieren, im Smoke-Test `eager`, weil Windows für den Standard einen C++-Compiler braucht).
- `run(variant)` baut ein neues Modell mit AdamW und erzeugt zufällige Tokens mit festem Seed, weil der Inhalt die Geschwindigkeit nicht ändert. Dann laufen die Aufwärmschritte, in denen auch kompiliert wird, und danach die Messschritte. Ein Messschritt ist ein ganzer Trainingsschritt: Gradienten löschen, Forward, Loss, Backward (bei fp16 mit Scaler) und Optimizer-Schritt. Auf CUDA misst der Benchmark mit CUDA-Events, auf der CPU mit `time.perf_counter`.
- Das Ergebnis enthält Tokens/s, ms pro Schritt, Spitzenspeicher (`torch.cuda.max_memory_allocated`, auf der CPU leer), FLOPs pro Token und MFU. Geht einer Variante der Speicher aus, steht das im Ergebnis, und der Benchmark läuft mit der nächsten weiter. Zwischen den Varianten gibt er den Speicher frei.
- FLOPs pro Token beim Training: `6 · (Summe aller nn.Linear-Gewichte) + 12 · n_layers · seq_len · emb_dim`. Der erste Teil sind die Linear-Schichten inklusive `out_head`, der zweite die beiden Matrixprodukte der Attention über die volle Länge, wie im PaLM-Paper.
- Für die MFU braucht der Benchmark die Spitzenleistung der GPU. Die Tabelle nennt dichte Werte ohne Sparsity in TFLOPS, geprüft am 19.11.2026 gegen die NVIDIA-Datenblätter. Wo ein Datenblatt nur den Wert mit Sparsity nennt, steht hier die Hälfte.

  | GPU | FP32 | FP16/BF16 (Tensor Cores) |
  |---|---|---|
  | T4 | 8,1 | 65, nur FP16 |
  | L4 | 30,3 | 121 |
  | A100 | 19,5 | 312 |
  | L40S | 91,6 | 362,05 |
  | H100 SXM | 67 | 989,5 |
  | H100 PCIe | 51 | 756,5 |
  | H100 NVL | 60 | 835,5 |

  `fp32` rechnet gegen die Spalte FP32, alle 16-Bit-Varianten gegen FP16/BF16. Die GPU wird am Namen aus `torch.cuda.get_device_name()` erkannt. Dabei muss „L40S“ vor „L4“ geprüft werden, und bei der H100 zählt, ob „PCIe“ im Namen steht. Bei einer unbekannten GPU bleibt die MFU leer.

### Varianten in der Reihenfolge der Folie

| Nr. | Schritt | Modell | Präzision | Attention-Kernel | compile |
|---|---|---|---|---|---|
| 1 | Ausgangsstand | V1 | fp32 | eigener Code | nein |
| 2 | 16 Bit mit `autocast` | V1 | bf16-mixed (T4: fp16-mixed) | eigener Code | nein |
| 3 | 16 Bit, ganzes Modell | V1 | bf16 | eigener Code | nein |
| 4 | torch.nn ohne Flash | V2 | wie Nr. 2 | math | nein |
| 5 | torch.nn mit Flash | V2 | wie Nr. 2 | auto | nein |
| 6 | `torch.compile` | V2 | wie Nr. 2 | auto | ja |

Der Unterschied zwischen Nr. 4 und Nr. 5 ist allein FlashAttention, das passt zur Paper-Präsentation.
Dafür erlaubt der Benchmark bei Nr. 4 16-Bit-Matrixprodukte im Math-Kernel (`allow_fp16_bf16_reduction_math_sdp`). Von sich aus rechnet PyTorch dort bf16-Eingaben in fp32, dann wäre Nr. 4 künstlich langsam.

### Notebook `workbook_timo_attention_v2.ipynb`

- Oben stehen Umgebung, GPU und `SMOKE_TEST`. Im Smoke-Test läuft ein winziges Modell auf der CPU, die Zahlen sagen dann nichts aus.
- Jede Aufgabe der Folie bekommt einen Abschnitt mit kurzer Erklärung (was sich ändert, warum es schneller wird, welche Folie), ein bis zwei Zeilen Code und der Ergebniszeile.
- Ein Abschnitt lädt die OpenAI-Gewichte in V1 und V2 und lässt beide denselben Prompt ohne Zufall fortsetzen. Wegen der GELU-Näherung können die Texte leicht abweichen.
- Am Ende stehen die Tabelle aller Varianten, zwei Balkendiagramme (Tokens/s und Speicher, direkt mit pandas gezeichnet), die MFU und das Fazit.

Vor dem Merge nach `main` gibt es die neuen Klassen nur auf `week/05`. Für einen Lauf in Colab davor `BRANCH` vorübergehend auf `week/05` setzen und vor dem Speichern wieder auf `main`, das prüft `tests/test_notebooks.py`.

Nach dem Lauf kommen die Ergebnisse in diese Datei, und die README nennt `gpt_v2.py` und `benchmark.py` in der Projektstruktur.

## Teil 3, optional: KV-Cache

Erst wenn Teil 2 fertig ist. `GPTModelV2` speichert beim Erzeugen Keys und Values der bisherigen Tokens, und `generate` nutzt sie. Der Benchmark misst dann Tokens/s beim Erzeugen mit und ohne Cache. Als Vorlage dient `gpt_with_kv_mha.py` aus Raschkas Ordner `ch04/10_kv-sharing`.

## Tests

Alle laufen auf der CPU.

- V2 liefert mit umgerechneten V1-Gewichten dieselben Logits wie V1, wenn der Test bei V1 die GELU gegen die tanh-Näherung tauscht. Außerdem ignoriert V2 zukünftige Tokens, die Umrechnung lädt strikt, und die Kernel `math` und `auto` liefern dasselbe Ergebnis.
- Die FLOPs-Formel stimmt an einem winzigen Modell mit einer Handrechnung überein. Die GPU-Erkennung hält die L40S nicht für eine L4. `run` liefert an einem winzigen Modell auf der CPU endliche Tokens/s und eine leere MFU. Ein Speicherfehler landet im Ergebnis, statt den Lauf abzubrechen.
- Die Bausteine aus Teil 1: `Trainer.fit` mit W&B im Modus `disabled` an einem winzigen Modell, `TextSplits` mit einem selbst geschriebenen Mini-Buch, die Zeilen von `ModelReport`, `Environment` lokal mit temporären Ordnern und der Versionsvergleich mit erfundenen Anforderungen.

## Reihenfolge und Branches

1. Teil 1 auf `feature/notebook-classes`, abgezweigt von `week/05`, mit Pull Request nach `week/05`.
2. Teil 2 und eventuell Teil 3 auf `feature/attention-v2`, abgezweigt vom Stand nach Teil 1, ebenfalls mit Pull Request nach `week/05`.

Commits und Pull Requests gibt es erst nach Timos OK. Nach jedem Teil laufen ruff, pyright, die Tests und der `SMOKE_TEST` der betroffenen Notebooks. Den echten Lauf auf der GPU macht Timo.

## Entscheidungen

Von Timo entschieden:

- Erst der Umbau, dann der Code für Woche 5.
- Nur eigener Code und eigene Notebooks, die anderen bleiben unberührt.
- Variante B, also Bausteine mit einer Zeile pro Schritt.
- Alles soll nachvollziehbar und präsentierbar sein, ohne Over-Engineering.
- Als Paper FlashAttention, später.

Mein Vorschlag, bitte prüfen:

- `GPTModelV2` als neue Klasse, `GPTModel` bleibt unverändert.
- Attention in V2 mit eigenem Q/K/V-Linear und `F.scaled_dot_product_attention` statt `nn.MultiheadAttention`.
- V2 nutzt die GELU mit tanh-Näherung wie das Original-GPT-2.
- 16 Bit in beiden Formen messen, mit `autocast` und als ganzes Modell in bf16.
- Eine eigene Benchmark-Klasse statt des unveränderten Moodle-Skripts, mit zufälligen Tokens, Aufwärmschritten und ganzem Trainingsschritt.
- Der Vergleich der Kernel `math` und `auto` als eigene Variante.
- Training mit V2, bf16 und `compile` erst in Woche 6. `train()` bleibt in Woche 5 unverändert.
- TF32, fused AdamW, aufgefülltes Vokabular und die Suche nach der größten Batchgröße lassen wir weg.
- Der KV-Cache ist optional und kommt zuletzt.
- Zwei Branches mit je einem Pull Request.

Dem Umsetzer überlassen sind die genauen Signaturen, die Aufteilung innerhalb der Module, die Texte im Notebook und die Einzelheiten der Tests.
