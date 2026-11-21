# Woche 6: Pre-Training auf dem Cluster

Woche 6, Aufgabe *Scaling Pre-Training*. Wir trainieren GPT-2 auf dem Cluster DACHS, mit den Optimierungen aus Woche 5 und mit DDP über mehrere Knoten. Alles soll für jede Person aus der Gruppe gleich funktionieren.

Stand: Entwurf vom 21.11.2026, noch nicht umgesetzt.

## Aufgabe

Folie 46:

- Setup the Cluster environment on BW-UniCluster and DACHS: activate entitlements, login via SSH, do the SLURM tutorial and write a start script.
- Run a hyper-parameter optimization (batch-size + LR) on GPT-2 Small, 1 epoch.
- Run a full pre-training of GPT-2 Small, multi-node (don't use too many nodes, scheduler favors small jobs). Dataset?
- Increase to GPT-Medium++ and train again.

Woche 6 ist geschafft, wenn wir auf den Cluster kommen und ein Testlauf zeigt, dass alles funktioniert. Große Läufe und gute Zahlen sind nicht das Ziel. Kein Lauf dauert länger als 2 Stunden.

| Punkt der Folie | Umsetzung |
|---|---|
| Entitlements | Jede Person braucht das bwUniCluster-Entitlement der HS Offenburg und registriert in bwIDM den Dienst DACHS mit 2FA und Service-Passwort. Timo hat das erledigt. Für den bwUniCluster kommt der Dienst „bwUniCluster 3.0“ dazu, klappt das während der Abschaltung nicht, dann danach. |
| Login per SSH | PowerShell mit dem eingebauten `ssh`, Anleitung in `docs/dachs.md`. Den bwUniCluster erreichen wir erst nach dem 04.12. |
| Slurm-Tutorial und Startskript | erst ein paar Befehle und ein Mini-Job aus der Slurm-Kurzanleitung, dann unsere Jobskripte in `slurm/` |
| Suche nach Batchgröße und Lernrate | Lauf Nr. 4 |
| Pre-Training auf mehreren Knoten | Lauf Nr. 6 auf 2 Knoten |
| Datensatz | der bereinigte Text von *The Lord of the Rings* aus Woche 4 |
| GPT-Medium++ | Lauf Nr. 7; GPT-2 Large nur, wenn danach Zeit bleibt |

## Ziele und Grenzen

Ziele:

- Jede Person aus der Gruppe kommt per SSH auf DACHS, richtet die Umgebung mit einem Befehl ein und startet die Läufe mit denselben Befehlen wie alle anderen.
- Ein Smoke-Lauf ohne Buchtext zeigt jeder Person in wenigen Minuten, ob ihre Einrichtung funktioniert.
- Ein Testlauf zeigt, dass der Cluster dieselben Ergebnisse liefert wie Colab in Woche 4.
- Das Training nutzt die Optimierungen aus Woche 5, also bf16 mit `autocast`, `GPTModelV2` mit FlashAttention und `torch.compile`, dazu fused AdamW.
- Das Training läuft mit DDP auf mehreren Knoten, und wir messen Speedup und Effizienz.
- Jeder Lauf schreibt eine JSON-Datei mit seinen Ergebnissen und allen Einstellungen.

Nicht dabei:

- Läufe auf dem bwUniCluster. Er ist vom 23.11. bis 04.12.2026 abgeschaltet. Die Skripte sind vorbereitet, getestet wird danach.
- Fortsetzen nach Abbruch, ein Datenformat für Milliarden Tokens und ein Lernraten-Plan mit Warmup. Das braucht erst das lange Pre-Training in Woche 9.
- FSDP, Tensor- und Pipeline-Parallelismus. Laut Folie 35 kommt das in diesem Kurs nicht vor.
- TF32, ein auf 50.304 aufgefülltes Vokabular und der KV-Cache (Begründung unter „Ansätze“).
- Weak Scaling und mehrere Seeds pro Einstellung.
- Colab. Dort bindet jede Person ihr eigenes Drive ein und legt den Ordner an wie in `docs/lotr-training.md` beschrieben. Daran ändert Woche 6 nichts.
- Die Notebooks der anderen und `workbook_timo_attention_v2.ipynb`.

Als Fehlschlag gilt, auch wenn alles läuft:

- Ein Schritt klappt nur mit Timos Konto, Drive oder Pfaden.
- Das LOTR-Notebook rechnet nach dem Umbau andere Zahlen.
- Ein Job hängt, statt abzubrechen, oder ein gescheiterter Lauf fehlt still in der Auswertung.
- In den gemessenen Tokens pro Sekunde steckt Kompilieren oder Auswerten.
- Ein Checkpoint lässt sich nicht in ein normales `GPTModel` oder `GPTModelV2` laden.
- Ein Ergebnis steht ohne die Einstellungen da, mit denen es entstanden ist.
- Buchtext landet im Repo, in Logs oder in Notebook-Ausgaben.

## Rahmenbedingungen

- **DACHS, Partition `gpu1`:** 45 Knoten mit je einer NVIDIA L40S (48 GB) und 48 Kernen, verbunden über InfiniBand HDR100. Ein Job darf bis zu 16 Knoten und 72 Stunden nutzen. Ohne eigene Angabe bekommt er 30 Minuten und 5 GB Hauptspeicher, und die Partition muss immer angegeben werden. Die Partitionen `gpu8` (ein Knoten mit 8 H100) und `gpu4` (AMD) nutzen wir nicht.
- **Zugang:** nur aus dem BelWü-Netz, von außen über das VPN der HS Offenburg ohne Split-Tunneling. Benutzername `of_<HSO-Kürzel>`, Host `dachs-login.hs-esslingen.de`. Auf den Login-Knoten läuft nur Kurzes: Einrichtung, Downloads, Jobs abschicken.
- **Erst Job 1 zeigt,** ob der Treiber CUDA 13 kann (torch 2.14.1 aus `uv.lock` bringt die CUDA-13.0-Laufzeit mit), ob die Rechenknoten ins Internet kommen, ob ein C-Compiler für `torch.compile` da ist und ob NCCL InfiniBand nutzt oder auf TCP ausweicht. Kann der Treiber kein CUDA 13, entscheidet Timo, wie es weitergeht. Ein anderer CUDA-Index in `pyproject.toml` gilt für alle Linux-Rechner.
- **Pakete:** torch 2.14.1, tiktoken 0.14.0 und wandb 0.28.1 sind fest. Neue Laufzeit-Abhängigkeiten kommen nicht dazu, TOML liest `tomllib` aus der Standardbibliothek.
- **Regeln aus Woche 4 und 5:** keine Default-Werte für Hyperparameter, `train()` bleibt frei von W&B, Tests laufen auf der CPU in Sekunden und auch unter Windows, kein Buchtext im Repo, Notebooks ändert nur ein `nbformat`-Skript.
- **Für alle gleich:** In Skripten und Konfigurationen stehen keine Namen und keine Pfade einer Person. Was sich pro Person unterscheidet, kommt aus `$USER`, `$HOME`, dem Ordner des eigenen Klons oder einer Umgebungsvariable. W&B ist kein Muss.
- **Buchtext:** Er ist urheberrechtlich geschützt und kommt nicht ins Repo. Jede Person, die mit dem Buch trainiert, legt ihre eigene Kopie in ihren Datenordner oder nutzt einen gemeinsamen Workspace der Gruppe.
- **Git:** Der Umsetzer committet und pusht mit Timo als Autor, ohne Co-Autor- und Session-Zeilen, auf Feature-Branches. Ergebnisse von Läufen committet, wer sie erzeugt hat. Der Cluster holt den Code von GitHub, was dort laufen soll, muss also vorher gepusht sein.
- **Wer was macht:** Der Umsetzer kommt nicht auf den Cluster und testet lokal auf der CPU. Die ersten Jobs startet Timo, danach jede Person, die mitarbeitet.

## Ansätze

### Wo DDP und die Optimierungen hinkommen

Timo hat Weg A gewählt.

| Ansatz | Vorteil | Nachteil |
|---|---|---|
| A (gewählt): `train()`, `Trainer`, `TextSplits` und `Environment` erweitern | Notebook und Cluster trainieren in derselben Schleife. Weicht ein Loss ab, liegt das an Hardware, Präzision oder DDP. | ändert Code, der schon geprüfte Zahlen liefert; abgesichert durch den Smoke-Test des LOTR-Notebooks vor und nach dem Umbau |
| B: eigene Trainingsschleife für den Cluster | vorhandener Code bleibt unberührt | eine zweite Schleife, die mit der Zeit von `train()` abweichen kann |
| C: PyTorch Lightning oder HF Accelerate | übernimmt DDP und Slurm fast ganz | neue Laufzeit-Abhängigkeit, versteckt genau das, was die Folien erklären |

### Prozesse starten

| Ansatz | Vorteil | Nachteil |
|---|---|---|
| `torchrun`, einmal pro Knoten über `srun` (Vorschlag) | empfiehlt Folie 35; derselbe Code läuft lokal mit `torchrun --nproc-per-node=2` auf der CPU | ein Werkzeug mehr im Jobskript |
| nur `srun`, Ränge aus den Slurm-Variablen (Princeton-Repo) | kein `torchrun` | ohne Slurm kaum zu testen |

### Suche nach Batchgröße und Lernrate

| Ansatz | Vorteil | Nachteil |
|---|---|---|
| Slurm-Job-Array, eine Kombination pro Aufgabe (Vorschlag) | läuft parallel, jede Aufgabe ist ein kleiner Job | 12 Jobs in der Queue |
| eine Schleife in einem Job | nur ein Job | läuft nacheinander |
| W&B-Sweep | Auswertung direkt in W&B | braucht Internet auf den Rechenknoten und W&B-Zugang für alle |

### Einstellungen

| Ansatz | Vorteil | Nachteil |
|---|---|---|
| eine TOML-Datei pro Experiment im Repo (Vorschlag) | versioniert, für alle gleich, geht vollständig an W&B, keine versteckten Defaults | eine Datei pro Experiment |
| nur Argumente auf der Kommandozeile | nichts zu pflegen | die Werte stehen nur im Jobskript oder in der Shell-History |

### Optimierungen aus Woche 5

Timo hat entschieden, die drei Schritte von Folie 20 ins Training zu übernehmen und dazu zwei Punkte aus Raschkas Liste.

| Optimierung | Im Training | Grund |
|---|---|---|
| bf16 mit `autocast` | ja | Nr. 2 im Benchmark aus Woche 5, die Gewichte bleiben in fp32 |
| `GPTModelV2` mit `scaled_dot_product_attention` | ja | Nr. 5, auf der L40S läuft damit FlashAttention |
| `torch.compile` | ja | Nr. 6 |
| fused AdamW | ja | PyTorch rechnet den Optimizer-Schritt in zusammengefassten Kernels, die Formel bleibt gleich |
| größte Batchgröße suchen | ja, mit dem Benchmark | liefert die Obergrenze für die Suche |
| TF32 | nein | Die Matrixprodukte im Training laufen mit bf16 ohnehin in 16 Bit. TF32 würde nur die Auswertung in fp32 schneller und ungenauer machen. |
| Vokabular auf 50.304 auffüllen | nein | Die OpenAI-Gewichte passen dann nicht mehr ohne Umbau, und das Modell ist nicht mehr dasselbe wie in Woche 4. |
| KV-Cache | nein | beschleunigt nur das Erzeugen von Text |
| ganzes Modell in bf16 | nein | AdamW verliert in reinem bf16 kleine Updates (Woche 5) |

Auf der A100 war die Kombination aus den ersten drei Zeilen (Nr. 6) 2,6-mal so schnell wie 16 Bit mit `GPTModel` (Nr. 2). Nr. 2 entspricht am ehesten dem bisherigen Training.

### LOTR-Notebook

Timo hat entschieden: `workbook_timo_lotr.ipynb` bleibt bei den Einstellungen von Woche 4. Es dokumentiert die Ergebnisse vom 12.11. Lauf Nr. 2 und Nr. 3 vergleichen beide Einstellungen auf derselben GPU.

## Teil 1: Einstieg für alle

### Anleitung

`docs/dachs.md` ist der Einstieg für jede Person aus der Gruppe, die README verweist im Abschnitt „Auf dem Cluster“ darauf. Die Anleitung führt durch diese Schritte:

1. **Konto:** bwUniCluster-Entitlement der HS Offenburg, Dienst DACHS in bwIDM, 2FA und Service-Passwort. Für den bwUniCluster zusätzlich der Dienst „bwUniCluster 3.0“.
2. **Verbindung:** VPN ohne Split-Tunneling, dann `ssh of_<HSO-Kürzel>@dachs-login.hs-esslingen.de`, erst OTP, dann Service-Passwort. Beim ersten Mal den Fingerabdruck prüfen (ED25519: `SHA256:X9eRJepYD3da3BM1pgiWxnvRc/Pt5eBLUr18tDUsZjU`). Ein Eintrag in `~/.ssh/config` verkürzt den Aufruf auf `ssh dachs`. Wer einen SSH-Schlüssel in bwIDM hinterlegt, spart Abfragen bei `scp`.
3. **Einrichtung:** Repo klonen und einmal `bash slurm/setup.sh` ausführen (siehe unten).
4. **Slurm kennenlernen:** `sinfo_t_idle`, `squeue -u $USER`, ein Mini-Job mit `srun -p gpu1 --gres=gpu:1 -t 5 nvidia-smi`, danach `sbatch`, `squeue --start`, `scancel` und `sacct`. Dazu Job-Arrays, weil die Suche eins ist.
5. **Erster eigener Job:** der Smoke-Lauf mit `configs/smoke.toml`, erst auf einem Knoten, dann auf zwei.
6. **Läufe mit dem Buch:** Buchtext in den Datenordner legen, dann die Läufe aus Teil 2 mit denselben Befehlen wie alle anderen. Logs lesen, Ergebnisse per `scp` holen oder auf dem Cluster committen.
7. **bwUniCluster:** die Unterschiede für später. Der Host heißt `uc3.scc.kit.edu`, die Partitionen heißen anders, und die GPU-Knoten haben 4 GPUs.

### `slurm/setup.sh`

Jede Person führt das Skript einmal auf dem Login-Knoten aus, aus dem eigenen Klon. Es ändert nichts außerhalb des Klons und des eigenen Caches und lässt sich gefahrlos wiederholen.

- Fehlt uv oder ist es älter als 0.13, installiert es uv in `~/.local/bin`.
- `UV_NO_DEV=1 uv sync --no-dev --locked`, dann den tiktoken-Cache unter `~/.cache/tiktoken` füllen und den Ordner `logs/` anlegen.
- Es schreibt das Mini-Buch in den Datenordner (siehe „Smoke-Tests“).
- Es prüft, ob Buchtext und OpenAI-Gewichte da sind und ob W&B angemeldet ist. Fehlt etwas, nennt es den Befehl, der das behebt. Ohne W&B-Login zeigt es, wie man mit `WANDB_MODE=disabled` trotzdem trainiert.
- Am Ende steht, wofür die Einrichtung reicht: nur für den Smoke-Lauf oder auch für die Läufe mit dem Buch.

### Datenordner

Eingaben sind der Buchtext, die OpenAI-Gewichte und das Mini-Buch. Sie liegen im Ordner `data/` des eigenen Klons, der in der `.gitignore` steht. Zeigt die Umgebungsvariable `SOTA2_DATA_DIR` auf einen anderen Ordner, gilt der. So kann die Gruppe Buchtext und Gewichte einmal in einem gemeinsamen Workspace ablegen (`ws_allocate`, Lesezugriff mit `ws_share`) statt in jedem Klon. Ob sie das will, entscheidet sie auf dem Cluster. Ausgaben wie Checkpoints, Logs und Ergebnisse liegen immer im eigenen Klon.

## Teil 2: Smoke-Tests

| Ebene | Wer, wann | Braucht | Prüft |
|---|---|---|---|
| Unittests | CI bei jedem Push, lokal vor jedem Commit | nichts | die Bausteine, auch DDP mit zwei CPU-Prozessen |
| Smoke-Lauf `configs/smoke.toml` | jede Person nach der Einrichtung und vor jedem neuen Jobskript, auf dem Cluster mit 1 und 2 Knoten, lokal mit `torchrun --nproc-per-node=2` | nur das Mini-Buch | die ganze Kette: Slurm, `torchrun`, DDP, W&B-Modus, Checkpoint und JSON. Bestanden heißt: Der Job endet ohne Fehler, der Validierungs-Loss sinkt, und die JSON-Datei ist da. |
| Notebook-Smoke-Test | wer gemeinsamen Code ändert, einmal vorher und einmal nachher | den Buchtext und `nbclient` | Das LOTR-Notebook rechnet dieselben Zahlen wie vorher. |

**Mini-Buch:** `write_mini_book` aus `tests/helpers.py` zieht nach `infm_sota2/data/` um und bekommt einen Aufruf über die Kommandozeile. Tests, `setup.sh` und der Smoke-Lauf nutzen dieselbe Funktion. Das Mini-Buch ist selbst geschrieben, bei allen gleich und frei von Urheberrechten. `configs/smoke.toml` zeigt darauf und auf ein winziges Modell, das in Sekunden trainiert.

**Notebook-Smoke-Test:** Das Hilfsskript aus Woche 5 kommt als `scripts/smoke_notebook.py` ins Repo. Es setzt `SMOKE_TEST = True`, führt das Notebook aus und schreibt alle Ausgaben in eine Datei. Ein `diff` der Dateien von vorher und nachher zeigt jede geänderte Zahl. Für den Umbau in Woche 6 lädt Timo den Buchtext in die Sitzung des Umsetzers hoch. Die Datei liegt dort in `notebooks/data/` und kommt nicht ins Repo.

## Teil 3: Läufe

Gemeinsam für alle Läufe:

- **Daten wie in Woche 4:** Erzählteil von `PROLOGUE` bis `APPENDIX A`, Aufteilung 80/10/10 an Absatzgrenzen, Fenster von 256 Tokens ohne Überlappung, Seed 123.
- **Modell:** GPT-2 Small mit `context_length` 256, das sind 163 Mio. Parameter, weil Ein- und Ausgabe-Embedding nicht geteilt werden.
- **Training:** AdamW mit Weight Decay 0,1, Gradient Clipping 1,0, Dropout 0,1 und konstanter Lernrate wie in Woche 4.
- **Auswertung** in fp32 auf dem ganzen Validierungs- bzw. Testteil, auf dem Modell ohne DDP und ohne `compile`.
- **Texte** wie in Woche 4: „The Elves are“, 3 Stück, 40 neue Tokens, Temperature 0,8, Top-k 40, Seeds 123 bis 125.
- **Batchgröße** meint immer die Gesamt-Batchgröße über alle GPUs.
- **Seed:** Alle Ränge starten mit demselben Seed, also auch mit denselben Anfangsgewichten.
- **Testteil:** Suche und Skalierung messen keinen Test-Loss und erzeugen keine Texte. So fließt der Testteil nicht in die Auswahl ein, wie in Woche 4.

| Nr. | Lauf | Modell und Einstellungen | Knoten | Wozu |
|---|---|---|---|---|
| 1 | Benchmark | kein Training. (a) die sechs Varianten aus Woche 5: GPT-2 Small, Batch 4, 1024 Tokens, 3 Aufwärm- und 10 Messschritte. (b) Batchgrößen 8, 16, 32, 64, 128, 256 und 512 mit `GPTModelV2`, bf16 und `compile` bei 256 Tokens, bis zum ersten Speicherfehler | 1 | GPU und Treiber funktionieren; L40S neben A100; Obergrenze der Batchgröße |
| 2 | Woche-4-Einstellungen | GPT-2 original nur messen. Dazu Small von Grund auf mit `GPTModel`, fp16, ohne `compile`, AdamW ohne fused, Lernrate 4e-4, Batch 8, 4 Epochen, Prüfung alle 100 Schritte, bester Checkpoint | 1 | Test-Loss 3,63 für GPT-2 original und ungefähr 4,35 von Grund auf: Der Cluster rechnet wie Colab |
| 3 | Optimiert | wie Nr. 2, aber `GPTModelV2`, bf16, `compile` und fused AdamW, ohne GPT-2 original | 1 | Wie viel schneller wird das echte Training, und bleibt der Loss gleich? |
| 4 | Suche | Small optimiert, 1 Epoche, Lernrate 1e-4, 3e-4, 1e-3 oder 3e-3 kombiniert mit Batch 8, 32 oder 128. 12 Aufgaben im Job-Array, höchstens 4 gleichzeitig, kein Checkpoint | je 1 | beste Kombination nach Validierungs-Loss (Folie 46) |
| 5 | Skalierung | Small optimiert, Batch 32 mit der besten Lernrate der Suche für Batch 32, 2 Epochen, Prüfung alle 10 Schritte, kein Checkpoint. Pro GPU 32, 16 und 8 | 1, 2, 4 | Speedup und Effizienz (Folie 5); fast gleicher Loss bei allen Knotenzahlen prüft DDP |
| 6 | Hauptlauf | Small optimiert mit der besten Kombination der Suche, 20 Epochen, Prüfung alle 100 Schritte, bester Checkpoint | 2 | Pre-Training auf mehreren Knoten (Folie 46); Test-Loss und Texte neben Woche 4 |
| 7 | GPT-2 Medium | 24 Schichten, 1024 Dimensionen, 16 Köpfe, rund 405 Mio. Parameter. Halbe Lernrate von Nr. 6, sonst wie Nr. 6 | 2 | „Increase to GPT-Medium“ (Folie 46) |
| (8) | GPT-2 Large, optional | 36 Schichten, 1280 Dimensionen, 20 Köpfe, rund 840 Mio. Parameter. Lernrate 2,5/6 der Lernrate von Nr. 6, sonst wie Nr. 6 | 2 | nur, wenn nach Nr. 7 Zeit bleibt |

Zu den Zahlen:

- **Gitter der Suche:** Die Lernraten liegen grob logarithmisch zwischen 1e-4 und 3e-3. Batch 128 ist ein Platzhalter. Passt er laut Nr. 1 nicht in den Speicher, nehmen wir die größte Zweierpotenz, die passt. Eine Epoche hat bei Batch 8 noch 261 Schritte, bei Batch 128 nur 16. Kleine Batches gewinnen deshalb nach einer Epoche fast sicher. Das ist der Zielkonflikt von Folie 26 und gehört so ins Fazit.
- **Eine zu hohe Lernrate** kann den Loss auf NaN treiben. Der Lauf bricht dann laut ab und erscheint in der Auswertung als gescheitert.
- **20 Epochen im Hauptlauf,** weil der Validierungs-Loss in Woche 4 nach 4 Epochen noch fiel. Diesmal sehen wir, ab wann das Modell anfängt auswendig zu lernen. Gespeichert wird der Stand mit dem besten Validierungs-Loss.
- **Lernraten für Medium und Large** folgen dem Verhältnis im GPT-3-Paper (Tabelle 2.1): 6e-4 für 125 Mio., 3e-4 für 350 Mio. und 2,5e-4 für 760 Mio. Parameter.
- **Die beste Kombination** der Suche trägt ein, wer die Suche auswertet, in die Konfiguration von Nr. 6, und committet sie.
- **Rechenzeit**, gerechnet, nicht gemessen: Eine Epoche hat 536.346 Tokens. Mit rund 100.000 Tokens/s auf einer L40S dauert sie Sekunden. Jeder Lauf braucht samt Kompilieren und Warten auf den Start nur Minuten, alle zusammen weit unter einer GPU-Stunde.

Erwartungen, vor dem Messen festgehalten:

- **Nr. 2:** GPT-2 original kommt auf 3,63. Die Auswertung läuft in fp32 auf denselben Fenstern, Abweichungen sollten erst in der dritten Nachkommastelle auftauchen. Von Grund auf kommt der Lauf ungefähr auf 4,35, aber nicht genau, weil die GPU nicht bitgenau gleich rechnet. Nr. 2 läuft in Schritt 2 noch mit dem heutigen Loader und sieht die Daten deshalb in derselben Reihenfolge wie in Woche 4. Nach dem Umbau mischt der `DistributedSampler` ab der zweiten Epoche anders.
- **Nr. 3:** ähnlicher Test-Loss wie Nr. 2, aber mehrfach so viele Tokens/s.
- **Nr. 5:** Der Speedup bleibt deutlich unter der Zahl der Knoten. Pro Schritt gehen rund 650 MB Gradienten (163 Mio. Parameter in fp32) über HDR100 mit 12,5 GB/s. Das dauert bei 2 Knoten etwa 50 ms, die Rechnung bei 16 Fenstern pro GPU grob geschätzt 35 ms. DDP überlappt einen Teil der Übertragung mit dem Backward-Pass, der Rest ist der serielle Anteil aus Folie 6.

## Teil 4: Code

### Änderungen an vorhandenen Bausteinen (Weg A)

**`train()` in `training/loop.py`:**

- Statt `use_amp` gibt es `precision` mit den Werten `fp32`, `fp16-mixed` und `bf16-mixed`. `fp16-mixed` rechnet wie bisher mit `autocast` in float16 und GradScaler und braucht CUDA. `bf16-mixed` rechnet mit `autocast` in bfloat16 ohne GradScaler und läuft auch auf der CPU, so lässt es sich lokal testen.
- Neu ist `eval_model`, das Modell ohne DDP und ohne `compile`. `evaluate()` und `save_checkpoint()` nutzen es. Checkpoints haben damit dieselben Schlüssel wie in Woche 4, und die Auswertung löst kein neues Kompilieren aus.
- Ist der Sampler ein `DistributedSampler`, ruft `train()` zu Beginn jeder Epoche `set_epoch(epoch)` auf. Ohne den Aufruf hätte jede Epoche dieselbe Reihenfolge.
- `train()` misst die Zeit der Trainingsschritte zwischen zwei Prüfungen, ohne die Prüfung selbst. `log` bekommt zusätzlich `throughput/tokens_per_second` mit den Tokens dieses Prozesses. `TrainResult` bekommt `tokens_per_second`: den Median aller Intervalle außer dem ersten, in dem kompiliert wird. Bei weniger als zwei Intervallen ist der Wert `None`.
- Abbruch bei NaN, Clipping und Speichern des besten Checkpoints bleiben, wie sie sind.

**`TrainConfig` und `Trainer.fit` in `training/trainer.py`:**

- `TrainConfig` bekommt `precision` statt `use_amp`, dazu `compile_backend` (`None` heißt: nicht kompilieren) und `fused_adamw`. `batch_size` ist die Gesamt-Batchgröße.
- Unter `torchrun` nimmt `fit` immer `distributed_train_loader`, auch mit nur einem Prozess. So sehen 1, 2 und 4 Knoten in jedem Schritt dieselben Fenster (siehe „Messgrößen“). Ohne `torchrun`, also im Notebook, bleibt es bei `train_loader`.
- Pro Prozess ist die Batchgröße `batch_size / world_size`. Geht das nicht auf, bricht `fit` mit `ValueError` ab.
- In DDP packt `fit` das Modell erst ab zwei Prozessen, mit `broadcast_buffers=False`, weil die Masken von `GPTModel` konstant sind. Ein einzelner Prozess ohne DDP ist die Basis für den Speedup. Danach kompiliert `fit` das Modell, wenn `compile_backend` gesetzt ist, so wie PyTorch es für DDP empfiehlt: erst DDP, dann `compile`.
- AdamW bekommt `fused=fused_adamw`. Fused verlangt CUDA, auf der CPU gibt es `ValueError`.
- W&B-Run, Ausgaben, Checkpoint und die Kopie an den bleibenden Ort gibt es nur auf Rang 0. Geloggte Tokens und Tokens/s rechnet `fit` auf alle Prozesse hoch.

**`TextSplits` in `data/splits.py`:** neue Methode `distributed_train_loader(batch_size, seed, rank, world_size)` mit `DistributedSampler(shuffle=True, seed=seed, drop_last=True)` und einem DataLoader mit `drop_last=True`. `train_loader` bleibt unverändert.

**`Environment` in `env.py`:**

- kennt `rank`, `local_rank`, `world_size` und weiß, ob `torchrun` den Prozess gestartet hat (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`). Das Gerät ist dann `cuda:<local_rank>`.
- Ist `WANDB_MODE` gesetzt (`online`, `offline` oder `disabled`), gilt dieser Modus, ein anderer Wert ergibt `ValueError`. So trainiert auch, wer keinen W&B-Zugang hat.
- Ist `SOTA2_DATA_DIR` gesetzt, ist das der Datenordner, sonst `data/` unter dem Arbeitsordner. Colab bleibt, wie es ist.
- Die Prozessgruppe (NCCL auf CUDA, Gloo auf der CPU) startet das Einstiegsskript, nicht `Environment.detect`.

**`model/pretrained.py`:** `GPT2_MEDIUM_CONFIG` und `GPT2_LARGE_CONFIG`, sonst wie `GPT2_SMALL_CONFIG`.

**`benchmark.py`:** Die sechs Varianten von Folie 20 stehen bisher nur im Notebook. Sie kommen zusätzlich als Funktion ins Modul, damit der Job sie nutzen kann. Das Notebook bleibt unverändert.

**`data/`:** `write_mini_book` zieht aus `tests/helpers.py` hierher, mit Aufruf über die Kommandozeile. `tests/helpers.py` importiert es von dort.

**LOTR-Notebook:** Das `nbformat`-Skript ändert nur die Zelle mit `TrainConfig`: `precision` (`fp16-mixed` auf der GPU, sonst `fp32`), `compile_backend=None` und `fused_adamw=False`. Die gespeicherten Ausgaben bleiben. Der Notebook-Smoke-Test vor und nach dem Umbau muss dieselben Losses und Texte liefern, nur die Zeile mit `environment` darf sich unterscheiden.

### Neu: Paket `infm_sota2.cluster`

| Modul | Aufgabe |
|---|---|
| `config` | liest die TOML-Datei, bricht bei fehlenden oder unbekannten Schlüsseln ab und wählt in der Suche die Kombination zu `--grid-index` |
| `check` | gibt Rechner, GPU, Treiber, CUDA, torch, NCCL, bf16, einen Test des Flash-Kernels, C-Compiler, Internet, tiktoken-Cache und den Datenordner aus; mit mehreren Prozessen dazu ein `all_reduce` über alle Knoten |
| `benchmark` | Lauf Nr. 1: Tabellen ausgeben, JSON schreiben |
| `train` | Smoke-Lauf und Läufe Nr. 2 bis 8: Konfiguration lesen, Modell bauen, auf Wunsch GPT-2 original messen, `Trainer.fit`, auf Rang 0 `ModelReport` und JSON |
| `results` | JSON schreiben und lesen, Speedup und Effizienz rechnen; gescheiterte Läufe bleiben sichtbar |

`train` fängt nur `FloatingPointError` und `torch.OutOfMemoryError` ab. Dann schreibt es eine JSON-Datei mit dem Fehler und ohne Loss und bricht danach mit Fehler ab. So taucht eine zu hohe Lernrate oder eine zu große Batchgröße in der Suche als Ergebnis auf, statt zu fehlen. Bricht ein Lauf ab, ohne eine Datei zu schreiben, etwa weil Slurm ihn nach Ablauf der Zeit beendet, merkt das die Auswertung: Sie vergleicht die Suche mit dem Gitter aus der Konfiguration und führt fehlende Kombinationen als „fehlt“.

In Schritt 2 kann `train` nur, was der heutige `Trainer` kann: `GPTModel` oder `GPTModelV2`, `fp32` oder `fp16-mixed`, ohne `compile` und ohne fused AdamW. Alles andere ergibt einen Fehler, bis Schritt 3 fertig ist. `lotr-small-week4.toml` schaltet zusätzlich die Messung von GPT-2 original ein.

### Konfiguration

`configs/` enthält eine TOML-Datei pro Experiment: `smoke.toml`, `lotr-small-week4.toml` (Nr. 2), `lotr-small.toml` (Nr. 3 bis 5, mit dem Gitter der Suche), `lotr-small-main.toml` (Nr. 6), `lotr-medium.toml` (Nr. 7) und `lotr-large.toml` (Nr. 8). Auf der Kommandozeile lassen sich nur Lernrate, Batchgröße, Epochen, Prüfabstand und Laufname überschreiben. Alle wirksamen Werte gehen an W&B und in die JSON-Datei.

Skizze von `lotr-small.toml`:

```toml
seed = 123
model_class = "GPTModelV2"

[data]
file = "The-lord-of-the-rings.txt"
start_line = "PROLOGUE"
end_line = "APPENDIX A"
book_title = "THE LORD OF THE RINGS"
train_ratio = 0.8
val_ratio = 0.1
seq_len = 256

[model]
vocab_size = 50257
context_length = 256
emb_dim = 768
n_heads = 12
n_layers = 12
drop_rate = 0.1
qkv_bias = true

[train]
batch_size = 8              # über alle GPUs zusammen
learning_rate = 4e-4
epochs = 4
weight_decay = 0.1
grad_clip = 1.0
eval_every = 100
precision = "bf16-mixed"
compile_backend = "inductor"
fused_adamw = true

[generation]
prompt = "The Elves are"
samples = 3
max_new_tokens = 40
temperature = 0.8
top_k = 40

[search]
learning_rates = [1e-4, 3e-4, 1e-3, 3e-3]
batch_sizes = [8, 32, 128]
epochs = 1
```

### Skripte

| Skript | Wofür | Besonderheit |
|---|---|---|
| `slurm/setup.sh` | Einrichtung, einmal pro Person auf dem Login-Knoten | siehe Teil 1 |
| `slurm/check.sbatch` | Nr. 1 | `nvidia-smi`, dann `cluster.check` auf allen Knoten; der Benchmark nur, wenn es ein Knoten ist |
| `slurm/train.sbatch` | Smoke-Lauf, Nr. 2, 3, 5 bis 8 | Knotenzahl beim Abschicken: `sbatch --nodes=2 slurm/train.sbatch --config configs/lotr-small-main.toml` |
| `slurm/search.sbatch` | Nr. 4 | `--array=0-11%4`, `torchrun --standalone` auf einem Knoten |
| `scripts/smoke_notebook.py` | Notebook-Smoke-Test | lokal, mit `uv run --with nbclient` |

Gemeinsam für die Jobskripte: Partition `gpu1` (auf dem bwUniCluster beim Abschicken mit `-p` überschreiben), `--gres=gpu:1`, eigener Hauptspeicher statt der 5 GB Default, eine Zeitgrenze von höchstens einer Stunde und Logs nach `logs/%x-%j.out`. `logs/` kommt in die `.gitignore`. Die Skripte setzen `UV_NO_DEV=1` und `TIKTOKEN_CACHE_DIR` und aktivieren `.venv` direkt. Mit `uv run` würden mehrere Knoten gleichzeitig dieselbe Umgebung auf dem gemeinsamen Dateisystem synchronisieren. Abgeschickt wird aus dem eigenen Klon. Namen, E-Mail-Adressen und feste Pfade stehen in keinem Skript.

Skizze von `train.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=sota2-train
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export UV_NO_DEV=1
export TIKTOKEN_CACHE_DIR="$HOME/.cache/tiktoken"
source .venv/bin/activate

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

srun torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc-per-node="$SLURM_GPUS_ON_NODE" \
  --rdzv-id="$SLURM_JOB_ID" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  -m infm_sota2.cluster.train "$@"
```

W&B läuft online, wenn Job 1 Internet auf den Rechenknoten findet. Sonst exportiert man vor dem Abschicken `WANDB_MODE=offline` und lädt die Läufe danach auf dem Login-Knoten mit `wandb sync` hoch. Ohne W&B-Zugang gilt `WANDB_MODE=disabled`. Für die Läufe mit mehreren Knoten steht `NCCL_DEBUG=INFO` im Skript, damit das Log zeigt, ob NCCL InfiniBand nutzt.

### Messgrößen

- **Tokens/s:** Summe über alle Prozesse, Median der Intervalle zwischen zwei Prüfungen ohne das erste Intervall. Die Prüfung selbst zählt nicht mit.
- **Speedup und Effizienz:** S_p = Tokens/s bei p Knoten geteilt durch Tokens/s bei einem Knoten, E_p = S_p / p (Folie 5). Weil die Gesamt-Batchgröße gleich bleibt, ist die Arbeit pro Schritt gleich.
- **Dieselben Batches bei jeder Knotenzahl:** Der `DistributedSampler` mischt alle Fenster mit `seed + epoch` und gibt Rang r die Fenster r, r+p, r+2p und so weiter. Bei gleicher Gesamt-Batchgröße B enthält Schritt k deshalb bei jeder Knotenzahl p genau die Fenster k·B bis (k+1)·B−1 der gemischten Reihenfolge. Ohne Dropout wären die Updates bis auf die Rundung gleich, mit Dropout 0,1 ist der Loss nur fast gleich.
- **MFU** wie in Woche 5, gegen 362,05 TFLOPS (L40S, bf16).
- **Speicher:** `torch.cuda.max_memory_allocated` auf Rang 0.
- **Loss:** mittlere Cross-Entropy pro Token in fp32, Perplexity gleich e hoch Loss.

### Ergebnisse und Notebook

Rang 0 jedes Laufs schreibt `results/week06/<benutzer>/<gruppe>/<lauf>.json`, der Benutzer ist `$USER`. Darin stehen die wirksame Konfiguration, Knoten, Job-ID, GPU, Treiber, die Versionen von torch, CUDA und NCCL, der Git-Commit, `TrainResult`, Test-Loss und Perplexity, Tokens/s, MFU, Speicher, die Wanduhrzeit von `fit`, der Verlauf der Validierungs-Losses, die Texte und gegebenenfalls der Fehler. Die Dateien kommen ins Repo. Sie enthalten keinen Buchtext, nur Texte, die die Modelle selbst erzeugt haben. Die W&B-Gruppen heißen `week6-smoke`, `week6-week4`, `week6-optimized`, `week6-search`, `week6-scaling`, `week6-main` und `week6-medium`. Die Namen der W&B-Läufe enthalten ebenfalls den Benutzernamen, damit sich Läufe verschiedener Personen nicht verwechseln lassen.

Ein neues Notebook `workbook_timo_cluster.ipynb` liest diese Dateien und läuft ohne GPU. Es zeigt die Umgebung aus Job 1, den Benchmark der L40S neben der A100 aus Woche 5, Woche 4 gegen optimiert, die Suche als Tabelle, Speedup und Effizienz als Diagramm mit der idealen Geraden, Hauptlauf und Medium mit Test-Loss, Verlauf und Texten sowie das Fazit. Die Logik dafür liegt getestet in `cluster/results.py`, so kann jede Person ein eigenes Notebook mit ihren Läufen bauen.

## Tests

Alle laufen auf der CPU in Sekunden.

- **`train()`:** `bf16-mixed` senkt auf der CPU den Loss. `fp16-mixed` auf der CPU ergibt `ValueError`. Ein Checkpoint aus einem Lauf mit kompiliertem Modell (`backend="eager"`) lädt strikt in ein frisches Modell. Mit `DistributedSampler` haben zwei Epochen verschiedene Reihenfolgen. `tokens_per_second` ist bei einem Intervall `None` und sonst eine Zahl über 0.
- **Gleiche Batches:** Für p = 1, 2 und 4 ergeben die lokalen Batches aller Ränge zusammen in jedem Schritt dieselben Fenster. Der Test braucht dafür keine Prozessgruppe.
- **DDP rechnet richtig:** Zwei CPU-Prozesse mit Gloo und je Batch 4 haben nach einem Schritt dieselben Gewichte wie ein Prozess mit Batch 8 (Dropout 0, `allclose`). Nur Rang 0 schreibt Checkpoint und JSON.
- **`Trainer.fit`:** Ohne `torchrun` bleibt es beim alten Loader, die bisherigen Tests bleiben grün. `compile_backend="eager"` mit `bf16-mixed` läuft. Eine Batchgröße, die nicht aufgeht, und fused AdamW auf der CPU ergeben `ValueError`.
- **`config`:** Fehlende und unbekannte Schlüssel ergeben einen Fehler. Jeder Index der Suche ergibt die erwartete Kombination, ein Index außerhalb einen Fehler.
- **`results`:** Speedup und Effizienz an Zahlen, die sich von Hand nachrechnen lassen. Fehlt der Lauf mit einem Knoten, gibt es einen Fehler. Ein gescheiterter oder fehlender Lauf erscheint in der Tabelle.
- **`Environment`:** liest die Variablen von `torchrun`, `WANDB_MODE` und `SOTA2_DATA_DIR`; ein ungültiger W&B-Modus ergibt `ValueError`.
- **Mini-Buch und Smoke-Lauf:** Der Aufruf schreibt das Mini-Buch. `cluster.train` mit `configs/smoke.toml` läuft auf der CPU durch, senkt den Validierungs-Loss und schreibt die JSON-Datei.
- **`check` und `benchmark`:** laufen mit einem winzigen Modell auf der CPU durch.
- **Skripte:** `bash -n` prüft die Syntax der Shell-Skripte, unter Windows wird der Test übersprungen. Kein Skript enthält einen Benutzernamen oder einen Pfad unter `/home`.
- **Notebooks:** `test_notebooks.py` prüft auch das neue Notebook auf `BRANCH = "main"`.

## Review-Fokus

1. **DDP hängt:** Ein Rang ruft eine kollektive Operation auf, die die anderen nicht aufrufen, zum Beispiel eine Auswertung nur auf Rang 0 über das DDP-Modell. Deshalb wertet jeder Rang auf dem Modell ohne DDP aus. Test mit zwei Prozessen.
2. **Checkpoint-Schlüssel** mit `module.` oder `_orig_mod.`. Test: strikt in ein frisches Modell laden.
3. **Batchgröße geht nicht auf:** `ValueError` vor dem ersten Schritt.
4. **Zahlen aus Woche 4 ändern sich:** Notebook-Smoke-Test vorher und nachher.
5. **Gescheiterte Läufe** müssen in der Auswertung sichtbar bleiben.
6. **Kompilieren in den Tokens/s:** Das erste Intervall zählt nicht.
7. **W&B blockiert den Job,** weil die Rechenknoten kein Internet haben oder die Person keinen Zugang hat. `WANDB_MODE` regelt das.
8. **Mehrere Knoten synchronisieren gleichzeitig mit uv:** Deshalb aktivieren die Skripte `.venv` direkt.
9. **Gleicher Port im Job-Array:** Die Suche nutzt `torchrun --standalone`.
10. **Etwas klappt nur bei Timo:** Name, Pfad oder Konto in einem Skript oder einer Konfiguration. Test und Review prüfen das.

## Reihenfolge und Branches

1. **Einstieg und Job 1:** `docs/dachs.md`, `slurm/setup.sh`, das Mini-Buch mit Aufruf, `cluster.check`, `cluster.benchmark`, `slurm/check.sbatch`, `logs/` in der `.gitignore`. Danach loggt sich Timo ein, richtet alles ein und startet Job 1.
2. **Smoke-Lauf und Job 2:** `cluster.config`, `cluster.train` und `cluster.results` mit dem heutigen `Trainer` auf einer GPU, dazu `configs/smoke.toml`, `configs/lotr-small-week4.toml` und `slurm/train.sbatch`. Timo startet den Smoke-Lauf auf einem Knoten, dann Nr. 2.
3. **Umbau nach Weg A:** die Änderungen aus Teil 4, die Zelle im LOTR-Notebook mit dem Notebook-Smoke-Test vorher und nachher, `scripts/smoke_notebook.py`, die DDP-Tests, die übrigen Konfigurationen und `slurm/search.sbatch`. Danach laufen der Smoke-Lauf auf 2 Knoten, Nr. 3 und Nr. 4 bis 7.
4. **Auswertung:** `workbook_timo_cluster.ipynb`, die Ergebnisse in dieser Datei, die README mit `configs/`, `slurm/`, `scripts/`, `results/` und `infm_sota2/cluster/`.

Schritt 1 und 2 sind klein, auf dem Cluster lässt sich also schon testen, während Schritt 3 entsteht. Schritt 1 und 2 liegen auf `feature/cluster-setup`, Schritt 3 und 4 auf `feature/cluster-training`, jeweils mit Pull Request nach `week/06`. Nach jeder Aufgabe laufen ruff, pyright und die Tests, dann committet der Umsetzer mit Timo als Autor und pusht.

## Entscheidungen

Von Timo entschieden:

- Woche 6 ist geschafft, wenn wir auf den Cluster kommen und getestet haben, dass alles funktioniert. Herr der Ringe oder ein kleiner Text, kein Lauf länger als 2 Stunden.
- Alles soll für die ganze Gruppe funktionieren, nicht nur mit Timos Konto, Drive oder Pfaden.
- Umfang „Folie 46 im Kleinen“ plus die Zusatzläufe Nr. 1, Nr. 2 und 3 sowie Nr. 5.
- Weg A: die vorhandenen Bausteine erweitern.
- Die Optimierungen aus Woche 5 kommen ins Training, dazu fused AdamW und die Suche nach der größten Batchgröße. TF32, das aufgefüllte Vokabular und der KV-Cache bleiben draußen.
- Das LOTR-Notebook bleibt bei den Einstellungen von Woche 4.
- Die Reihenfolge der Schritte und die Zahlen der Läufe: Gitter der Suche, 20 Epochen, Skalierung auf 1, 2 und 4 Knoten mit Batch 32, halbe Lernrate für Medium.
- Für den Notebook-Smoke-Test lädt Timo den Buchtext in die Sitzung des Umsetzers hoch.
- SSH aus der PowerShell.
- Der Umsetzer committet und pusht mit Timo als Autor, ohne Co-Autor- und Session-Zeilen.

Mein Vorschlag, bitte prüfen:

- `torchrun` über `srun`, auf einem Knoten `--standalone`.
- Unter `torchrun` immer `DistributedSampler`, auch mit einem Prozess; DDP erst ab zwei Prozessen.
- `batch_size` meint die Gesamt-Batchgröße.
- Auswertung in fp32 auf dem Modell ohne DDP und ohne `compile`, auf allen Rängen.
- Tokens/s als Median der Intervalle ohne das erste.
- TOML-Dateien und `--grid-index` für die Suche; nur wenige Werte lassen sich auf der Kommandozeile überschreiben.
- Konstante Lernrate ohne Warmup wie in Woche 4.
- Höchstens 4 Aufgaben der Suche gleichzeitig.
- `.venv` aktivieren statt `uv run` in den Jobs.
- Einstieg über `docs/dachs.md` und `slurm/setup.sh`; Datenordner über `SOTA2_DATA_DIR`, damit die Gruppe einen gemeinsamen Workspace nutzen kann; W&B per `WANDB_MODE` abschaltbar.
- Smoke-Lauf mit dem selbst geschriebenen Mini-Buch, damit er bei allen ohne Buchtext läuft; der Notebook-Smoke-Test als Skript im Repo.
- Ergebnisse als JSON im Repo, getrennt nach Benutzer, dazu das Notebook für die Vorlesung.
- GPT-2 Large nur, wenn Zeit bleibt.
- Zwei Feature-Branches.

Dem Umsetzer überlassen sind die genauen Signaturen, die Aufteilung innerhalb der Module, die Werte für Hauptspeicher, CPUs und Zeit in den Jobskripten, das Format der JSON-Dateien, die Texte in Anleitung und Notebook und die Einzelheiten der Tests.
