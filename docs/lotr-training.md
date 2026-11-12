# GPT-2 Small auf „Der Herr der Ringe“ trainieren

Woche 4, Aufgabe *Pre-Training*. Branch `feature/lotr-training`, Notebook [`notebooks/workbook_timo_lotr.ipynb`](../notebooks/workbook_timo_lotr.ipynb).

Wir trainieren GPT-2 Small mit dem Text von *The Lord of the Rings* und vergleichen dann, wie verschiedene Modelle den Satzanfang **„The Elves are“** fortsetzen.
Die Vorgabe verlangt eine Trainings-Pipeline mit W&B in Colab, mehrere Läufe mit je einer Epoche zur Suche nach Hyperparametern,
Data Loader und Optimizer sowie ein Training auf einem kleinen Datensatz mit anschließendem Test.

## Was verglichen wird

| Modell | Startpunkt | Training | Gemessen |
|---|---|---|---|
| Untrainiert | Zufallsgewichte | keins | Test-Loss, 3 Texte |
| LOTR von Grund auf | Zufallsgewichte | 3 Lernraten × 1 Epoche, dann 4 Epochen mit der besten | Test-Loss, 3 Texte |
| GPT-2 original | OpenAI-Gewichte | keins | Test-Loss, 3 Texte |
| GPT-2 nachtrainiert | OpenAI-Gewichte | 2 Epochen mit Lernrate 5e-5 | Test-Loss, 3 Texte |
| Claude | – | – | 3 Texte aus dem Inkognito-Chat |

Belastbar vergleichen lässt sich nur der **Test-Loss** und die daraus berechnete Perplexity, und zwar zwischen den vier GPT-2-Varianten.
Die Texte zeigen, wie die Modelle schreiben. Für eine Rangfolge reichen drei Stück pro Modell nicht.

## Entscheidungen und Gründe

### Daten

| Entscheidung | Warum | Verworfen |
|---|---|---|
| **Nur der Erzähltext**, vom Prolog bis zum Ende des letzten Kapitels „The Grey Havens“ | Die Datei ist ein umgewandeltes Buch. Ab etwa 86 % folgen Anhänge mit Stammbäumen, Kalendern und Tabellen, danach der Index. Bei einer Aufteilung am Ende bestünden Validierung und Test fast nur daraus. Vorwort und Inhaltsverzeichnis sind kein Erzähltext. | Rohtext; nur die Anhänge abschneiden |
| **Seitenkopfzeilen entfernen**, also Zeilen wie „22 THE LORD OF THE RINGS“ und „A LONG-EXPECTED PARTY 23“ sowie reine Seitenzahlen | Entfernt wurden 926 Zeilen: 470 mit dem Buchtitel, 446 mit einem von 61 Kapitelnamen, 10 reine Seitenzahlen. Das Modell würde sonst lernen, mitten im Satz „THE LORD OF THE RINGS 123“ zu schreiben. | Stehen lassen |
| **Text über Seitenwechsel zusammenfügen**, wenn der Satz weiterläuft | Die meisten Seitenwechsel liegen mitten im Satz. Nur die Kopfzeile zu löschen hätte dort einen falschen Absatz erzeugt. Fortgesetzt wird, wenn die vorige Zeile nicht mit einem Satzzeichen endet oder die nächste kleingeschrieben beginnt. Ausnahme: Eine Überschrift (nur Großbuchstaben) beendet immer den Absatz. Sonst wird ein neuer Absatz angenommen. Ergebnis: 598 Seitenwechsel zusammengefügt, 324 als Absatz. | Jeder Seitenwechsel als Absatz |
| **Harte Zeilenumbrüche zu Absätzen zusammenfügen**, Absätze durch eine Leerzeile getrennt | Die Datei bricht etwa alle 42 Zeichen um. Diese Umbrüche gehören nicht zum Text und kosten Tokens im Kontextfenster. Leerzeilen trennen in der Datei die Absätze. Ergebnis: 8.749 Absätze. | Umbrüche behalten |
| **Silbentrennung nur auflösen, wenn es das Wort im Text auch ungetrennt gibt** | Zeilen, die mit Bindestrich enden und kleingeschrieben weitergehen, sind teils Trennungen („some-/thing“) und teils echte Bindestriche („long-/expected“). Ein Wörterbuch haben wir nicht, also dient der Text selbst als Wörterbuch. Ergebnis: 567 Trennungen aufgelöst, 159 Bindestriche behalten. | Immer zusammenziehen; nie zusammenziehen |
| Jeder Bereinigungsschritt **zählt mit**, was er ändert, und das Notebook zeigt die entfernten Kopfzeilen | So sieht man, ob die Regeln zu viel oder zu wenig erwischen. | Stille Bereinigung |
| **Aufteilung 80/10/10** in Training, Validierung und Test, als zusammenhängende Blöcke an Absatzgrenzen | Mit der Validierung werden Lernrate und bester Checkpoint ausgewählt. Ein Vergleich auf diesen Daten würde unser Modell gegenüber GPT-2 original bevorzugen, deshalb gibt es ein eigenes Testset. Zusammenhängende Blöcke verhindern, dass sich Trainings- und Testfenster überlappen. | 90/10 wie im Buch; zufällig gemischte Fenster |
| Text und Checkpoints **nicht ins Repo**, und das Notebook gibt keine Ausschnitte des Buchs aus | Der Text ist urheberrechtlich geschützt. `data/`, `checkpoints/` und `*.pth` stehen in der `.gitignore`. Notebook-Ausgaben landen beim Speichern aus Colab auf GitHub. | – |

**Daten nach der Bereinigung:** 2,52 Mio. Zeichen, 671.408 GPT-2-Tokens. Davon gehen 536.346 ins Training, 66.468 in die Validierung und 68.592 in den Test.
Bei Fenstern von 256 Tokens und Batchgröße 8 sind das 261 Schritte pro Epoche.

### Modell und Training

| Entscheidung | Warum | Verworfen |
|---|---|---|
| **GPT-2-Small-Architektur** (12 Schichten, 768 Dimensionen, 12 Köpfe) für alle GPT-2-Varianten | Das verlangt die Vorgabe, und nur so lassen sich die OpenAI-Gewichte laden. Unser Modell hat 163 Mio. Parameter statt 124 Mio., weil Ein- und Ausgabe-Embedding nicht geteilt werden (wie im Buch). | Kleineres Modell |
| **Kontext 256** beim Training von Grund auf, 1024 bei GPT-2 original | 256 wie in Kapitel 5 des Buchs, das spart Speicher und Zeit. Die OpenAI-Gewichte bringen 1024 Positionen mit. Gemessen werden alle auf denselben 256-Token-Fenstern. | Überall 1024 |
| **Ein Tokenizer für alle**, GPT-2-BPE über tiktoken | Nur mit gleichem Tokenizer sind die Losses vergleichbar. | – |
| **Kleine Suche: 3 Lernraten (1e-4, 4e-4, 1e-3) × 1 Epoche** | Die Vorgabe will mehrere Läufe mit je einer Epoche. Ein W&B-Sweep bräuchte die zwei- bis dreifache GPU-Zeit. Gewählt wird die Lernrate mit dem kleinsten Validierungs-Loss. | W&B-Sweep |
| **Festgelegt:** Batchgröße 8, AdamW mit Weight Decay 0.1, Gradient Clipping 1.0, Dropout 0.1, Seed 123 | Werte aus Kapitel 5 und Anhang D des Buchs. Sie zu variieren würde die Suche vervielfachen. | Mitsuchen |
| **Hauptlauf 4 Epochen, gespeichert wird der beste Checkpoint nach Validierungs-Loss** | Bei so wenig Text lernt das Modell den Text bald auswendig, und der Validierungs-Loss steigt wieder. Der beste Checkpoint ist der Stand davor. | Letzten Stand nehmen |
| **Fine-Tuning mit Lernrate 5e-5, 2 Epochen** | Eine kleine Lernrate verändert die vortrainierten Gewichte nur behutsam, 5e-5 ist der übliche Startwert beim Nachtrainieren von GPT-2. Zwei Epochen halten die GPU-Zeit klein. | Gleiche Lernrate wie von Grund auf |
| **Mixed Precision auf der GPU** (float16 mit GradScaler) | Etwa doppelt so schnell auf der T4. Der Loss wird trotzdem in float32 berechnet. Auf der CPU ist sie aus. | Nur float32 |
| **Abbruch bei NaN- oder Inf-Loss** | Ein Lauf mit kaputtem Loss soll laut scheitern und keine Zahl liefern, die echt aussieht. | Weiterlaufen |

### Messung und Vergleich

| Entscheidung | Warum | Verworfen |
|---|---|---|
| **Plausibilitätsprüfung:** Der Loss des untrainierten Modells muss nahe ln(50257) ≈ 10,8 liegen | Raten über 50.257 Tokens ergibt genau diesen Loss. Weicht er stark ab, stimmt etwas mit Daten, Zielen oder Loss nicht. | – |
| **Texte erzeugen mit Temperature 0.8 und Top-k 40, 40 neue Tokens, 3 Stück mit festen Seeds** | Ohne Zufall (immer das wahrscheinlichste Token) wiederholen sich kleine Modelle oft. Feste Seeds machen die Texte reproduzierbar, und alle Modelle laufen mit denselben Einstellungen. | Greedy |
| **Auswendiglern-Check:** längste Wortfolge eines erzeugten Texts, die wörtlich im Trainingstext steht | Bei 163 Mio. Parametern und weniger als 1 Mio. Tokens kann das Modell ganze Sätze kopieren. Der Check zeigt, ob es selbst formuliert. | – |
| **Claude per Hand im Inkognito-Chat**, gleicher Prompt, 3 neue Chats | Kein API-Key und keine Kosten. Im Inkognito-Chat greift Claude weder auf die Memory-Funktion noch auf frühere Chats zu. Das Vorwissen über Tolkien bleibt aber, und den übrigen Kontext von claude.ai wie das aktuelle Datum bekommt Claude auch im Inkognito-Chat (siehe Ergebnisse). Claude bekommt eine Anweisung statt eines bloßen Satzanfangs, und einen Loss gibt es nicht. Deshalb ist Claude nur ein qualitativer Vergleich. | Per API |

### Code

| Entscheidung | Warum | Verworfen |
|---|---|---|
| **Trainings-, Mess- und Generierungscode in `src/infm_sota2/`**, das Notebook ruft ihn nur auf | So verlangt es die README. Alle Modelle nutzen denselben Code, damit sich die Varianten nur in Startpunkt und Training unterscheiden. Der Code hat Unittests und läuft später auch auf dem Cluster. | Alles im Notebook |
| **W&B nicht im Trainingscode**, `train()` bekommt stattdessen eine Funktion `log` übergeben | Die Tests laufen dann ohne W&B-Konto, und der Code hängt nicht an einem bestimmten Werkzeug. | `wandb.log` in `train()` |
| **Keine Default-Werte für Hyperparameter** in den Funktionen, alle Werte stehen in einem `CONFIG` im Notebook und gehen nach W&B | Versteckte Defaults sind versteckte Annahmen. So ist jeder verwendete Wert dokumentiert. | Defaults in Signaturen |
| **OpenAI-Gewichte über einen festen Commit laden** (`rasbt/gpt2-from-scratch-pytorch` @ `402728f`), mit SHA-256-Prüfung und `torch.load(weights_only=True)` | Reproduzierbar: immer dieselbe Datei. Sicher: Eine veränderte Datei wird abgelehnt, und beim Laden läuft kein fremder Code. Vorab geprüft, dass alle 209 Schlüssel zu unserem `GPTModel` passen. | Laden über `main` ohne Prüfung |
| **`datasets` in die dev-Gruppe** verschoben | Der Eintrag `datasets>=5.1.0` hat den Colab-Versions-Check aus der README abstürzen lassen. Außerdem installierte der Cluster es mit, obwohl nur Notebooks es brauchen. | – |

## Aufbau

| Pfad | Inhalt |
|---|---|
| `src/infm_sota2/data/cleaning.py` | `clean_book_text()`: schneidet den Erzähltext aus, entfernt Seitenkopfzeilen, fügt Absätze zusammen und löst die Silbentrennung auf. `split_text()` teilt an Absatzgrenzen in Training, Validierung und Test. |
| `src/infm_sota2/model/pretrained.py` | `GPT2_SMALL_CONFIG`; `download_weights()` lädt die Datei und prüft die SHA-256. |
| `src/infm_sota2/training/loop.py` | `evaluate()`, `train()`, `save_checkpoint()`, `load_checkpoint()` |
| `src/infm_sota2/training/generate.py` | `generate()` mit Temperature und Top-k, `complete()` liefert die Fortsetzung eines Prompts als Text |
| `src/infm_sota2/training/analysis.py` | `longest_copied_span()` für den Auswendiglern-Check |

## Anleitung

### In Colab (für den richtigen Lauf)

1. **Text nach Google Drive legen:** `MyDrive/INFM_SOTA2/data/The-lord-of-the-rings.txt`. Die Datei ist nicht im Repo und kommt auch nicht hinein.
2. **Notebook öffnen:** In Colab *Datei → Notebook öffnen → GitHub*, Repo `Timbo3399/INFM_SOTA2`, Branch `feature/lotr-training`, Datei `notebooks/workbook_timo_lotr.ipynb`.
3. **GPU einschalten:** *Laufzeit → Laufzeittyp ändern → T4-GPU*.
4. **W&B-Key als Secret** `WANDB_API_KEY` anlegen (Schlüssel-Symbol links) und dem Notebook Zugriff geben, siehe README.
5. **Alle Zellen ausführen.** Das Notebook bindet Drive ein, lädt beim ersten Mal die OpenAI-Gewichte (700 MB) nach `MyDrive/INFM_SOTA2/checkpoints/` und trainiert.
   Insgesamt geschätzt 15–30 Minuten auf einer T4. Das ist gerechnet, nicht gemessen.
   Während eines Laufs liegen die Checkpoints auf der schnellen lokalen Platte von Colab. Am Ende jedes Laufs wird der beste nach Drive kopiert.
   Trennt Colab mitten in einem Lauf die Verbindung, ist nur dieser Lauf verloren.
6. **Claude befragen:** In claude.ai einen *Inkognito-Chat* öffnen, den Prompt aus der Zelle „Claude“ unverändert einfügen und die Antwort kopieren.
   Das dreimal, jeweils in einem neuen Inkognito-Chat. Die drei Antworten in die Liste `CLAUDE_COMPLETIONS` eintragen und die Vergleichszellen noch einmal ausführen.
7. **Fazit** in die letzte Zelle schreiben.
8. **Speichern** mit *Datei → Kopie in GitHub speichern*, Branch `feature/lotr-training`. Dabei den **Dateipfad prüfen**: Er muss `notebooks/workbook_timo_lotr.ipynb` lauten.
   Wegen des Schrägstrichs im Branch-Namen hat Colab beim ersten Lauf `lotr-training/notebooks/…` gespeichert.
   Colab fügt außerdem oben eine eigene „Open in Colab“-Zelle mit Link auf den Branch ein. Die muss vor dem Pull Request wieder raus, der Link in der ersten Zelle zeigt auf `main`.

Die Läufe stehen in W&B unter [`hochschule-offenburg/SOTA_2`](https://wandb.ai/hochschule-offenburg/SOTA_2), Gruppen `lotr-lr-search`, `lotr-main` und `lotr-finetune`.

### Lokal (zum Ausprobieren)

Text nach `notebooks/data/The-lord-of-the-rings.txt` legen. Dann `uv sync` ausführen und das Notebook in VS Code mit dem Kernel `.venv` öffnen.
Mit `SMOKE_TEST = True` in der ersten Code-Zelle läuft alles mit einem winzigen Modell und wenigen Schritten auf der CPU.
W&B ist dabei aus, die OpenAI-Gewichte werden nicht geladen, und das Notebook sagt das bei den betroffenen Zellen auch.
Damit lässt sich die Pipeline in etwa 30 Sekunden prüfen. Die Zahlen und Texte aus diesem Modus haben keine Aussagekraft.

## Ergebnisse (Lauf vom 12.11.2026, Tesla T4)

| Modell | Test-Loss | Perplexity |
|---|---|---|
| Untrainiert | 10,98 | 58.620 |
| LOTR von Grund auf | 4,35 | 77,7 |
| GPT-2 original | 3,63 | 37,6 |
| GPT-2 nachtrainiert | **3,09** | **22,0** |

- **Die Plausibilitätsprüfung passt:** Das untrainierte Modell liegt mit 10,98 nahe ln(50257) ≈ 10,8.
- **Lernrate 4e-4** war nach einer Epoche am besten (Validierungs-Loss 4,92; 1e-4: 5,19; 1e-3: 5,09).
- **Die Annahme zum Auswendiglernen hat sich nicht bestätigt.** Der beste Validierungs-Loss lag in jedem Lauf beim letzten Schritt, im Hauptlauf auch nach 4 Epochen (4,46).
  Das Modell hat also noch nicht ausgelernt. Ein nächster Lauf sollte mehr Epochen bekommen.
- **Vorwissen bringt am meisten:** GPT-2 original hat ohne gezieltes Training die halbe Perplexity unseres Modells. Nachtrainiert halbiert sie sich noch einmal.
- **Kein Modell kopiert längere Passagen:** Die längsten wörtlich übernommenen Wortfolgen haben 3–5 Wörter. Claude kommt mit Allerweltswendungen auf 3–4.
- **Claude (Haiku 5.5)** schreibt am flüssigsten, denkt bei „The Elves are“ aber nicht an Tolkien, sondern an Waldelfen, eine Werkstatt im Winter und Hausgeister aus der deutschen Folklore.
  „This November“ zeigt, dass claude.ai auch im Inkognito-Chat das Datum mitgibt.

Die Texte aller Modelle und das ausführliche Fazit stehen im Notebook.

## Grenzen

- **Ein Lauf pro Variante.** Ohne Wiederholungen mit anderen Seeds wissen wir nicht, wie stark der Test-Loss streut. Kleine Unterschiede sind deshalb nicht belastbar.
- **Drei Texte pro Modell sind Beispiele**, keine Statistik.
- **GPT-2 original und Claude** haben beim Vortraining möglicherweise Teile des Buchs gesehen. Claude kennt Tolkien sicher.
- **Claude ist nicht kontextfrei.** Der Inkognito-Chat schaltet nur Memory und Chatverlauf ab. Den übrigen Kontext von claude.ai bekommt Claude weiter, nachweislich das Datum.
  Ob auch Sprache oder Standort dazugehören, lässt sich von außen nicht prüfen. Verglichen wurde außerdem nur das kleine Modell Haiku 5.5.
- **Das Testset ist das Ende von Buch 6.** Inhaltlich unterscheidet es sich etwas vom Training, aber es ist derselbe Erzählstil.
- **Läufe auf der GPU sind nicht bitgenau reproduzierbar.** Gleicher Seed heißt gleiche Daten-Reihenfolge, aber minimale Rechenunterschiede bleiben.
