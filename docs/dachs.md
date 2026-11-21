# Auf dem Cluster DACHS

Diese Anleitung bringt jede Person aus der Gruppe auf den Cluster DACHS der Hochschule Esslingen. Alle richten die Umgebung mit demselben Skript ein und starten die Jobs mit denselben Befehlen. In keinem Skript steht ein Name oder ein Pfad einer Person. Welche Läufe es gibt und warum, steht in [Woche 6: Pre-Training auf dem Cluster](cluster-training.md).

Der Weg: Konto anlegen, per SSH einloggen, Repo klonen, `bash slurm/setup.sh`, Job 1, Smoke-Lauf, dann die Läufe mit dem Buch.

## Was DACHS bietet

- **Partition `gpu1`:** 45 Knoten mit je einer NVIDIA L40S (48 GB) und 48 Kernen, verbunden über InfiniBand HDR100. Ein Job darf bis zu 16 Knoten und 72 Stunden nutzen.
- **Ohne eigene Angaben** bekommt ein Job 30 Minuten und 5 GB Hauptspeicher, und die Partition muss immer dabeistehen. Unsere Jobskripte setzen alle drei.
- **Werktags von 8 bis 20 Uhr** reserviert der JupyterHub 4 Knoten. Tagsüber sind also weniger Knoten frei.
- **Login-Knoten** sind für Kurzes da: einrichten, Dateien bearbeiten und hochladen, Jobs abschicken. Rechenjobs auf einem Login-Knoten beendet der Betreiber ohne Vorwarnung.

Die Partitionen `gpu8` (ein Knoten mit 8 H100) und `gpu4` (AMD) nutzen wir nicht. Mehr im [bwHPC-Wiki zu DACHS](https://wiki.bwhpc.de/e/DACHS).

## 1. Konto

Einmal pro Person:

1. Das bwUniCluster-Entitlement der HS Offenburg. Es kommt von der Hochschule.
2. In bwIDM (<https://login.bwidm.de>) den Dienst DACHS registrieren, die Zwei-Faktor-Anmeldung mit einer OTP-App einrichten und ein Service-Passwort setzen. Das Service-Passwort gilt nur für den Cluster und ist ein anderes als das der Hochschule.
3. Der Benutzername ist `of_` und das HSO-Kürzel. Unten steht dafür `of_<HSO-Kürzel>`.

Die Schritte mit Bildern stehen im Wiki unter [Registration](https://wiki.bwhpc.de/e/Registration) und [DACHS/Login](https://wiki.bwhpc.de/e/DACHS/Login).

## 2. Einloggen

### Netz

DACHS nimmt nur Verbindungen aus dem BelWü-Netz an. Außerhalb der Hochschule erst das VPN der HS Offenburg verbinden, und zwar ohne Split-Tunneling. Mit Split-Tunneling kommt keine Verbindung zustande.

### Erster Login aus der PowerShell

Windows bringt `ssh` und `scp` mit. In der PowerShell:

```powershell
ssh of_<HSO-Kürzel>@dachs-login.hs-esslingen.de
```

Beim ersten Mal zeigt ssh den Fingerabdruck des Servers und fragt, ob er stimmt. Er muss so aussehen:

```text
ED25519 key fingerprint is SHA256:X9eRJepYD3da3BM1pgiWxnvRc/Pt5eBLUr18tDUsZjU
```

Nur bei genau diesem Wert `yes` eingeben, sonst abbrechen. Die Fingerabdrücke für ECDSA und RSA stehen im [Wiki](https://wiki.bwhpc.de/e/DACHS/Login). Danach fragt der Server erst `Your OTP:` nach dem Code aus der App und dann `Password:` nach dem Service-Passwort.

### Kurzer Name in `~/.ssh/config`

Mit einem Eintrag in `C:\Users\<Name>\.ssh\config` reicht `ssh dachs`, und `scp` versteht `dachs:` als Ziel. Am einfachsten öffnet man die Datei in VS Code, in der PowerShell mit `code $HOME\.ssh\config`. Notepad hängt an eine neue Datei ohne Endung gern `.txt` an, dann findet ssh sie nicht. Eintragen:

```text
Host dachs
    HostName dachs-login.hs-esslingen.de
    User of_<HSO-Kürzel>
```

### VS Code

Die Erweiterung Remote-SSH nutzt denselben Eintrag: Befehl „Remote-SSH: Connect to Host…“, dann `dachs`. VS Code fragt OTP und Service-Passwort oben im Fenster ab. Erscheinen die Abfragen nicht, hilft in den Einstellungen `"remote.SSH.showLoginTerminal": true`. Danach mit „Open Folder“ den eigenen Klon öffnen.

Das Terminal in VS Code läuft auf dem Login-Knoten. Dort gilt dasselbe wie oben: einrichten und Jobs abschicken, trainiert wird nur mit `sbatch`.

### SSH-Schlüssel

Für DACHS nennt das Wiki keine SSH-Schlüssel aus bwIDM (Stand November 2026, Seite [Registration/SSH](https://wiki.bwhpc.de/e/Registration/SSH)). Jede neue Verbindung fragt deshalb nach OTP und Service-Passwort, auch jedes `scp`. Dateien lädt man am bequemsten über die offene VS-Code-Verbindung hoch: aus dem Windows-Explorer in die Dateiliste von VS Code ziehen.

## 3. Einrichten

Einmal pro Person, auf dem Login-Knoten, zum Beispiel im Home-Verzeichnis:

```bash
git clone https://github.com/Timbo3399/INFM_SOTA2.git
cd INFM_SOTA2
git switch week/06
bash slurm/setup.sh
```

`setup.sh` erledigt den Rest:

- Fehlt uv oder ist es älter als 0.13, installiert es uv nach `~/.local/bin`.
- Es installiert die Pakete aus `uv.lock` ohne die Notebook-Pakete. Beim ersten Mal sind das mehrere GB für torch.
- Es füllt den tiktoken-Cache, weil die Rechenknoten womöglich kein Internet haben, und legt den Ordner `logs/` an.
- Es schreibt das Mini-Buch für den Smoke-Lauf nach `data/`.
- Es prüft Buchtext, GPT-2-Gewichte und W&B. Für alles, was fehlt, nennt es den Befehl, der es behebt.

Die letzte Zeile sagt, wofür die Einrichtung reicht: „bereit für den Smoke-Lauf“ oder „bereit für alle Läufe“. Außerhalb des Klons ändert das Skript nur den eigenen Cache und bei Bedarf uv in `~/.local/bin`. Es darf jederzeit wieder laufen, etwa nach einem `git pull`.

### Buchtext

Der Text von *The Lord of the Rings* ist urheberrechtlich geschützt. Wer mit dem Buch trainiert, legt die eigene Kopie als `data/The-lord-of-the-rings.txt` in den eigenen Klon. Es ist dieselbe Datei wie in Woche 4 ([Anleitung](lotr-training.md)). `data/` steht in der `.gitignore`, die Datei kommt nie ins Repo.

Hochladen geht per Ziehen in VS Code oder aus der PowerShell, im Ordner mit der Datei:

```powershell
scp The-lord-of-the-rings.txt dachs:INFM_SOTA2/data/
```

Das Ziel gilt, wenn der Klon direkt im Home-Verzeichnis liegt. `setup.sh` gibt den Befehl mit dem genauen Pfad aus.

### GPT-2-Gewichte

Nur Lauf Nr. 2 braucht die Gewichte von OpenAI, rund 700 MB. `setup.sh` gibt den Befehl aus, der sie auf dem Login-Knoten nach `data/` lädt und die Prüfsumme kontrolliert.

### W&B

W&B ist kein Muss. Wer den Zugang zum Team hat, meldet sich einmal auf dem Login-Knoten an, mit dem API-Key von <https://wandb.ai/authorize>:

```bash
uv run wandb login --verify
```

Ohne W&B vor dem Abschicken `export WANDB_MODE=disabled` setzen. `sbatch` gibt die Variablen des Terminals an den Job weiter. Kommen die Rechenknoten nicht ins Internet (das zeigt Job 1), heißt es `export WANDB_MODE=offline`. Die Läufe lädt man dann nach dem Job auf dem Login-Knoten hoch:

```bash
uv run wandb sync wandb/offline-run-*
```

### Daten in einem anderen Ordner

Zeigt die Umgebungsvariable `SOTA2_DATA_DIR` auf einen Ordner, nehmen `setup.sh` und die Jobs ihn statt `data/`. Wer das will, trägt die Variable in `~/.bashrc` ein und führt `setup.sh` noch einmal aus. Der Ordner muss beschreibbar sein, weil `setup.sh` dort das Mini-Buch ablegt. Checkpoints, Logs und Ergebnisse landen immer im eigenen Klon.

### Gemeinsamer Workspace

Die Gruppe kann Buchtext und Gewichte auch einmal in einem Workspace ablegen statt in jedem Klon. Eine Person legt ihn an, kopiert die Dateien hinein und gibt den anderen Lesezugriff:

```bash
ws_allocate sota2-data 30                     # 30 Tage, verlängern mit ws_extend sota2-data 30
ws_share share sota2-data of_<HSO-Kürzel>     # Lesezugriff für eine andere Person
ws_find sota2-data                            # Pfad des Workspace
```

Die anderen verlinken die Dateien in ihr eigenes `data/`, mit dem Pfad aus `ws_find`:

```bash
ln -s <Pfad>/The-lord-of-the-rings.txt data/
ln -s <Pfad>/gpt2-small-124M.pth data/
```

`SOTA2_DATA_DIR` passt hier nicht, weil die anderen im Workspace nur lesen dürfen. Ein Workspace läuft nach der angegebenen Zahl von Tagen ab, wer ihn weiter braucht, verlängert ihn rechtzeitig. Wie die Workspaces funktionieren, steht im [Wiki](https://wiki.bwhpc.de/e/Workspace).

## 4. Slurm kennenlernen

Ein paar Befehle zum Ausprobieren:

```bash
sinfo_t_idle                                # freie Knoten pro Partition
squeue -u $USER                             # eigene Jobs, ST: PD wartet, R läuft
srun -p gpu1 --gres=gpu:1 -t 5 nvidia-smi   # Mini-Job: 1 GPU, höchstens 5 Minuten
```

`srun` wartet, bis ein Knoten frei ist, und zeigt die Ausgabe im Terminal. Für alles, was länger dauert, gibt es `sbatch`. Es stellt ein Jobskript in die Warteschlange und kehrt sofort zurück, die Ausgabe landet in einer Logdatei.

```bash
sbatch slurm/check.sbatch                   # Antwort: Submitted batch job <Job-ID>
squeue -u $USER --start                     # geschätzte Startzeit
scancel <Job-ID>                            # Job abbrechen
sacct -j <Job-ID> --format=JobID,JobName,State,Elapsed,MaxRSS   # Zustand, Laufzeit und Speicher
```

Die Zeilen mit `#SBATCH` oben im Jobskript sind Voreinstellungen. Optionen beim Abschicken haben Vorrang, so wird aus dem Job für einen Knoten einer für zwei:

```bash
sbatch --nodes=2 slurm/train.sbatch --config configs/smoke.toml
```

Alles hinter dem Namen des Skripts bekommt das Skript selbst, hier `--config configs/smoke.toml`.

Die Suche nach Batchgröße und Lernrate ist ein Job-Array. `--array=0-11%4` schickt zwölf Aufgaben auf einmal ab, von denen höchstens vier gleichzeitig laufen. Jede Aufgabe ist ein eigener kleiner Job und liest ihre Nummer aus `SLURM_ARRAY_TASK_ID`. `squeue` zeigt sie als `<Job-ID>_<Nummer>`, und `scancel <Job-ID>` bricht alle ab.

`squeue` bitte nicht in einer Schleife oder mit `watch` aufrufen. DACHS erlaubt einen Aufruf höchstens alle 30 Sekunden. Zum Weiterlesen: [Slurm-Kurzanleitung](https://slurm.schedmd.com/quickstart.html), [Befehlsübersicht](https://slurm.schedmd.com/pdfs/summary.pdf) und [Partitionen von DACHS](https://wiki.bwhpc.de/e/DACHS/Queues).

## 5. Unsere Jobs

Abgeschickt wird immer aus dem eigenen Klon, weil die Jobs dort Code, `.venv` und Konfiguration suchen. Ein Job aus einem anderen Ordner bricht mit einer Meldung ab. Jede Einstellung eines Laufs steht in einer Datei in `configs/`.

| Schritt | Befehl | Lauf |
|---|---|---|
| Job 1: Umgebung prüfen, Benchmark | `sbatch slurm/check.sbatch` | Nr. 1 |
| Netz zwischen zwei Knoten | `sbatch --nodes=2 slurm/check.sbatch` | |
| Smoke-Lauf | `sbatch slurm/train.sbatch --config configs/smoke.toml` | |
| Smoke-Lauf auf zwei Knoten | `sbatch --nodes=2 slurm/train.sbatch --config configs/smoke.toml` | |
| Einstellungen aus Woche 4 | `sbatch slurm/train.sbatch --config configs/lotr-small-week4.toml` | Nr. 2 |
| Optimiert | `sbatch slurm/train.sbatch --config configs/lotr-small.toml` | Nr. 3 |
| Suche nach Batchgröße und Lernrate | `sbatch slurm/search.sbatch` | Nr. 4 |
| Skalierung | `sbatch --nodes=1 slurm/train.sbatch --config configs/lotr-small-scaling.toml`, dann mit `--nodes=2` und `--nodes=4` | Nr. 5 |
| Hauptlauf | `sbatch --nodes=2 slurm/train.sbatch --config configs/lotr-small-main.toml` | Nr. 6 |
| GPT-2 Medium | `sbatch --nodes=2 slurm/train.sbatch --config configs/lotr-medium.toml` | Nr. 7 |
| GPT-2 Large, optional | `sbatch --nodes=2 slurm/train.sbatch --config configs/lotr-large.toml` | Nr. 8 |

Der Smoke-Lauf braucht nur das Mini-Buch und dauert wenige Minuten. Er ist bestanden, wenn der Job ohne Fehler endet, der Validierungs-Loss sinkt und eine JSON-Datei unter `results/` liegt. Nach jeder Änderung an einem Jobskript läuft er zuerst.

Nr. 5 bis 8 bauen auf der Suche auf. Wer die Suche auswertet, trägt die beste Lernrate für Batch 32 in `lotr-small-scaling.toml` ein und die beste Kombination in `lotr-small-main.toml`, Medium und Large leiten ihre Lernrate davon ab. Danach committen und pushen, damit alle mit denselben Werten trainieren.

## 6. Logs und Ergebnisse

- **Log:** Jeder Job schreibt nach `logs/<Jobname>-<Job-ID>.out`, die Suche nach `logs/<Jobname>-<Job-ID>_<Nummer>.out`. Mitlesen mit `tail -f logs/<Datei>`, `Strg+C` beendet nur `tail`, nicht den Job.
- **Ergebnis:** Jeder Lauf schreibt eine JSON-Datei nach `results/week06/<Benutzer>/<Gruppe>/<Lauf>-job<Job-ID>.json`, mit allen Einstellungen. Wegen der Job-ID überschreibt eine Wiederholung nichts.
- **Checkpoints** liegen in `checkpoints/` im Klon und kommen nicht ins Repo.

Die Ergebnisse gehören ins Repo. Am einfachsten holt man sie auf den eigenen Rechner, per Rechtsklick und „Download…“ in VS Code oder aus der PowerShell:

```powershell
scp -r dachs:INFM_SOTA2/results/week06 .
```

Dann wie immer auf einem Branch committen und einen Pull Request stellen. Auf dem Cluster zu committen geht auch, zum Pushen braucht man dort aber einen eigenen Zugang zu GitHub, etwa einen SSH-Schlüssel oder ein Token.

## 7. Typische Fehler

| Was man sieht | Ursache | Abhilfe |
|---|---|---|
| `Connection timed out` beim Einloggen | kein VPN oder VPN mit Split-Tunneling | VPN der HS Offenburg ohne Split-Tunneling |
| `REMOTE HOST IDENTIFICATION HAS CHANGED` | Der Server meldet einen anderen Schlüssel. | nicht verbinden, Fingerabdruck mit dem Wiki vergleichen |
| `Permission denied` nach `Your OTP:` | Code abgelaufen oder falsches Passwort | nächsten Code abwarten, das Service-Passwort nehmen, nicht das der Hochschule |
| Job steht lange auf `PD` mit `(Priority)` oder `(Resources)` | alle Knoten belegt | warten, `squeue -u $USER --start` zeigt die Schätzung. Weniger Knoten starten früher. |
| `DUE TO TIME LIMIT` am Ende des Logs | Zeitgrenze des Jobs erreicht | mehr Zeit beim Abschicken, etwa `sbatch --time=01:00:00 ...` |
| `oom_kill` im Log, `OUT_OF_MEMORY` in `sacct` | Hauptspeicher des Knotens zu knapp | mehr `--mem` beim Abschicken |
| `torch.OutOfMemoryError: CUDA out of memory` | Batch passt nicht in die 48 GB der GPU | kleinere Batchgröße. In der Suche zählt das als gescheiterter Lauf. |
| `Keine .venv in ...` | Einrichtung fehlt oder Job aus dem falschen Ordner | im Klon `bash slurm/setup.sh`, aus dem Klon abschicken |
| Meldung, dass der Ordner kein Klon ist | `sbatch` außerhalb des Klons aufgerufen | erst `cd` in den Klon |
| Job endet sofort, in `logs/` liegt kein Log | Slurm legt das Log in `logs/` unter dem Ordner an, aus dem der Job kam. Fehlt der Ordner dort, gibt es kein Log. | aus dem Klon abschicken, `setup.sh` legt `logs/` an. `sacct -j <Job-ID>` zeigt den Zustand. |
| `FileNotFoundError` mit einem Pfad in `data/` | Buchtext fehlt oder `SOTA2_DATA_DIR` zeigt woandershin | Buchtext hochladen, `setup.sh` prüft das |
| W&B bricht ab, weil kein API-Key da ist | kein Login auf dem Cluster | `uv run wandb login --verify` oder `export WANDB_MODE=disabled` |
| `The NVIDIA driver on your system is too old` | Treiber kann kein CUDA 13 | Job 1 zeigt den Treiber, Timo entscheidet über einen anderen CUDA-Index (README, „Auf dem Cluster“) |
| `Disk quota exceeded` | Home-Verzeichnis voll | `uv cache clean`, alte Checkpoints löschen |
| `uv sync` meldet, dass `uv.lock` aktualisiert werden müsste | `pyproject.toml` im Klon geändert | `git status`, eigene Änderung zurücknehmen, `git pull` |
| `$'\r': command not found` | Skript mit Windows-Zeilenenden, etwa per `scp` kopiert | Skripte nur über git holen, die `.gitattributes` erzwingt LF |

## 8. Später: bwUniCluster 3.0

Der bwUniCluster ist vom 23.11. bis 04.12.2026 abgeschaltet. Woche 6 läuft deshalb auf DACHS, die Skripte funktionieren dort aber genauso.

- **Zugang:** in bwIDM zusätzlich den Dienst „bwUniCluster 3.0“ registrieren. Login über `uc3.scc.kit.edu`, sonst wie bei DACHS ([Wiki](https://wiki.bwhpc.de/e/BwUniCluster3.0/Login)).
- **SSH-Schlüssel** gehen dort: Ein interaktiver Schlüssel in bwIDM gilt nach einem Login mit OTP und Service-Passwort 8 Stunden lang ([Registration/SSH](https://wiki.bwhpc.de/e/Registration/SSH)).
- **GPU-Partitionen** heißen anders, und ein Knoten hat 4 GPUs: `gpu_h100` (H100), `gpu_a100_il` und `gpu_h100_il` (A100 bzw. H100), dazu `gpu_a100_short` und `gpu_h100_short` für höchstens 30 Minuten und `dev_gpu_h100` zum Testen ([Wiki](https://wiki.bwhpc.de/e/BwUniCluster3.0/Running_Jobs)).
- **Unsere Skripte** stehen auf `gpu1`. Beim Abschicken überschreibt `-p` die Partition, und `--gres=gpu:4` holt alle 4 GPUs. `torchrun` startet dann einen Prozess pro GPU:

  ```bash
  sbatch -p gpu_h100 --gres=gpu:4 slurm/train.sbatch --config configs/smoke.toml
  ```

  Die Gesamt-Batchgröße muss durch die Zahl der GPUs teilbar sein. Ausprobiert wird das erst, wenn der Cluster wieder läuft.
