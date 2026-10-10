# INFM_SOTA2

Gruppenprojekt im Modul SOTA 2 (Master Informatik, Hochschule Offenburg).

Wir bauen ein Large Language Model von Grund auf – Schritt für Schritt in PyTorch und Google Colab.
Gruppenprojekt im Modul SOTA 2, Master Informatik an der Hochschule Offenburg.

## Arbeitsweise

### Branches

| Branch | Zweck |
|---|---|
| `main` | Abgestimmter Stand. Geschützt: Änderungen nur per Pull Request mit Review. |
| `Artur`, `Marvin`, `Timo`, `Valentin` | Persönlicher Arbeitsbranch. Jede Person arbeitet nur in ihrem eigenen. |

### Mit Colab arbeiten

1. Notebook aus dem **eigenen Branch** öffnen: in Colab *Datei → Notebook öffnen → GitHub*, Repo `Timbo3399/INFM_SOTA2` und den eigenen Branch wählen.
2. Speichern: *Datei → Kopie in GitHub speichern*, dabei **den eigenen Branch** auswählen.
   Speichern auf `main` schlägt fehl, das ist Absicht.
3. Aussagekräftige Commit-Message statt „Mit Colab erstellt“, z. B. `ch02: Tokenizer-Vergleich ergänzt`.

### Etwas nach `main` bringen

1. Auf GitHub: *Pull requests → New pull request*, `base: main` ← `compare: <eigener Branch>`.
2. Eine andere Person reviewt und approved.
3. Mergen mit **„Create a merge commit“**, nicht Squash. Sonst laufen die persönlichen Branches auseinander.

### Eigenen Branch aktualisieren

Wenn ihr die Änderungen der anderen braucht: *New pull request* mit `base: <eigener Branch>` ← `compare: main` und direkt mergen.
Lokal geht auch `git pull origin main` im eigenen Branch.

## Regeln

- **Ein Notebook, eine Person.** Notebooks sind JSON-Dateien, gleichzeitige Änderungen an derselben Datei lassen sich kaum mergen.
  Wer ein bestehendes Notebook erweitern will, legt eine Kopie an, z. B. `template_marvin.ipynb`, oder spricht sich vorher ab.
- **Keine Secrets im Code.** API-Tokens (Hugging Face, OpenAI, Kaggle …) nur über Colab-Secrets (Schlüssel-Symbol links):
  ```python
  from google.colab import userdata
  hf_token = userdata.get("HF_TOKEN")
  ```
  Tokens nie in eine Zelle schreiben und nie ausgeben. Ein versehentlich gepushter Token gilt als kompromittiert und muss sofort widerrufen werden.
- **Keine Modellgewichte oder Datensätze committen.** Die `.gitignore` fängt die üblichen Fälle ab (`*.pth`, `gpt2/`, `data/` …).
  Colab speichert nur das Notebook selbst, Dateien aus `/content` landen nicht im Repo.
- **Outputs klein halten.** Vor dem Speichern lange Ausgaben (Trainings-Logs, große Tabellen) löschen.
