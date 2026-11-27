"""Ergebnisse der Läufe als JSON-Dateien und die Tabellen der Auswertung.

Jeder Lauf schreibt ``results/week06/<benutzer>/<gruppe>/<lauf>-job<id>.json``. Die
Job-ID im Namen sorgt dafür, dass eine Wiederholung nichts überschreibt. Die
Tabellen lassen gescheiterte und fehlende Läufe sichtbar, statt sie wegzulassen,
und brechen ab, wenn ein Lauf doppelt vorliegt.
"""

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def result_path(
    base_dir: Path, user: str, group: str, label: str, job_id: str | None
) -> Path:
    """Ohne Job-ID, also außerhalb von Slurm, endet der Name auf ``-local.json``."""
    name = f"{label}-job{job_id}.json" if job_id is not None else f"{label}-local.json"
    return base_dir / "results" / "week06" / user / group / name


def write_result(path: Path, record: dict[str, object]) -> None:
    """Schreibt ``record`` als JSON und überschreibt nie eine vorhandene Datei."""
    text = json.dumps(record, indent=2, ensure_ascii=False)  # vor dem Öffnen
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        file.write(text + "\n")


def load_results(directory: Path) -> list[dict[str, object]]:
    """Alle JSON-Dateien unter ``directory``, sortiert, jede mit ``file``."""
    records = []
    for path in sorted(directory.rglob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        records.append(record | {"file": str(path)})
    return records


def git_commit(repo_dir: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def scaling_table(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Speedup und Effizienz pro Zahl der GPUs (Folie 5).

    S_p = Tokens/s mit p GPUs geteilt durch Tokens/s mit einer GPU, E_p = S_p / p.
    """
    by_size = _unique(records, lambda record: _get(record, "environment.world_size"))
    single = by_size.get(1)
    if single is None or _tokens_per_second(single) is None:
        raise ValueError("speedup needs a successful run with one GPU")
    baseline = _tokens_per_second(single)
    assert baseline is not None
    rows = []
    for world_size in sorted(by_size):
        record = by_size[world_size]
        tokens_per_second = _tokens_per_second(record)
        speedup = None if tokens_per_second is None else tokens_per_second / baseline
        rows.append(
            {
                "world_size": world_size,
                "tokens_per_second": tokens_per_second,
                "speedup": speedup,
                "efficiency": None if speedup is None else speedup / world_size,
                "status": _status(record),
                "file": record["file"],
            }
        )
    return rows


def search_table(
    records: list[dict[str, object]],
    learning_rates: Sequence[float],
    batch_sizes: Sequence[int],
) -> list[dict[str, object]]:
    """Eine Zeile pro Kombination, zeilenweise wie ``with_grid``.

    Fehlt eine Kombination, steht ``"fehlt"`` als Status da, etwa weil Slurm den
    Lauf nach Ablauf der Zeit beendet hat, bevor er eine Datei schreiben konnte.
    """
    by_combination = _unique(
        records,
        lambda record: (
            _get(record, "config.train.learning_rate"),
            _get(record, "config.train.batch_size"),
        ),
    )
    grid = [(lr, bs) for lr in learning_rates for bs in batch_sizes]
    outside = [by_combination[key]["file"] for key in by_combination if key not in grid]
    if outside:
        raise ValueError(
            f"runs outside the search grid: {', '.join(map(str, outside))}"
        )
    rows = []
    for learning_rate, batch_size in grid:
        record = by_combination.get((learning_rate, batch_size))
        failed = record is None or record["error"] is not None
        rows.append(
            {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "best_val_loss": None
                if failed
                else _get(record, "result.best_val_loss"),
                "status": "fehlt" if record is None else _status(record),
                "file": None if record is None else record["file"],
            }
        )
    return rows


def _get(record: object, dotted: str) -> Any:
    """Wert unter ``"config.train.batch_size"``, ein fehlender Schlüssel bricht ab."""
    value = record
    for key in dotted.split("."):
        if not isinstance(value, dict):
            raise KeyError(f"{dotted}: {key!r} is not inside a table")
        value = value[key]
    return value


def _unique(
    records: list[dict[str, object]], key_of: Any
) -> dict[Any, dict[str, object]]:
    found: dict[Any, dict[str, object]] = {}
    for record in records:
        key = key_of(record)
        if key in found:
            raise ValueError(
                f"two runs for {key}: {found[key]['file']} and {record['file']}"
            )
        found[key] = record
    return found


def _tokens_per_second(record: dict[str, object]) -> float | None:
    if record["error"] is not None:
        return None
    return _get(record, "result.tokens_per_second")


def _status(record: dict[str, object]) -> str:
    error = record["error"]
    return "ok" if error is None else str(error)
