"""Ein selbst geschriebenes Mini-Buch für Tests und Smoke-Läufe.

Es hat denselben Aufbau wie ein umgewandeltes Buch (Vorspann, Erzählteil, Anhang),
ist bei allen gleich und frei von Urheberrechten. So läuft der Smoke-Lauf auch bei
Personen, die den Buchtext nicht haben.

Aufruf: ``python -m infm_sota2.data.mini_book <ordner>``
"""

import argparse
from pathlib import Path

# Marker und Titel, wie TextSplits.from_book sie aus der Konfiguration bekommt
MINI_BOOK = {
    "file": "mini-book.txt",
    "start_line": "PROLOGUE",
    "end_line": "APPENDIX A",
    "book_title": "THE BOOK OF TESTS",
}


def write_mini_book(directory: Path) -> str:
    """Legt ein Mini-Buch mit Vorspann, Erzählteil und Anhang an.

    Gibt den Dateinamen zurück."""
    paragraphs = [
        f"Sentence number {i} tells a short story about tests." for i in range(200)
    ]
    lines = ["CONTENTS", "", "PROLOGUE", "", "\n\n".join(paragraphs), "", "APPENDIX A"]
    (directory / MINI_BOOK["file"]).write_text("\n".join(lines), encoding="utf-8")
    return MINI_BOOK["file"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Mini-Buch in einen Ordner schreiben")
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    args.directory.mkdir(parents=True, exist_ok=True)
    name = write_mini_book(args.directory)
    print(f"Mini-Buch: {args.directory / name}")


if __name__ == "__main__":
    main()
