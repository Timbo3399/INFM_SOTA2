"""Führt ein Notebook mit ``SMOKE_TEST = True`` aus und speichert alle Ausgaben.

Die Ausgaben landen in einer Textdatei. So lässt sich prüfen, dass eine Änderung
an gemeinsamem Code keine Zahl im Notebook ändert: einmal vor und einmal nach der
Änderung ausführen und die Dateien mit ``diff`` vergleichen. Das Notebook selbst
wird nicht gespeichert.

Aufruf (nbclient steht bewusst nicht in den Abhängigkeiten)::

    uv run --with nbclient python scripts/smoke_notebook.py <notebook> <ausgabe>
"""

import argparse
from pathlib import Path

import nbformat

SMOKE_OFF = "SMOKE_TEST = False"
SMOKE_ON = "SMOKE_TEST = True"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Notebook mit SMOKE_TEST = True ausführen, Ausgaben speichern."
    )
    parser.add_argument("notebook", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)

    from nbclient import NotebookClient  # type: ignore[import-not-found]

    notebook = nbformat.read(args.notebook, as_version=4)
    switched = 0
    for cell in notebook.cells:
        if cell.cell_type == "code" and cell.source.strip() == SMOKE_OFF:
            cell.source = SMOKE_ON
            switched += 1
    if switched != 1:
        raise ValueError(
            f"{args.notebook} needs exactly one cell '{SMOKE_OFF}', found {switched}"
        )

    resources = {"metadata": {"path": str(args.notebook.parent)}}
    NotebookClient(notebook, timeout=900, resources=resources).execute()

    with args.output.open("w", encoding="utf-8") as out:
        for cell in notebook.cells:
            for output in cell.get("outputs", []):
                if output.output_type == "stream" and output.name == "stdout":
                    out.write(output.text)
    print(f"Ausgaben geschrieben nach {args.output}")


if __name__ == "__main__":
    main()
