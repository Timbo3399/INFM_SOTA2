"""Ein Einstieg für alle Befehle: ``python -m infm_sota2 <befehl> [optionen]``.

Unter torchrun heißt er ``torchrun ... -m infm_sota2 train --config <datei>``.
Die Optionen eines Befehls zeigt ``python -m infm_sota2 <befehl> -h``.
"""

import sys
from collections.abc import Callable

from infm_sota2 import data
from infm_sota2.cluster import benchmark, check, train

COMMANDS: dict[str, Callable[[list[str] | None], None]] = {
    "check": check.main,  # Job 1: Umgebung prüfen
    "benchmark": benchmark.main,  # Job 1: Lauf Nr. 1
    "train": train.main,  # ein Lauf nach einer TOML-Datei
    "mini-book": data.main,  # Mini-Buch für den Smoke-Lauf
}
USAGE = f"Aufruf: python -m infm_sota2 {{{','.join(COMMANDS)}}} [optionen]"


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] in ("-h", "--help"):
        print(USAGE)
        return
    if not args or args[0] not in COMMANDS:
        print(USAGE, file=sys.stderr)
        raise SystemExit(2)
    COMMANDS[args[0]](args[1:])


if __name__ == "__main__":
    main()
