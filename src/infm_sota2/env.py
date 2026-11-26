"""Laufzeitumgebung: Ordner, Gerät, Rang unter torchrun und W&B-Modus."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

import torch

WANDB_ENTITY = "hochschule-offenburg"
WANDB_PROJECT = "SOTA_2"

WandbMode = Literal["online", "offline", "disabled"]


@dataclass(frozen=True)
class Environment:
    base_dir: Path
    device: torch.device
    wandb_mode: WandbMode
    data_dir: Path
    # Unter torchrun aus RANK und WORLD_SIZE, sonst Rang 0 von 1
    rank: int
    world_size: int

    @classmethod
    def detect(cls, smoke_test: bool) -> "Environment":
        """Alles liegt unter dem Arbeitsordner. ``smoke_test`` schaltet W&B ab.

        Zwei Umgebungsvariablen gelten, wenn sie gesetzt und nicht leer sind:
        ``WANDB_MODE`` (``online``, ``offline`` oder ``disabled``, außer im
        Smoke-Test) und ``SOTA2_DATA_DIR`` als Datenordner statt ``data/``.
        Mit GPU ist das Gerät ``cuda:<LOCAL_RANK>``, ohne torchrun ``cuda:0``."""
        base_dir = Path.cwd()
        rank, local_rank, world_size = _torchrun_ranks()
        if torch.cuda.is_available():
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        wandb_mode = "disabled" if smoke_test else _wandb_mode_from_environment()
        # Leer zählt wie nicht gesetzt, wie ${SOTA2_DATA_DIR:-data} in setup.sh
        data_dir = Path(os.environ.get("SOTA2_DATA_DIR") or base_dir / "data")
        return cls(base_dir, device, wandb_mode, data_dir, rank, world_size)

    @property
    def checkpoint_dir(self) -> Path:
        return self.base_dir / "checkpoints"

    @property
    def gpu_name(self) -> str | None:
        if self.device.type != "cuda":
            return None
        return torch.cuda.get_device_name(self.device)


def _wandb_mode_from_environment() -> WandbMode:
    """``WANDB_MODE``, falls gesetzt und nicht leer, sonst ``online``."""
    mode = os.environ.get("WANDB_MODE") or "online"
    for allowed in get_args(WandbMode):
        if mode == allowed:
            return allowed
    raise ValueError(
        f"WANDB_MODE is {mode!r}, allowed: {', '.join(get_args(WandbMode))}"
    )


_TORCHRUN_VARIABLES = ("RANK", "LOCAL_RANK", "WORLD_SIZE")


def _torchrun_ranks() -> tuple[int, int, int]:
    """Rang, lokaler Rang und Zahl der Prozesse, ohne torchrun ``(0, 0, 1)``.

    Fehlen alle drei Variablen, ist es ein einzelner Prozess. Fehlt nur ein Teil,
    ist der Start kaputt, und das bricht ab, statt still Rang 0 von 1 anzunehmen.
    """
    present = [name for name in _TORCHRUN_VARIABLES if name in os.environ]
    if not present:
        return 0, 0, 1
    if len(present) != len(_TORCHRUN_VARIABLES):
        raise ValueError(
            f"torchrun variables incomplete: {', '.join(_TORCHRUN_VARIABLES)} "
            f"needed, found only {', '.join(present)}"
        )
    rank, local_rank, world_size = (int(os.environ[n]) for n in _TORCHRUN_VARIABLES)
    return rank, local_rank, world_size
