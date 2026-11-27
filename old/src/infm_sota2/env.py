"""Laufzeitumgebung der Notebooks: Colab oder lokal, Pfade, Gerät und W&B."""

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import requires, version
from pathlib import Path
from typing import Literal, get_args

import torch
import wandb

WANDB_ENTITY = "hochschule-offenburg"
WANDB_PROJECT = "SOTA_2"

WandbMode = Literal["online", "offline", "disabled"]

_COLAB_BASE_DIR = Path("/content/drive/MyDrive/INFM_SOTA2")
_COLAB_RUN_DIR = Path("/content/runs")  # schnelle lokale Platte während des Trainings


def version_mismatches(
    requirements: list[str], installed: Callable[[str], str]
) -> list[str]:
    """Eine Meldung pro Paket, dessen Version nicht zum Pin ``name==version`` passt.

    Ein lokales Suffix wie ``+cu130`` zählt nicht als Abweichung.
    """
    messages = []
    for requirement in requirements:
        name, _, wanted = requirement.partition("==")
        have = installed(name).split("+")[0]
        if have != wanted:
            messages.append(
                f"Achtung: {name} {have} installiert, pyproject.toml erwartet {wanted}"
            )
    return messages


@dataclass(frozen=True)
class Environment:
    in_colab: bool
    base_dir: Path
    run_dir: Path
    device: torch.device
    wandb_mode: WandbMode
    data_dir: Path
    # Unter torchrun aus RANK, LOCAL_RANK und WORLD_SIZE, sonst Rang 0 von 1
    rank: int
    local_rank: int
    world_size: int
    launched_by_torchrun: bool

    @classmethod
    def detect(cls, smoke_test: bool) -> "Environment":
        """Lokal liegt alles unter dem Arbeitsordner (beim Notebook ``notebooks/``),
        in Colab auf Google Drive. ``smoke_test`` schaltet W&B ab.

        Zwei Umgebungsvariablen gelten, wenn sie gesetzt und nicht leer sind:
        ``WANDB_MODE`` (``online``, ``offline`` oder ``disabled``, außer im
        Smoke-Test) und ``SOTA2_DATA_DIR`` als Datenordner statt ``data/``.
        Unter ``torchrun`` ist das Gerät ``cuda:<LOCAL_RANK>``."""
        in_colab = "google.colab" in sys.modules
        if in_colab:
            from google.colab import drive  # type: ignore

            drive.mount("/content/drive")
            base_dir, run_dir = _COLAB_BASE_DIR, _COLAB_RUN_DIR
        else:
            base_dir = Path.cwd()
            run_dir = base_dir / "checkpoints" / "runs"
        rank, local_rank, world_size, launched_by_torchrun = _torchrun_ranks()
        if not torch.cuda.is_available():
            device = torch.device("cpu")
        elif launched_by_torchrun:
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda")
        wandb_mode = "disabled" if smoke_test else _wandb_mode_from_environment()
        # Leer zählt wie nicht gesetzt, wie ${SOTA2_DATA_DIR:-data} in setup.sh
        data_dir = Path(os.environ.get("SOTA2_DATA_DIR") or base_dir / "data")
        return cls(
            in_colab,
            base_dir,
            run_dir,
            device,
            wandb_mode,
            data_dir,
            rank,
            local_rank,
            world_size,
            launched_by_torchrun,
        )

    @property
    def checkpoint_dir(self) -> Path:
        """Bleibender Ort für Checkpoints, in Colab auf Drive."""
        return self.base_dir / "checkpoints"

    @property
    def gpu_name(self) -> str | None:
        if self.device.type != "cuda":
            return None
        return torch.cuda.get_device_name(self.device)

    def login_wandb(self) -> None:
        if self.in_colab:
            from google.colab import userdata  # type: ignore

            wandb.login(key=userdata.get("WANDB_API_KEY"))
        else:
            wandb.login()  # nutzt den Key aus `uv run wandb login`

    def check_versions(self) -> list[str]:
        """Vergleicht torch, tiktoken und wandb mit den Pins aus pyproject.toml."""
        return version_mismatches(requires("infm-sota2") or [], version)

    def describe(self) -> dict[str, object]:
        return {
            "device": self.device.type,
            "gpu": self.gpu_name,
            "torch": torch.__version__,
            "colab": self.in_colab,
        }


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


def _torchrun_ranks() -> tuple[int, int, int, bool]:
    """Rang, lokaler Rang, Zahl der Prozesse und ob torchrun den Prozess startete.

    Fehlen alle drei Variablen, ist es ein einzelner Prozess. Fehlt nur ein Teil,
    ist der Start kaputt, und das bricht ab, statt still Rang 0 von 1 anzunehmen.
    """
    present = [name for name in _TORCHRUN_VARIABLES if name in os.environ]
    if not present:
        return 0, 0, 1, False
    if len(present) != len(_TORCHRUN_VARIABLES):
        raise ValueError(
            f"torchrun variables incomplete: {', '.join(_TORCHRUN_VARIABLES)} "
            f"needed, found only {', '.join(present)}"
        )
    rank, local_rank, world_size = (int(os.environ[n]) for n in _TORCHRUN_VARIABLES)
    return rank, local_rank, world_size, True
