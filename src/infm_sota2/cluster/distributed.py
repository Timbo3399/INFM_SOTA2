"""Prozessgruppe für mehrere Prozesse unter ``torchrun``.

Ein einzelner Prozess läuft ohne Prozessgruppe und ohne DDP. Er ist die Basis für
den Speedup in Lauf Nr. 5. Ab zwei Prozessen nimmt die Gruppe NCCL auf der GPU und
Gloo auf der CPU, so lässt sich der Code lokal mit
``torchrun --nproc-per-node=2`` testen.
"""

import torch
import torch.distributed as dist

from infm_sota2.env import Environment


def init_distributed(env: Environment) -> None:
    """Setzt die GPU des Prozesses und startet ab zwei Prozessen die Gruppe."""
    if env.device.type == "cuda":
        torch.cuda.set_device(env.device)
    if env.world_size > 1:
        backend = "nccl" if env.device.type == "cuda" else "gloo"
        dist.init_process_group(backend, rank=env.rank, world_size=env.world_size)


def destroy_distributed(env: Environment) -> None:
    if env.world_size > 1:
        dist.destroy_process_group()
