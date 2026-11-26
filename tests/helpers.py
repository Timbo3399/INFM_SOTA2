"""Gemeinsame Test-Bausteine: ein winziges GPTModel, das auf der CPU schnell ist,
das selbst geschriebene Mini-Buch aus dem Paket und eine lokale Umgebung."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from infm_sota2.data.mini_book import MINI_BOOK, write_mini_book
from infm_sota2.env import Environment
from infm_sota2.model import GPTModel

__all__ = [
    "MINI_BOOK",
    "TINY_CONFIG",
    "local_env",
    "pattern_loader",
    "tiny_model",
    "write_mini_book",
]

TINY_CONFIG = {
    "vocab_size": 20,
    "context_length": 8,
    "emb_dim": 16,
    "n_heads": 2,
    "n_layers": 2,
    "drop_rate": 0.0,
    "qkv_bias": True,
}


def tiny_model(seed: int, **overrides: int | float | bool) -> GPTModel:
    torch.manual_seed(seed)
    return GPTModel({**TINY_CONFIG, **overrides})


def pattern_loader(n_windows: int, batch_size: int) -> DataLoader:
    """Fenster aus der Folge 0, 1, ..., 9, 0, 1, ...: das nächste Token ist immer +1."""
    seq_len = TINY_CONFIG["context_length"]
    tokens = torch.arange(n_windows * seq_len + 1) % 10
    inputs = tokens[:-1].view(n_windows, seq_len)
    targets = tokens[1:].view(n_windows, seq_len)
    return DataLoader(TensorDataset(inputs, targets), batch_size=batch_size)


def local_env(base_dir: Path) -> Environment:
    """Auf der CPU, ohne W&B, alle Ordner unter ``base_dir``."""
    return Environment(
        base_dir=base_dir,
        device=torch.device("cpu"),
        wandb_mode="disabled",
        data_dir=base_dir / "data",
        rank=0,
        world_size=1,
    )
