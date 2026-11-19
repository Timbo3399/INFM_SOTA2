"""Gemeinsame Test-Bausteine: ein winziges GPTModel, das auf der CPU schnell ist,
und ein selbst geschriebenes Mini-Buch."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from infm_sota2 import GPTModel

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


# Marker und Titel, wie TextSplits.from_book sie aus CONFIG["data"] bekommt
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
