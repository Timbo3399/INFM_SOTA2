"""Buchtext bereinigen, in Training, Validierung und Test teilen und Loader bauen."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tiktoken
from torch.utils.data import DataLoader, DistributedSampler

from infm_sota2.data.cleaning import clean_book_text, split_text
from infm_sota2.data.dataset import BPEDataset

Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class TextSplits:
    train: str
    val: str
    test: str
    datasets: dict[Split, BPEDataset]

    @classmethod
    def from_book(
        cls,
        data_dir: Path,
        file: str,
        start_line: str,
        end_line: str,
        book_title: str,
        train_ratio: float,
        val_ratio: float,
        max_chars: dict[str, int] | None,
        seq_len: int,
    ) -> "TextSplits":
        """Liest ``data_dir / file``, bereinigt den Erzählteil und teilt ihn an
        Absatzgrenzen. ``max_chars`` kürzt jeden Teil (nur für den Smoke-Test)."""
        raw_text = (data_dir / file).read_text(encoding="utf-8")
        text, _ = clean_book_text(
            raw_text, start_line=start_line, end_line=end_line, book_title=book_title
        )
        train, val, test = split_text(text, train_ratio, val_ratio)
        if max_chars is not None:
            train = train[: max_chars["train"]]
            val = val[: max_chars["val"]]
            test = test[: max_chars["test"]]
        datasets: dict[Split, BPEDataset] = {
            "train": BPEDataset(train, seq_len=seq_len, stride=seq_len),
            "val": BPEDataset(val, seq_len=seq_len, stride=seq_len),
            "test": BPEDataset(test, seq_len=seq_len, stride=seq_len),
        }
        return cls(train=train, val=val, test=test, datasets=datasets)

    @property
    def tokenizer(self) -> tiktoken.Encoding:
        return self.datasets["train"].tokenizer

    def train_loader(
        self, batch_size: int, seed: int, rank: int, world_size: int
    ) -> DataLoader:
        """Train-Loader für Rang ``rank`` von ``world_size``, ``batch_size`` Fenster
        pro Prozess. Braucht keine Prozessgruppe, ohne torchrun ist es Rang 0 von 1.

        Der Sampler mischt alle Fenster mit ``seed + epoch`` und gibt Rang r die
        Fenster r, r + p, r + 2p usw. Bei gleicher Gesamt-Batchgröße B enthält
        Schritt k deshalb bei jeder Zahl p der Prozesse die Fenster k·B bis
        (k+1)·B − 1 der gemischten Reihenfolge. Vor jeder Epoche muss
        ``sampler.set_epoch(epoch)`` laufen, sonst bleibt die Reihenfolge gleich.
        """
        sampler = DistributedSampler(
            self.datasets["train"],
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=True,
        )
        return DataLoader(
            self.datasets["train"],
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
        )

    def loader(self, split: Literal["val", "test"], batch_size: int) -> DataLoader:
        return DataLoader(self.datasets[split], batch_size=batch_size)
