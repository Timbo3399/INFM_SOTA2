"""Buchtext bereinigen, in Training, Validierung und Test teilen und Loader bauen."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tiktoken
import torch
from torch.utils.data import DataLoader

from infm_sota2.data.cleaning import CleaningStats, clean_book_text, split_text
from infm_sota2.data.dataset import BPEDataset

Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class TextSplits:
    train: str
    val: str
    test: str
    stats: CleaningStats
    raw_chars: int
    cleaned_chars: int
    max_chars: dict[str, int] | None
    seq_len: int
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
        text, stats = clean_book_text(
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
        return cls(
            train=train,
            val=val,
            test=test,
            stats=stats,
            raw_chars=len(raw_text),
            cleaned_chars=len(text),
            max_chars=max_chars,
            seq_len=seq_len,
            datasets=datasets,
        )

    @property
    def tokenizer(self) -> tiktoken.Encoding:
        return self.datasets["train"].tokenizer

    def train_loader(self, batch_size: int, seed: int) -> DataLoader:
        """Gemischt mit festem Seed, jeder Lauf sieht dieselbe Reihenfolge."""
        return DataLoader(
            self.datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            generator=torch.Generator().manual_seed(seed),
        )

    def loader(self, split: Literal["val", "test"], batch_size: int) -> DataLoader:
        return DataLoader(self.datasets[split], batch_size=batch_size)

    def summary(self) -> str:
        """Was die Bereinigung geändert hat."""
        stats = self.stats
        lines = [
            f"Zeichen roh: {self.raw_chars:,}, bereinigt: {self.cleaned_chars:,}",
            f"Zeilen im Erzählteil:          {stats.narrative_lines:,}",
            f"Kopfzeilen entfernt:           {stats.headers_removed:,}",
            f"Seitenwechsel zusammengefügt:  {stats.page_breaks_joined:,}",
            f"Seitenwechsel als Absatz:      {stats.page_breaks_paragraph:,}",
            f"Silbentrennungen aufgelöst:    {stats.hyphens_joined:,}",
            f"Bindestriche behalten:         {stats.hyphens_kept:,}",
            f"Absätze:                       {stats.paragraphs:,}",
            "",
            "Entfernte Kopfzeilen nach Titel:",
        ]
        by_count = sorted(stats.header_titles.items(), key=lambda item: -item[1])
        lines += [f"  {count:4d}  {title}" for title, count in by_count]
        return "\n".join(lines)

    def split_summary(self, batch_size: int) -> str:
        """Tokens und Fenster pro Teil, dazu die Schritte pro Epoche."""
        lines = []
        if self.max_chars is not None:
            lines.append(f"SMOKE_TEST: Texte gekürzt auf {self.max_chars}")
        for split, dataset in self.datasets.items():
            lines.append(
                f"{split:5s}: {len(dataset.tokens):>9,} Tokens, "
                f"{len(dataset):>6,} Fenster à {self.seq_len}"
            )
        lines.append(
            f"Schritte pro Epoche: {len(self.train_loader(batch_size, seed=0))}"
        )
        return "\n".join(lines)
