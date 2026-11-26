"""Text für das Training: Buchtext bereinigen und teilen, Token-Fenster und
Loader, dazu das Mini-Buch für Smoke-Läufe.

Die Regeln der Bereinigung und ihre Begründung stehen in
old/docs/lotr-training.md. Das Mini-Buch ist selbst geschrieben, bei allen gleich
und frei von Urheberrechten. So läuft der Smoke-Lauf auch bei Personen, die den
Buchtext nicht haben. ``python -m infm_sota2 mini-book <ordner>`` schreibt es.
"""

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# Kapitelname in Großbuchstaben plus Seitenzahl, wie OCR sie liest (I statt 1 usw.)
_ODD_PAGE_HEADER = re.compile(r"([A-Z][A-Z ,'\-]+) [0-9IOQSlZ]{1,5}")
_PAGE_NUMBER = re.compile(r"[0-9]{1,4}")
# Buchstabenfolgen inklusive Akzenten (Lórien, Éomer), ohne Ziffern und "_"
_WORD = re.compile(r"[^\W\d_]+")
_WORD_BEFORE_HYPHEN = re.compile(r"([^\W\d_]+)-$")
_SENTENCE_END = (".", "!", "?", '"', "”", "’", "'")
_EM_DASH = "—"
_PAGE_NUMBER_KEY = "(Seitenzahl)"


@dataclass(frozen=True)
class CleaningStats:
    """Zählt, was jeder Bereinigungsschritt geändert hat."""

    narrative_lines: int
    headers_removed: int
    header_titles: dict[str, int]
    page_breaks_joined: int
    page_breaks_paragraph: int
    hyphens_joined: int
    hyphens_kept: int
    paragraphs: int


def clean_book_text(
    text: str, start_line: str, end_line: str, book_title: str
) -> tuple[str, CleaningStats]:
    """Schneidet den Erzählteil aus und macht daraus Absätze ohne Seitenkopfzeilen.

    Erzählteil: von der Zeile ``start_line`` (inklusive) bis ``end_line`` (exklusive).
    Ergebnis: Absätze, getrennt durch eine Leerzeile.
    """
    lines = [line.strip() for line in text.split("\n")]
    start = _index_of_unique_line(lines, start_line)
    end = _index_of_unique_line(lines, end_line)
    if start >= end:
        raise ValueError(f"{start_line!r} must come before {end_line!r}")
    narrative = lines[start:end]

    even_page_header = re.compile(r"\S{1,6} " + re.escape(book_title))
    header_titles: Counter[str] = Counter()

    def header_title(line: str) -> str | None:
        if even_page_header.fullmatch(line):
            return book_title
        if match := _ODD_PAGE_HEADER.fullmatch(line):
            return match.group(1)
        if _PAGE_NUMBER.fullmatch(line):
            return _PAGE_NUMBER_KEY
        return None

    # Seitenwechsel (Kopfzeilen samt Leerzeilen drumherum) durch Fortsetzung
    # oder Absatzgrenze ersetzen. "" steht für eine Absatzgrenze.
    merged: list[str] = []
    joined = paragraph_breaks = 0
    i = 0
    while i < len(narrative):
        if header_title(narrative[i]) is None:
            merged.append(narrative[i])
            i += 1
            continue
        while merged and merged[-1] == "":
            merged.pop()
        while i < len(narrative):
            if narrative[i]:
                title = header_title(narrative[i])
                if title is None:
                    break
                header_titles[title] += 1
            i += 1
        if not merged or i == len(narrative):
            continue
        previous, following = merged[-1], narrative[i]
        # Überschriften (nur Großbuchstaben) enden nie im Satz
        continues = not previous.endswith(_SENTENCE_END) or following[:1].islower()
        if continues and not previous.isupper():
            joined += 1
        else:
            merged.append("")
            paragraph_breaks += 1

    # Kopfzeilen sind hier schon entfernt
    vocabulary = {word.lower() for line in merged for word in _WORD.findall(line)}
    paragraphs: list[str] = []
    hyphens_joined = hyphens_kept = 0
    block: list[str] = []
    for line in [*merged, ""]:
        if line:
            block.append(line)
            continue
        if not block:
            continue
        paragraph = block[0]
        for following in block[1:]:
            left = _WORD_BEFORE_HYPHEN.search(paragraph)
            right = _WORD.match(following)
            if left and right and following[:1].islower():
                if (left.group(1) + right.group(0)).lower() in vocabulary:
                    paragraph = paragraph[:-1] + following
                    hyphens_joined += 1
                else:
                    paragraph += following
                    hyphens_kept += 1
            elif paragraph.endswith(_EM_DASH):
                paragraph += following
            else:
                paragraph += " " + following
        paragraphs.append(paragraph)
        block = []

    stats = CleaningStats(
        narrative_lines=len(narrative),
        headers_removed=sum(header_titles.values()),
        header_titles=dict(header_titles),
        page_breaks_joined=joined,
        page_breaks_paragraph=paragraph_breaks,
        hyphens_joined=hyphens_joined,
        hyphens_kept=hyphens_kept,
        paragraphs=len(paragraphs),
    )
    return "\n\n".join(paragraphs), stats


def split_text(text: str, train_ratio: float, val_ratio: float) -> tuple[str, str, str]:
    """Teilt in drei zusammenhängende Blöcke, jeweils direkt hinter einer Absatzgrenze.

    Es gilt ``train + val + test == text``. Der Test-Anteil ist ``1 - train - val``.
    """
    if train_ratio <= 0 or val_ratio <= 0:
        raise ValueError("train_ratio and val_ratio must be positive")
    if train_ratio + val_ratio >= 1:
        raise ValueError(
            f"train_ratio + val_ratio = {train_ratio + val_ratio} leaves no test part"
        )
    first = _cut_after_paragraph(text, int(len(text) * train_ratio))
    second = _cut_after_paragraph(text, int(len(text) * (train_ratio + val_ratio)))
    if not 0 < first < second < len(text):
        raise ValueError("Text too short to split into three non-empty parts")
    return text[:first], text[first:second], text[second:]


def _index_of_unique_line(lines: list[str], marker: str) -> int:
    count = lines.count(marker)
    if count != 1:
        raise ValueError(
            f"Marker line {marker!r} must occur exactly once, found {count}"
        )
    return lines.index(marker)


def _cut_after_paragraph(text: str, position: int) -> int:
    boundary = text.find("\n\n", position)
    if boundary == -1:
        raise ValueError(f"No paragraph boundary after character {position}")
    return boundary + 2


class BPEDataset(Dataset):
    """Sliding Window Dataset mit BPE Tokenization."""

    def __init__(
        self,
        text: str,
        seq_len: int = 128,
        stride: int | None = None,
        encoding: str = "gpt2",
    ):
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if stride is not None and stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")

        self.tokenizer = tiktoken.get_encoding(encoding)
        self.tokens = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        if len(self.tokens) < seq_len + 1:
            raise ValueError(
                f"Text must contain at least {seq_len + 1} tokens, "
                f"got {len(self.tokens)}"
            )

        self.seq_len = seq_len
        self.stride = seq_len if stride is None else stride
        self.n_samples = (len(self.tokens) - seq_len - 1) // self.stride + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        seq = self.tokens[start : start + self.seq_len + 1]
        return seq[:-1], seq[1:]


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
    parser = argparse.ArgumentParser(
        prog="python -m infm_sota2 mini-book",
        description="Mini-Buch in einen Ordner schreiben",
    )
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    args.directory.mkdir(parents=True, exist_ok=True)
    name = write_mini_book(args.directory)
    print(f"Mini-Buch: {args.directory / name}")
