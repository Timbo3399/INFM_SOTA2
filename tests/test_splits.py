import tempfile
from pathlib import Path
from unittest import TestCase

import torch

from infm_sota2.data import TextSplits, clean_book_text
from tests.helpers import MINI_BOOK, write_mini_book

SEQ_LEN = 8


class MiniBookCase(TestCase):
    """Mini-Buch in einem temporären Ordner, ohne eigene Tests."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        write_mini_book(self.dir)

    def tearDown(self):
        self.tmp.cleanup()

    def splits(self, max_chars: dict[str, int] | None = None) -> TextSplits:
        return TextSplits.from_book(
            self.dir,
            **MINI_BOOK,
            train_ratio=0.8,
            val_ratio=0.1,
            max_chars=max_chars,
            seq_len=SEQ_LEN,
        )


class TextSplitsTest(MiniBookCase):
    def test_splits_cover_the_cleaned_text(self):
        data = self.splits()

        raw = (self.dir / MINI_BOOK["file"]).read_text(encoding="utf-8")
        cleaned, _ = clean_book_text(
            raw,
            start_line=MINI_BOOK["start_line"],
            end_line=MINI_BOOK["end_line"],
            book_title=MINI_BOOK["book_title"],
        )
        self.assertEqual(data.train + data.val + data.test, cleaned)

    def test_max_chars_cuts_each_split(self):
        data = self.splits({"train": 300, "val": 100, "test": 100})

        self.assertEqual(
            (len(data.train), len(data.val), len(data.test)), (300, 100, 100)
        )

    def test_same_seed_gives_same_train_order(self):
        data = self.splits()

        first, _ = next(iter(data.train_loader(4, seed=1, rank=0, world_size=1)))
        again, _ = next(iter(data.train_loader(4, seed=1, rank=0, world_size=1)))
        other, _ = next(iter(data.train_loader(4, seed=2, rank=0, world_size=1)))

        self.assertTrue(torch.equal(first, again))
        self.assertFalse(torch.equal(first, other))

    def test_eval_loader_keeps_order(self):
        data = self.splits()

        inputs, _ = next(iter(data.loader("val", 4)))

        dataset = data.datasets["val"]
        expected = torch.stack([dataset[i][0] for i in range(4)])
        self.assertTrue(torch.equal(inputs, expected))


def step_rows(loaders: list) -> list[list[tuple[int, ...]]]:
    """Pro Schritt die Eingaben aller Ränge zusammen, als sortierte Zeilen."""
    steps = []
    for batches in zip(*loaders, strict=True):
        inputs = torch.cat([batch_inputs for batch_inputs, _ in batches])
        steps.append(sorted(tuple(row.tolist()) for row in inputs))
    return steps


class DistributedTrainLoaderTest(MiniBookCase):
    def test_same_windows_per_step_for_any_world_size(self):
        # Gesamt-Batch 8: Schritt k enthält bei jeder Zahl der Prozesse dieselben
        # Fenster, so rechnen 1, 2 und 4 Knoten in Lauf Nr. 5 dieselbe Arbeit
        data = self.splits()
        by_world_size = {
            p: step_rows([data.train_loader(8 // p, 123, rank, p) for rank in range(p)])
            for p in (1, 2, 4)
        }

        self.assertGreater(len(by_world_size[1]), 2)
        self.assertEqual(by_world_size[2], by_world_size[1])
        self.assertEqual(by_world_size[4], by_world_size[1])

    def test_set_epoch_changes_order(self):
        # Ohne set_epoch sähe jede Epoche dieselbe Reihenfolge
        loader = self.splits().train_loader(4, 123, 0, 1)
        first_epoch, _ = next(iter(loader))

        loader.sampler.set_epoch(1)  # type: ignore[attr-defined]
        second_epoch, _ = next(iter(loader))

        self.assertFalse(torch.equal(first_epoch, second_epoch))
