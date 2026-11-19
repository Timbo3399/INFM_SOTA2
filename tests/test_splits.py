import tempfile
from pathlib import Path
from unittest import TestCase

import torch

from infm_sota2.data import TextSplits, clean_book_text
from tests.helpers import MINI_BOOK, write_mini_book

SEQ_LEN = 8


class TextSplitsTest(TestCase):
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

        first, _ = next(iter(data.train_loader(4, seed=1)))
        again, _ = next(iter(data.train_loader(4, seed=1)))
        other, _ = next(iter(data.train_loader(4, seed=2)))

        self.assertTrue(torch.equal(first, again))
        self.assertFalse(torch.equal(first, other))

    def test_eval_loader_keeps_order(self):
        data = self.splits()

        inputs, _ = next(iter(data.loader("val", 4)))

        dataset = data.datasets["val"]
        expected = torch.stack([dataset[i][0] for i in range(4)])
        self.assertTrue(torch.equal(inputs, expected))

    def test_summary_lists_cleaning_counts(self):
        data = self.splits()

        self.assertIn(
            f"Absätze:                       {data.stats.paragraphs:,}", data.summary()
        )

    def test_split_summary_names_steps_per_epoch(self):
        data = self.splits()

        self.assertTrue(
            data.split_summary(4).endswith(
                f"Schritte pro Epoche: {len(data.train_loader(4, seed=1))}"
            )
        )
