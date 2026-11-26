import io
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest import TestCase

import tiktoken
import torch

import tests.helpers
from infm_sota2.data import (
    MINI_BOOK,
    BPEDataset,
    TextSplits,
    clean_book_text,
    main,
    split_text,
    write_mini_book,
)

TITLE = "THE BOOK OF TESTS"


def book(*body_lines: str) -> str:
    """Mini-Buch mit Vorspann, Erzählteil und Anhang, wie die echte Datei aufgebaut."""
    return "\n".join(
        ["CONTENTS", "PROLOGUE 1", "", "PROLOGUE", "", *body_lines, "", "APPENDIX A"]
        + ["Tables and dates.", ""]
    )


def clean(text: str):
    return clean_book_text(
        text, start_line="PROLOGUE", end_line="APPENDIX A", book_title=TITLE
    )


class CleanBookTextTest(TestCase):
    def test_keeps_only_text_between_markers(self):
        cleaned, _ = clean(book("The story starts here."))

        self.assertEqual(cleaned, "PROLOGUE\n\nThe story starts here.")

    def test_rejects_missing_marker(self):
        with self.assertRaisesRegex(ValueError, "APPENDIX A.*0"):
            clean("PROLOGUE\n\nNo appendix in this text.")

    def test_rejects_duplicate_marker(self):
        with self.assertRaisesRegex(ValueError, "PROLOGUE.*2"):
            clean(book("PROLOGUE", "", "Twice."))

    def test_removes_even_and_odd_page_headers(self):
        text = book(
            "A short sentence.",
            "",
            f"22 {TITLE}",
            "",
            "Another sentence.",
            "",
            "THE FIRST CHAPTER 23",
            "",
            "A last sentence.",
        )

        cleaned, stats = clean(text)

        self.assertNotIn(TITLE, cleaned)
        self.assertNotIn("CHAPTER 23", cleaned)
        self.assertEqual(stats.headers_removed, 2)
        self.assertEqual(stats.header_titles, {TITLE: 1, "THE FIRST CHAPTER": 1})

    def test_removes_page_number_lines(self):
        cleaned, stats = clean(book("One sentence.", "", "1086", "", "Two sentences."))

        self.assertEqual(cleaned, "PROLOGUE\n\nOne sentence.\n\nTwo sentences.")
        self.assertEqual(stats.headers_removed, 1)

    def test_joins_sentence_across_page_break(self):
        text = book("The road went on", "", f"22 {TITLE}", "", "and on until night.")

        cleaned, stats = clean(text)

        self.assertIn("The road went on and on until night.", cleaned)
        self.assertEqual(stats.page_breaks_joined, 1)
        self.assertEqual(stats.page_breaks_paragraph, 0)

    def test_page_break_after_full_sentence_starts_paragraph(self):
        text = book("The road ended.", "", f"22 {TITLE}", "", "Night came.")

        cleaned, stats = clean(text)

        self.assertIn("The road ended.\n\nNight came.", cleaned)
        self.assertEqual(stats.page_breaks_joined, 0)
        self.assertEqual(stats.page_breaks_paragraph, 1)

    def test_unwraps_lines_into_paragraphs(self):
        text = book("The first line", "of a paragraph.", "", "A second", "paragraph.")

        cleaned, stats = clean(text)

        self.assertEqual(
            cleaned, "PROLOGUE\n\nThe first line of a paragraph.\n\nA second paragraph."
        )
        self.assertEqual(stats.paragraphs, 3)

    def test_dehyphenates_known_word(self):
        text = book("They found some-", "thing old.", "", "Something new.")

        cleaned, stats = clean(text)

        self.assertIn("They found something old.", cleaned)
        self.assertEqual(stats.hyphens_joined, 1)
        self.assertEqual(stats.hyphens_kept, 0)

    def test_keeps_hyphen_of_compound(self):
        cleaned, stats = clean(book("It was a long-", "expected party."))

        self.assertIn("It was a long-expected party.", cleaned)
        self.assertEqual(stats.hyphens_joined, 0)
        self.assertEqual(stats.hyphens_kept, 1)

    def test_dehyphenates_across_page_break(self):
        text = book(
            "They found some-", "", f"22 {TITLE}", "", "thing.", "", "Something."
        )

        cleaned, _ = clean(text)

        self.assertIn("They found something.", cleaned)

    def test_handles_hyphen_before_accented_letter(self):
        text = book("They met in Lóth-", "lórien.", "", "Lóthlórien again.")

        cleaned, stats = clean(text)

        self.assertIn("They met in Lóthlórien.", cleaned)
        self.assertEqual(stats.hyphens_joined, 1)

    def test_heading_before_page_break_starts_paragraph(self):
        text = book("BOOK FOUR", "", f"22 {TITLE}", "", "Frodo woke.")

        cleaned, stats = clean(text)

        self.assertIn("BOOK FOUR\n\nFrodo woke.", cleaned)
        self.assertEqual(stats.page_breaks_paragraph, 1)

    def test_joins_em_dash_without_space(self):
        cleaned, _ = clean(book("He stopped —", "and listened."))

        self.assertIn("He stopped —and listened.", cleaned)

    def test_counts_narrative_lines(self):
        _, stats = clean(book("One.", "Two."))

        # PROLOGUE, "", One., Two., "" vor dem Anhang
        self.assertEqual(stats.narrative_lines, 5)


class SplitTextTest(TestCase):
    def setUp(self):
        self.text = "\n\n".join(
            f"Paragraph number {i} has some words." for i in range(100)
        )

    def test_split_preserves_all_text(self):
        train, val, test = split_text(self.text, train_ratio=0.8, val_ratio=0.1)

        self.assertEqual(train + val + test, self.text)

    def test_split_cuts_at_paragraph_boundaries(self):
        train, val, test = split_text(self.text, train_ratio=0.8, val_ratio=0.1)

        self.assertTrue(train.endswith("\n\n"))
        self.assertTrue(val.endswith("\n\n"))
        self.assertTrue(val.startswith("Paragraph"))
        self.assertTrue(test.startswith("Paragraph"))

    def test_split_follows_ratios(self):
        train, val, test = split_text(self.text, train_ratio=0.8, val_ratio=0.1)

        total = len(self.text)
        self.assertAlmostEqual(len(train) / total, 0.8, delta=0.02)
        self.assertAlmostEqual(len(val) / total, 0.1, delta=0.02)
        self.assertAlmostEqual(len(test) / total, 0.1, delta=0.02)

    def test_split_rejects_ratios_without_test_part(self):
        with self.assertRaisesRegex(ValueError, "test"):
            split_text(self.text, train_ratio=0.9, val_ratio=0.1)

    def test_split_rejects_text_without_paragraph_boundary(self):
        with self.assertRaisesRegex(ValueError, "paragraph"):
            split_text("one long paragraph " * 50, train_ratio=0.8, val_ratio=0.1)


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


class TrainLoaderTest(MiniBookCase):
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


class MiniBookTest(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_writes_book_that_cleans_to_201_paragraphs(self):
        name = write_mini_book(self.dir)

        text = (self.dir / name).read_text(encoding="utf-8")
        _, stats = clean_book_text(
            text,
            start_line=MINI_BOOK["start_line"],
            end_line=MINI_BOOK["end_line"],
            book_title=MINI_BOOK["book_title"],
        )
        # PROLOGUE und 200 Sätze
        self.assertEqual(stats.paragraphs, 201)

    def test_cli_creates_directory_and_book(self):
        target = self.dir / "neu"

        with redirect_stdout(io.StringIO()):
            main([str(target)])

        self.assertTrue((target / "mini-book.txt").is_file())

    def test_helpers_use_package_version(self):
        self.assertIs(tests.helpers.write_mini_book, write_mini_book)

    def test_module_call_runs_without_warning(self):
        # So ruft slurm/setup.sh das Mini-Buch auf. Jede Warnung beim Start bricht
        # hier mit einem Fehler ab.
        command = [sys.executable, "-W", "error", "-m", "infm_sota2", "mini-book"]

        result = subprocess.run(
            [*command, str(self.dir / "neu")], capture_output=True, text=True
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")


class BPEDatasetTest(TestCase):
    def test_samples_have_seq_len_tokens_and_shifted_targets(self):
        dataset = BPEDataset("hello world " * 20, seq_len=8)

        inputs, targets = dataset[0]

        self.assertEqual(inputs.shape, (8,))
        self.assertEqual(targets.shape, (8,))
        self.assertTrue(torch.equal(inputs[1:], targets[:-1]))
        self.assertIsInstance(dataset.tokens, torch.Tensor)

    def test_defaults_to_non_overlapping_windows(self):
        dataset = BPEDataset("hello world " * 20, seq_len=8)

        self.assertEqual(dataset.stride, 8)

    def test_rejects_text_shorter_than_sequence_length(self):
        with self.assertRaisesRegex(ValueError, "at least 9 tokens"):
            BPEDataset("hello", seq_len=8)

    def test_rejects_non_positive_stride(self):
        with self.assertRaisesRegex(ValueError, "stride must be positive"):
            BPEDataset("hello world " * 20, seq_len=8, stride=0)

    def test_rejects_non_positive_sequence_length(self):
        with self.assertRaisesRegex(ValueError, "seq_len must be positive"):
            BPEDataset("hello world " * 20, seq_len=0)

    def test_supports_overlapping_windows(self):
        dataset = BPEDataset("hello world " * 20, seq_len=8, stride=2)

        first_inputs, _ = dataset[0]
        second_inputs, _ = dataset[1]

        self.assertTrue(torch.equal(first_inputs[2:], second_inputs[:6]))

    def test_supports_exactly_one_window(self):
        text = "hello world " * 20
        token_count = len(tiktoken.get_encoding("gpt2").encode(text))
        dataset = BPEDataset(text, seq_len=token_count - 1)

        self.assertEqual(len(dataset), 1)
        inputs, targets = dataset[0]
        self.assertEqual(inputs.shape, (token_count - 1,))
        self.assertEqual(targets.shape, (token_count - 1,))

    def test_supports_stride_larger_than_sequence_length(self):
        dataset = BPEDataset("hello world " * 20, seq_len=8, stride=16)

        self.assertEqual(dataset.stride, 16)
        self.assertGreaterEqual(len(dataset), 1)
