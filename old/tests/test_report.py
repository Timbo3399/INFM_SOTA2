import io
import math
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest import TestCase

import torch

from infm_sota2.data import TextSplits
from infm_sota2.training import ModelReport, evaluate, longest_copied_span
from tests.helpers import MINI_BOOK, tiny_model, write_mini_book

CPU = torch.device("cpu")
PROMPT = "Sentence number"


class ModelReportTest(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        write_mini_book(base)
        self.data = TextSplits.from_book(
            base,
            **MINI_BOOK,
            train_ratio=0.8,
            val_ratio=0.1,
            max_chars=None,
            seq_len=8,
        )
        self.model = tiny_model(seed=1, vocab_size=50257)

    def tearDown(self):
        self.tmp.cleanup()

    def report(self) -> ModelReport:
        return ModelReport(
            self.data,
            CPU,
            batch_size=4,
            seed=1,
            prompt=PROMPT,
            samples=2,
            max_new_tokens=3,
            temperature=0.8,
            top_k=5,
        )

    def add(self, report: ModelReport, name: str) -> None:
        with redirect_stdout(io.StringIO()):
            report.add(name, self.model, context_size=8)

    def test_losses_match_evaluate(self):
        report = self.report()

        self.add(report, "A")

        expected = evaluate(self.model, self.data.loader("test", 4), CPU, None)
        self.assertEqual(
            report.losses(),
            [{"Modell": "A", "Test-Loss": expected, "Perplexity": math.exp(expected)}],
        )
        self.assertEqual(report.test_loss("A"), expected)

    def test_same_seed_gives_same_texts(self):
        first, second = self.report(), self.report()

        self.add(first, "A")
        self.add(second, "A")

        self.assertEqual(first.completions(), second.completions())

    def test_external_texts_appear_only_in_completions(self):
        report = self.report()
        self.add(report, "A")

        report.add_texts("Claude", ["x", "y"])

        names = [row["Modell"] for row in report.completions()]
        self.assertEqual(names, ["A", "A", "Claude", "Claude"])
        self.assertEqual([row["Modell"] for row in report.losses()], ["A"])

    def test_completion_row_counts_copied_words(self):
        report = self.report()
        report.add_texts("B", [" number 5 tells a short story"])

        row = report.completions()[0]

        copied = longest_copied_span(" number 5 tells a short story", self.data.train)
        self.assertEqual(row["Text"], PROMPT + " number 5 tells a short story")
        self.assertEqual(row["Kopiert (Wörter)"], len(copied.split()))
        self.assertEqual(row["Längste kopierte Folge"], copied)
