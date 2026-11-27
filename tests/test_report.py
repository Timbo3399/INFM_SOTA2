import io
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest import TestCase

import tiktoken
import torch

from infm_sota2.data import TextSplits
from infm_sota2.loop import evaluate
from infm_sota2.report import ModelReport, complete, generate, longest_copied_span
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

        expected = evaluate(self.model, self.data.loader("test", 4), CPU)
        self.assertEqual(report.test_loss("A"), expected)

    def test_same_seed_gives_same_texts(self):
        first, second = self.report(), self.report()

        self.add(first, "A")
        self.add(second, "A")

        self.assertEqual(first.completions(), second.completions())

    def test_completion_row_counts_copied_words(self):
        report = self.report()
        self.add(report, "B")

        rows = report.completions()

        self.assertEqual([row["Modell"] for row in rows], ["B", "B"])
        for row in rows:
            # Kopiert zählt nur die Fortsetzung, nicht den Prompt
            continuation = str(row["Text"]).removeprefix(PROMPT)
            copied = longest_copied_span(continuation, self.data.train)
            self.assertEqual(row["Längste kopierte Folge"], copied)
            self.assertEqual(row["Kopiert (Wörter)"], len(copied.split()))


def seeded(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


class GenerateTest(TestCase):
    def setUp(self):
        self.model = tiny_model(seed=123)
        self.prompt = torch.tensor([[1, 2, 3]])

    def run_generate(self, **kwargs):
        args = {
            "idx": self.prompt,
            "max_new_tokens": 5,
            "context_size": 8,
            "temperature": 0.0,
            "top_k": None,
            "generator": None,
        }
        return generate(self.model, **{**args, **kwargs})

    def test_appends_max_new_tokens(self):
        output = self.run_generate()

        self.assertEqual(output.shape, (1, 8))
        self.assertTrue(torch.equal(output[:, :3], self.prompt))

    def test_greedy_picks_argmax(self):
        output = self.run_generate(max_new_tokens=1)

        with torch.no_grad():
            expected = self.model.eval()(self.prompt)[0, -1].argmax()
        self.assertEqual(output[0, -1], expected)

    def test_top_k_one_equals_greedy(self):
        greedy = self.run_generate()
        sampled = self.run_generate(temperature=1.0, top_k=1, generator=seeded(7))

        self.assertTrue(torch.equal(greedy, sampled))

    def test_same_seed_gives_same_sample(self):
        first = self.run_generate(temperature=1.0, top_k=5, generator=seeded(7))
        second = self.run_generate(temperature=1.0, top_k=5, generator=seeded(7))

        self.assertTrue(torch.equal(first, second))

    def test_crops_prompt_longer_than_context(self):
        long_prompt = torch.arange(10).unsqueeze(0) % 20

        output = self.run_generate(idx=long_prompt, max_new_tokens=1)

        with torch.no_grad():
            expected = self.model.eval()(long_prompt[:, -8:])[0, -1].argmax()
        self.assertEqual(output.shape, (1, 11))
        self.assertEqual(output[0, -1], expected)

    def test_rejects_negative_temperature(self):
        with self.assertRaisesRegex(ValueError, "temperature"):
            self.run_generate(temperature=-0.5)

    def test_rejects_top_k_below_one(self):
        with self.assertRaisesRegex(ValueError, "top_k"):
            self.run_generate(temperature=1.0, top_k=0, generator=seeded(7))

    def test_restores_training_mode(self):
        self.model.train()

        self.run_generate()

        self.assertTrue(self.model.training)


class CompleteTest(TestCase):
    def test_complete_returns_only_continuation(self):
        tokenizer = tiktoken.get_encoding("gpt2")
        model = tiny_model(seed=123, vocab_size=tokenizer.n_vocab, emb_dim=8)
        prompt_ids = tokenizer.encode("The Elves are")

        text = complete(
            model,
            tokenizer,
            "The Elves are",
            max_new_tokens=4,
            context_size=8,
            temperature=0.0,
            top_k=None,
            generator=None,
        )

        full = generate(
            model,
            torch.tensor([prompt_ids]),
            max_new_tokens=4,
            context_size=8,
            temperature=0.0,
            top_k=None,
            generator=None,
        )
        self.assertEqual(text, tokenizer.decode(full[0, len(prompt_ids) :].tolist()))


REFERENCE = (
    "The old road went on and on. Far over the hills the other travellers slept."
)


class LongestCopiedSpanTest(TestCase):
    def test_finds_longest_shared_word_sequence(self):
        span = longest_copied_span(
            "and then the road went on and on forever", REFERENCE
        )

        self.assertEqual(span, "road went on and on")

    def test_respects_word_boundaries(self):
        # "the" steckt in "other", zählt dort aber nicht als Wort
        self.assertEqual(longest_copied_span("he travel", "the other travellers"), "")

    def test_returns_empty_when_nothing_shared(self):
        self.assertEqual(longest_copied_span("completely new words", REFERENCE), "")

    def test_ignores_whitespace_differences(self):
        span = longest_copied_span("over  the\nhills", REFERENCE)

        self.assertEqual(span, "over the hills")

    def test_keeps_accented_words_and_apostrophes_whole(self):
        reference = "the Elves of Lórien don't sleep"

        span = longest_copied_span("Elves of Lórien don’t", reference)

        self.assertEqual(span, "elves of lórien don't")

    def test_ignores_case_and_punctuation(self):
        span = longest_copied_span("On and on, far over", REFERENCE)

        self.assertEqual(span, "on and on far over")
