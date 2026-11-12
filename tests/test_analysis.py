from unittest import TestCase

from infm_sota2.training import longest_copied_span

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
