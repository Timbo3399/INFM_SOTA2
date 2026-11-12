from unittest import TestCase

from infm_sota2.data import clean_book_text, split_text

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
