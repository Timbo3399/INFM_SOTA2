import io
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest import TestCase

import tests.helpers
from infm_sota2.data import clean_book_text
from infm_sota2.data.mini_book import MINI_BOOK, main, write_mini_book


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
        # So ruft slurm/setup.sh das Mini-Buch auf. Importiert das Paket das Modul
        # vorher schon, warnt Python bei jedem Aufruf mit einer RuntimeWarning.
        command = [sys.executable, "-W", "error", "-m", "infm_sota2.data.mini_book"]

        result = subprocess.run(
            [*command, str(self.dir / "neu")], capture_output=True, text=True
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
