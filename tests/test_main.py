import io
import subprocess
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import TestCase, mock

from infm_sota2.__main__ import COMMANDS, USAGE, main


class MainTest(TestCase):
    def test_command_gets_remaining_arguments(self):
        fake = mock.Mock()

        with mock.patch.dict(COMMANDS, {"train": fake}):
            main(["train", "--config", "x.toml"])

        fake.assert_called_once_with(["--config", "x.toml"])

    def test_every_command_is_listed(self):
        # Diese vier rufen die Skripte in slurm/ auf
        self.assertEqual(set(COMMANDS), {"check", "benchmark", "train", "mini-book"})
        for name in COMMANDS:
            with self.subTest(command=name):
                self.assertIn(name, USAGE)

    def test_unknown_or_missing_command_exits_with_2(self):
        for argv in (["nope"], []):
            with self.subTest(argv=argv):
                stderr = io.StringIO()
                with redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
                    main(argv)
                self.assertEqual(caught.exception.code, 2)
                self.assertIn(USAGE, stderr.getvalue())

    def test_help_prints_usage(self):
        for flag in ("-h", "--help"):
            with self.subTest(flag=flag):
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    main([flag])
                self.assertIn(USAGE, stdout.getvalue())

    def test_mini_book_command_writes_the_book(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            main(["mini-book", tmp])

            self.assertTrue((Path(tmp) / "mini-book.txt").is_file())

    def test_python_m_runs_the_entry_point(self):
        result = subprocess.run(
            [sys.executable, "-m", "infm_sota2"],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn(USAGE, result.stderr)
