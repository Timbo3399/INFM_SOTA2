import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from infm_sota2.check import collect_info, format_info
from tests.helpers import local_env


class CollectInfoTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    @unittest.skipIf(torch.cuda.is_available(), "prüft den Fall ohne GPU")
    def test_collect_info_on_cpu(self):
        info = collect_info(local_env(self.base), check_internet=False)

        self.assertIs(info["cuda_available"], False)
        self.assertIsNone(info["gpu"])
        self.assertIsNone(info["internet"])

    def test_data_files_lists_present_inputs(self):
        (self.base / "data").mkdir()
        (self.base / "data" / "mini-book.txt").write_text("x", encoding="utf-8")

        info = collect_info(local_env(self.base), check_internet=False)

        self.assertEqual(info["data_dir"], str(self.base / "data"))
        self.assertEqual(info["data_files"], ["mini-book.txt"])

    def test_format_info_lists_every_key(self):
        info = collect_info(local_env(self.base), check_internet=False)

        text = format_info(info)

        for key in info:
            with self.subTest(key=key):
                self.assertIn(key, text)


@unittest.skipIf(os.name == "nt", "torchrun mit Gloo nur unter Linux und macOS")
class CheckMainTest(unittest.TestCase):
    def test_two_processes_reduce_over_all_ranks(self):
        # Wie auf dem Cluster mit zwei Knoten, hier zwei CPU-Prozesse mit Gloo
        torchrun = shutil.which("torchrun", path=str(Path(sys.executable).parent))
        assert torchrun is not None, "torchrun fehlt in der Umgebung"
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    torchrun,
                    "--standalone",
                    "--nproc-per-node=2",
                    "-m",
                    "infm_sota2",
                    "check",
                    "--no-internet",
                ],
                cwd=tmp,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=120,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Rang 0 von 2", result.stdout)
        self.assertIn("Rang 1 von 2", result.stdout)
        self.assertIn("all_reduce über 2 Prozesse: 3 (erwartet 3)", result.stdout)
