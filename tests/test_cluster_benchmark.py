import io
import json
import os
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest import TestCase, mock

import torch
from torch import nn

from infm_sota2.benchmark import Variant
from infm_sota2.cluster.benchmark import SCAN_BATCH_SIZES, batch_scan, main
from infm_sota2.model import GPTModel, GPTModelV2
from tests.helpers import TINY_CONFIG

CPU = torch.device("cpu")


class OutOfMemoryFromBatch4(nn.Module):
    """Tut so, als reiche der GPU-Speicher ab Batch 4 nicht mehr."""

    def __init__(self, config):
        super().__init__()
        self.inner = GPTModel(config)

    def forward(self, in_idx):
        if in_idx.shape[0] >= 4:
            raise torch.OutOfMemoryError("CUDA out of memory (simuliert)")
        return self.inner(in_idx)


def scan(variant: Variant, batch_sizes: tuple[int, ...]):
    return batch_scan(TINY_CONFIG, 8, batch_sizes, variant, 1, 1, CPU, 1)


class BatchScanTest(TestCase):
    def test_scan_sizes_from_spec(self):
        self.assertEqual(SCAN_BATCH_SIZES, (8, 16, 32, 64, 128, 256, 512))

    def test_batch_scan_stops_after_first_error(self):
        variant = Variant("OOM ab 4", OutOfMemoryFromBatch4, "fp32", "auto", None)

        results = scan(variant, (2, 4, 8))

        self.assertEqual([batch for batch, _ in results], [2, 4])
        self.assertIsNone(results[0][1].error)
        self.assertEqual(results[1][1].error, "zu wenig Speicher")

    def test_batch_scan_compiles_fresh_for_each_batch_size(self):
        # Sonst sammelt torch.compile pro Batchgröße eine Kompilierung an, und nach
        # dem Limit liefen spätere Batchgrößen still ohne compile
        compiled = Variant("compile", GPTModelV2, "fp32", "auto", "eager")
        plain = Variant("ohne compile", GPTModelV2, "fp32", "auto", None)

        with mock.patch("torch.compiler.reset", wraps=torch.compiler.reset) as reset:
            scan(compiled, (2, 4))
            compiled_resets = reset.call_count
            scan(plain, (2, 4))

        self.assertEqual(compiled_resets, 2)
        self.assertEqual(reset.call_count, 2)


class BenchmarkMainTest(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_cwd = Path.cwd()
        os.chdir(self.tmp.name)

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.tmp.cleanup()

    def test_benchmark_smoke_writes_json(self):
        with redirect_stdout(io.StringIO()):
            main(["--smoke"])

        files = list(Path(self.tmp.name).glob("results/week06/*/benchmark/*.json"))
        self.assertEqual(len(files), 1, files)
        record = json.loads(files[0].read_text(encoding="utf-8"))
        self.assertEqual(len(record["variants"]), 6)
        for row in record["variants"]:
            with self.subTest(variant=row["variant"]):
                self.assertIsNone(row["error"])
        self.assertEqual([row["batch_size"] for row in record["batch_scan"]], [2, 4])
        self.assertIn("settings", record)
        self.assertIn("environment", record)
