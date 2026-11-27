import math
from unittest import TestCase

import torch
from torch import nn

from infm_sota2.benchmark import (
    Benchmark,
    Variant,
    flops_per_token,
    peak_tflops,
    slide20_variants,
)
from infm_sota2.model import GPTModel, GPTModelV2
from tests.helpers import TINY_CONFIG, tiny_model

CPU = torch.device("cpu")
TINY_FLOPS = 41_856  # 6 · 6464 Linear-Gewichte + 12 · 2 Schichten · 8 Tokens · 16 Dim.


class OutOfMemoryModel(nn.Module):
    """Tut so, als ginge der GPU der Speicher aus."""

    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, x):
        raise torch.OutOfMemoryError("CUDA out of memory (simuliert)")


class FlopsTest(TestCase):
    def test_flops_per_token_for_tiny_models(self):
        self.assertEqual(flops_per_token(tiny_model(1), TINY_CONFIG, 8), TINY_FLOPS)
        self.assertEqual(
            flops_per_token(GPTModelV2(TINY_CONFIG), TINY_CONFIG, 8), TINY_FLOPS
        )

    def test_peak_tflops_by_gpu_name(self):
        cases = [
            ("NVIDIA L40S", "bf16-mixed", 362.05),
            ("NVIDIA L4", "bf16", 121.0),
            ("NVIDIA H100 PCIe", "fp32", 51.0),
            ("NVIDIA H100 NVL", "bf16", 835.5),
            ("NVIDIA H100 80GB HBM3", "bf16-mixed", 989.5),
            ("Tesla T4", "fp16-mixed", 65.0),
            ("Tesla T4", "bf16-mixed", None),
            (None, "fp32", None),
            ("Unknown", "fp32", None),
        ]
        for gpu, precision, expected in cases:
            with self.subTest(gpu=gpu, precision=precision):
                self.assertEqual(peak_tflops(gpu, precision), expected)


class BenchmarkTest(TestCase):
    def setUp(self):
        self.bench = Benchmark(
            TINY_CONFIG,
            batch_size=2,
            seq_len=8,
            warmup_steps=1,
            measure_steps=2,
            device=CPU,
            seed=1,
        )

    def test_run_on_cpu_reports_speed(self):
        result = self.bench.run(Variant("V1 fp32", GPTModel, "fp32", "auto", None))

        self.assertIsNone(result.error)
        assert result.tokens_per_second is not None
        self.assertTrue(math.isfinite(result.tokens_per_second))
        self.assertGreater(result.tokens_per_second, 0)
        self.assertIsNone(result.peak_memory_gb)
        self.assertIsNone(result.mfu)
        self.assertEqual(result.flops_per_token, TINY_FLOPS)

    def test_bf16_and_math_kernel_run_on_cpu(self):
        variants = [
            Variant("V2 bf16 math", GPTModelV2, "bf16-mixed", "math", None),
            Variant("V1 bf16", GPTModel, "bf16", "auto", None),
        ]
        for variant in variants:
            with self.subTest(variant=variant.name):
                self.assertIsNone(self.bench.run(variant).error)

    def test_compiled_variant_runs(self):
        variant = Variant("V2 compile", GPTModelV2, "fp32", "auto", "eager")

        self.assertIsNone(self.bench.run(variant).error)

    def test_out_of_memory_is_recorded_and_next_run_works(self):
        failed = self.bench.run(Variant("OOM", OutOfMemoryModel, "fp32", "auto", None))
        after = self.bench.run(Variant("V1", GPTModel, "fp32", "auto", None))

        self.assertEqual(failed.error, "zu wenig Speicher")
        self.assertIsNone(failed.tokens_per_second)
        self.assertIsNone(after.error)

    def test_math_kernel_keeps_16_bit_matmuls(self):
        # Sonst rechnet der Math-Kernel bf16-Eingaben in fp32, und Nr. 4 gegen
        # Nr. 5 misst nicht allein FlashAttention
        seen = []

        class FlagSpy(GPTModelV2):
            def forward(self, in_idx):
                seen.append(torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed())
                return super().forward(in_idx)

        before = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()

        self.bench.run(Variant("V2 math", FlagSpy, "bf16-mixed", "math", None))

        self.assertTrue(all(seen))
        self.assertEqual(
            torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(), before
        )

    def test_rejects_seq_len_longer_than_context(self):
        with self.assertRaisesRegex(ValueError, "context_length"):
            Benchmark(TINY_CONFIG, 2, 9, 1, 2, CPU, 1)

    def test_fp16_needs_cuda(self):
        with self.assertRaisesRegex(ValueError, "CUDA"):
            self.bench.run(Variant("V1 fp16", GPTModel, "fp16-mixed", "auto", None))


class Slide20VariantsTest(TestCase):
    def test_slide20_variants_match_week5(self):
        # fp16-mixed statt bf16-mixed, damit sich die übergebene Präzision von
        # der festen bf16 bei Nr. 3 unterscheidet
        variants = slide20_variants("fp16-mixed", "eager")

        self.assertEqual(
            [v.name for v in variants],
            [
                "1 Ausgangsstand",
                "2 16 Bit mit autocast",
                "3 16 Bit, ganzes Modell",
                "4 torch.nn ohne Flash",
                "5 torch.nn mit Flash",
                "6 torch.compile",
            ],
        )
        self.assertEqual(
            [v.model_cls for v in variants], [GPTModel] * 3 + [GPTModelV2] * 3
        )
        self.assertEqual(
            [v.precision for v in variants],
            ["fp32", "fp16-mixed", "bf16", "fp16-mixed", "fp16-mixed", "fp16-mixed"],
        )
        self.assertEqual(
            [v.attention for v in variants],
            ["auto", "auto", "auto", "math", "auto", "auto"],
        )
        self.assertEqual([v.compile_backend for v in variants], [None] * 5 + ["eager"])
