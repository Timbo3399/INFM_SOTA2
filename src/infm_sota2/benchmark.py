"""Trainingsgeschwindigkeit messen: Tokens/s, Spitzenspeicher und MFU (Woche 5).

Vorlage ist das Benchmark-Skript aus Moodle (Raschka, ``ch04/02_performance-analysis``).
Anders als dort wird erst aufgewärmt und dann über mehrere ganze Trainingsschritte
gemessen, die Attention zählt in den FLOPs mit, und die MFU rechnet gegen die
Spitzenwerte der Tensor Cores.
"""

import gc
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from infm_sota2.loop import batch_loss
from infm_sota2.model import GPTModel, GPTModelV2

Precision = Literal["fp32", "fp16-mixed", "bf16-mixed", "bf16"]
Attention = Literal["auto", "math"]

# Beeinflusst die Geschwindigkeit nicht, deshalb fest
_LEARNING_RATE = 1e-4

# Spitzenleistung in TFLOPS ohne Sparsity, aus den NVIDIA-Datenblättern
# (nvidia.com/en-us/data-center/{tesla-t4,l4,l40s,a100,h100}, H100 PCIe aus dem
# H100-Datenblatt, geprüft am 19.11.2026). Wo das Datenblatt nur den Wert mit
# Sparsity nennt, steht hier die Hälfte. FP32 ohne Tensor Cores.
# Die Reihenfolge zählt: "L40S" vor "L4", "H100 PCIe" und "H100 NVL" vor "H100".
_PEAK_TFLOPS: list[tuple[str, float, float, float | None]] = [
    # Namensteil, FP32, FP16, BF16
    ("L40S", 91.6, 362.05, 362.05),
    ("L4", 30.3, 121.0, 121.0),
    ("A100", 19.5, 312.0, 312.0),
    ("H100 PCIe", 51.0, 756.5, 756.5),
    ("H100 NVL", 60.0, 835.5, 835.5),
    ("H100", 67.0, 989.5, 989.5),  # SXM
    ("T4", 8.1, 65.0, None),  # kann bf16 nicht in Hardware
]


@dataclass(frozen=True)
class Variant:
    """``compile_backend=None`` heißt: nicht kompilieren."""

    name: str
    model_cls: Callable[[dict[str, Any]], nn.Module]
    precision: Precision
    attention: Attention
    compile_backend: str | None


def slide20_variants(mixed: Precision, compile_backend: str) -> list[Variant]:
    """Die sechs Varianten von Folie 20, wie in ``workbook_timo_attention_v2.ipynb``.

    ``mixed`` ist ``bf16-mixed``, auf GPUs ohne bf16 in Hardware ``fp16-mixed``.
    Nr. 3 rechnet immer in bf16 und gehört auf solchen GPUs übersprungen.
    """
    return [
        Variant("1 Ausgangsstand", GPTModel, "fp32", "auto", None),
        Variant("2 16 Bit mit autocast", GPTModel, mixed, "auto", None),
        Variant("3 16 Bit, ganzes Modell", GPTModel, "bf16", "auto", None),
        Variant("4 torch.nn ohne Flash", GPTModelV2, mixed, "math", None),
        Variant("5 torch.nn mit Flash", GPTModelV2, mixed, "auto", None),
        Variant("6 torch.compile", GPTModelV2, mixed, "auto", compile_backend),
    ]


@dataclass(frozen=True)
class BenchmarkResult:
    variant: str
    tokens_per_second: float | None
    ms_per_step: float | None
    peak_memory_gb: float | None
    flops_per_token: int
    mfu: float | None
    error: str | None


def flops_per_token(model: nn.Module, config: dict[str, Any], seq_len: int) -> int:
    """FLOPs für Forward und Backward pro Token (PaLM-Paper, Anhang B).

    ``6 ·`` alle Gewichte der Linear-Schichten (inklusive ``out_head``), dazu die
    beiden Matrixprodukte der Attention über die volle Sequenzlänge.
    """
    linear = sum(m.weight.numel() for m in model.modules() if isinstance(m, nn.Linear))
    return 6 * linear + 12 * config["n_layers"] * seq_len * config["emb_dim"]


def peak_tflops(gpu_name: str | None, precision: Precision) -> float | None:
    """Spitzenleistung der GPU für diese Präzision, ``None`` wenn unbekannt."""
    if gpu_name is None:
        return None
    for part, fp32, fp16, bf16 in _PEAK_TFLOPS:
        if part in gpu_name:
            by_precision = {"fp32": fp32, "fp16-mixed": fp16, "bf16-mixed": bf16}
            return by_precision.get(precision, bf16)  # "bf16" wie "bf16-mixed"
    return None


@contextmanager
def _math_kernel() -> Iterator[None]:
    """Erzwingt den Math-Kernel, mit 16-Bit-Matrixprodukten wie unser eigener Code.

    Ohne die Einstellung rechnet PyTorch bf16- und fp16-Eingaben im Math-Kernel in
    fp32, dann misst Nr. 4 gegen Nr. 5 nicht allein FlashAttention.
    """
    allowed = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
    try:
        with sdpa_kernel(SDPBackend.MATH):
            yield
    finally:
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(allowed)


class Benchmark:
    def __init__(
        self,
        model_config: dict[str, Any],
        batch_size: int,
        seq_len: int,
        warmup_steps: int,
        measure_steps: int,
        device: torch.device,
        seed: int,
    ) -> None:
        """Alle Varianten laufen mit denselben Werten, sonst sind sie nicht
        vergleichbar."""
        if seq_len > model_config["context_length"]:
            raise ValueError(
                f"seq_len {seq_len} is longer than context_length "
                f"{model_config['context_length']}"
            )
        self.model_config = model_config
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.warmup_steps = warmup_steps
        self.measure_steps = measure_steps
        self.device = device
        self.seed = seed

    def run(self, variant: Variant) -> BenchmarkResult:
        """Misst ganze Trainingsschritte mit zufälligen Tokens.

        Reicht der Speicher nicht, steht das im Ergebnis, statt abzubrechen.
        """
        if variant.precision == "fp16-mixed" and self.device.type != "cuda":
            raise ValueError("fp16-mixed needs CUDA (GradScaler)")
        torch.manual_seed(self.seed)
        model = variant.model_cls(self.model_config).to(self.device)
        flops = flops_per_token(model, self.model_config, self.seq_len)
        try:
            ms_per_step, peak_memory_gb = self._measure(model, variant)
        except torch.OutOfMemoryError:
            return BenchmarkResult(
                variant.name, None, None, None, flops, None, "zu wenig Speicher"
            )
        finally:
            del model
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        tokens_per_second = self.batch_size * self.seq_len / (ms_per_step / 1000)
        gpu = (
            torch.cuda.get_device_name(self.device)
            if self.device.type == "cuda"
            else None
        )
        peak = peak_tflops(gpu, variant.precision)
        mfu = None if peak is None else tokens_per_second * flops / (peak * 1e12)
        return BenchmarkResult(
            variant.name,
            tokens_per_second,
            ms_per_step,
            peak_memory_gb,
            flops,
            mfu,
            None,
        )

    def _measure(
        self, model: nn.Module, variant: Variant
    ) -> tuple[float, float | None]:
        """Millisekunden pro Schritt und Spitzenspeicher in GB (nur CUDA)."""
        if variant.precision == "bf16":
            model.to(torch.bfloat16)  # ganzes Modell in bf16, wie im Moodle-Skript
        step_model = model
        if variant.compile_backend is not None:
            # Auf ein Modul angewendet liefert torch.compile wieder ein Modul
            step_model = cast(
                nn.Module, torch.compile(model, backend=variant.compile_backend)
            )
        optimizer = torch.optim.AdamW(model.parameters(), lr=_LEARNING_RATE)
        scaler = torch.amp.GradScaler(
            self.device.type, enabled=variant.precision == "fp16-mixed"
        )
        autocast_dtype = {"fp16-mixed": torch.float16, "bf16-mixed": torch.bfloat16}
        dtype = autocast_dtype.get(variant.precision)
        generator = torch.Generator().manual_seed(self.seed)
        tokens = torch.randint(
            0,
            self.model_config["vocab_size"],
            (self.batch_size, self.seq_len + 1),
            generator=generator,
        ).to(self.device)
        inputs, targets = tokens[:, :-1], tokens[:, 1:]

        def step() -> None:
            optimizer.zero_grad(set_to_none=True)
            kernel = _math_kernel() if variant.attention == "math" else nullcontext()
            mixed = torch.autocast(
                self.device.type, dtype=dtype, enabled=dtype is not None
            )
            with kernel, mixed:
                loss = batch_loss(step_model, inputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Beim Aufwärmen wird auch kompiliert, das soll nicht in die Messung
        for _ in range(self.warmup_steps):
            step()
        if self.device.type != "cuda":
            start_time = time.perf_counter()
            for _ in range(self.measure_steps):
                step()
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return elapsed_ms / self.measure_steps, None

        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(self.measure_steps):
            step()
        end.record()
        torch.cuda.synchronize(self.device)
        peak_memory_gb = torch.cuda.max_memory_allocated(self.device) / 1024**3
        return start.elapsed_time(end) / self.measure_steps, peak_memory_gb
