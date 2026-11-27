"""Trainingsgeschwindigkeit messen: Tokens/s, Spitzenspeicher und MFU (Woche 5).

Vorlage ist das Benchmark-Skript aus Moodle (Raschka, ``ch04/02_performance-analysis``).
Anders als dort wird erst aufgewärmt und dann über mehrere ganze Trainingsschritte
gemessen, die Attention zählt in den FLOPs mit, und die MFU rechnet gegen die
Spitzenwerte der Tensor Cores.

Job 1, Teil 2 (Lauf Nr. 1) misst auf einer GPU, ohne Training:

(a) Die sechs Varianten von Folie 20 aus Woche 5 mit GPT-2 Small, Batch 4 und
1024 Tokens. So steht die GPU des Clusters neben der A100 aus Woche 5.
(b) Variante 6 (``GPTModelV2``, 16 Bit mit ``autocast``, ``torch.compile``) mit
256 Tokens wie im Training, für wachsende Batchgrößen bis zum ersten
Speicherfehler. Das ergibt die Obergrenze für die Suche nach Batchgröße und
Lernrate.

Aufruf: ``python -m infm_sota2 benchmark [--smoke]``, auf dem Cluster
über ``slurm/check.sbatch``.
"""

import argparse
import gc
import getpass
import os
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from infm_sota2.check import collect_info
from infm_sota2.env import Environment, has_bf16
from infm_sota2.loop import batch_loss
from infm_sota2.model import GPT2_SMALL_CONFIG, GPTModel, GPTModelV2
from infm_sota2.results import result_path, write_result

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


SCAN_BATCH_SIZES = (8, 16, 32, 64, 128, 256, 512)

# (a) wie im Notebook von Woche 5
VARIANT_SETTINGS = {
    "batch_size": 4,
    "seq_len": 1024,
    "warmup_steps": 3,
    "measure_steps": 10,
    "seed": 123,
}
# (b) Kontext wie im Training, Schritte wie in (a)
SCAN_SEQ_LEN = 256

# --smoke: das winzige Modell aus dem Smoke-Test von Woche 5
SMOKE_CONFIG = {
    **GPT2_SMALL_CONFIG,
    "context_length": 32,
    "emb_dim": 32,
    "n_heads": 2,
    "n_layers": 2,
}
SMOKE_SETTINGS = {
    "batch_size": 2,
    "seq_len": 32,
    "warmup_steps": 1,
    "measure_steps": 2,
    "seed": 123,
}
SMOKE_BATCH_SIZES = (2, 4)


def batch_scan(
    model_config: dict[str, Any],
    seq_len: int,
    batch_sizes: Sequence[int],
    variant: Variant,
    warmup_steps: int,
    measure_steps: int,
    device: torch.device,
    seed: int,
) -> list[tuple[int, BenchmarkResult]]:
    """Misst ``variant`` für jede Batchgröße und hört nach dem ersten Fehler auf."""
    results = []
    for batch_size in batch_sizes:
        if variant.compile_backend is not None:
            # Frisch kompilieren wie in einem Training mit fester Batchgröße. Sonst
            # sammelt torch.compile Kompilierungen an, und nach seinem Limit liefen
            # spätere Batchgrößen still ohne compile.
            torch.compiler.reset()
        bench = Benchmark(
            model_config, batch_size, seq_len, warmup_steps, measure_steps, device, seed
        )
        result = bench.run(variant)
        results.append((batch_size, result))
        if result.error is not None:
            break
    return results


def format_table(first_column: str, rows: list[tuple[str, BenchmarkResult]]) -> str:
    def number(value: float | None, digits: int) -> str:
        return "-" if value is None else f"{value:,.{digits}f}"

    lines = [
        f"{first_column:<26} {'Tokens/s':>10} {'ms/Schritt':>10} "
        f"{'Speicher (GB)':>13} {'MFU':>6}  Fehler"
    ]
    for label, result in rows:
        lines.append(
            f"{label:<26} {number(result.tokens_per_second, 0):>10} "
            f"{number(result.ms_per_step, 1):>10} "
            f"{number(result.peak_memory_gb, 2):>13} {number(result.mfu, 3):>6}  "
            f"{result.error or ''}"
        )
    return "\n".join(lines)


def _run_variants(
    model_config: dict[str, Any],
    settings: dict[str, int],
    variants: list[Variant],
    device: torch.device,
) -> list[BenchmarkResult]:
    bench = Benchmark(model_config, device=device, **settings)
    results = []
    for variant in variants:
        if variant.precision == "bf16" and not has_bf16(device):
            results.append(
                BenchmarkResult(
                    variant.name,
                    None,
                    None,
                    None,
                    0,
                    None,
                    "übersprungen, die GPU kann bf16 nicht in Hardware",
                )
            )
            continue
        results.append(bench.run(variant))
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m infm_sota2 benchmark",
        description="Lauf Nr. 1: Varianten aus Woche 5 und größte Batchgröße",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="winziges Modell, Batchgrößen 2 und 4, compile mit eager",
    )
    args = parser.parse_args(argv)
    env = Environment.detect(smoke_test=args.smoke)

    if args.smoke:
        variant_config, settings = SMOKE_CONFIG, SMOKE_SETTINGS
        scan_config, scan_seq_len = SMOKE_CONFIG, SMOKE_SETTINGS["seq_len"]
        batch_sizes, compile_backend = SMOKE_BATCH_SIZES, "eager"
    else:
        variant_config, settings = GPT2_SMALL_CONFIG, VARIANT_SETTINGS
        scan_config = {**GPT2_SMALL_CONFIG, "context_length": SCAN_SEQ_LEN}
        scan_seq_len, batch_sizes = SCAN_SEQ_LEN, SCAN_BATCH_SIZES
        compile_backend = "inductor"
    mixed: Precision = "bf16-mixed" if has_bf16(env.device) else "fp16-mixed"
    variants = slide20_variants(mixed, compile_backend)

    print(f"(a) Varianten von Folie 20, {settings}", flush=True)
    variant_results = _run_variants(variant_config, settings, variants, env.device)
    print(format_table("Variante", [(r.variant, r) for r in variant_results]))

    print(f"\n(b) {variants[5].name}, {scan_seq_len} Tokens", flush=True)
    scan_results = batch_scan(
        scan_config,
        scan_seq_len,
        batch_sizes,
        variants[5],
        settings["warmup_steps"],
        settings["measure_steps"],
        env.device,
        settings["seed"],
    )
    print(format_table("Batch", [(str(batch), r) for batch, r in scan_results]))

    job_id = os.environ.get("SLURM_JOB_ID")
    record = {
        "job_id": job_id,
        "environment": collect_info(env, check_internet=False),
        "settings": {
            "variants": {
                "model_config": variant_config,
                **settings,
                "mixed": mixed,
                "compile_backend": compile_backend,
            },
            "batch_scan": {
                "model_config": scan_config,
                "seq_len": scan_seq_len,
                "batch_sizes": list(batch_sizes),
                "variant": variants[5].name,
                "warmup_steps": settings["warmup_steps"],
                "measure_steps": settings["measure_steps"],
                "seed": settings["seed"],
            },
        },
        "variants": [asdict(result) for result in variant_results],
        "batch_scan": [
            {"batch_size": batch, **asdict(result)} for batch, result in scan_results
        ],
    }
    label = (env.gpu_name or "cpu").replace(" ", "-")
    path = result_path(env.base_dir, getpass.getuser(), "benchmark", label, job_id)
    write_result(path, record)
    print(f"\nErgebnis: {path}")
