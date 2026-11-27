"""Job 1, Teil 2 (Lauf Nr. 1): Benchmark auf einer GPU, ohne Training.

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
import getpass
import os
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

import torch

from infm_sota2.benchmark import (
    Benchmark,
    BenchmarkResult,
    Precision,
    Variant,
    slide20_variants,
)
from infm_sota2.cluster.check import collect_info
from infm_sota2.cluster.results import result_path, write_result
from infm_sota2.env import Environment, has_bf16
from infm_sota2.model import GPT2_SMALL_CONFIG

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


if __name__ == "__main__":
    main()
