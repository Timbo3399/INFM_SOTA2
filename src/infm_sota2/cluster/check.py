"""Job 1, Teil 1: prüft die Umgebung, bevor wir trainieren.

Gibt Rechner, GPU, Treiber, CUDA, torch, NCCL, bf16, einen Test des Flash-Kernels,
C-Compiler, Internet, tiktoken-Cache und den Datenordner aus. Unter ``torchrun``
mit mehreren Prozessen meldet sich jeder Rang, und ein ``all_reduce`` über alle
Ränge zeigt, dass die Knoten miteinander rechnen können.

Aufruf: ``python -m infm_sota2.cluster.check [--no-internet]``, auf dem Cluster
über ``slurm/check.sbatch``.
"""

import argparse
import os
import shutil
import socket
import subprocess
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda import nccl
from torch.nn.attention import SDPBackend, sdpa_kernel

from infm_sota2.benchmark import has_bf16
from infm_sota2.cluster.distributed import destroy_distributed, init_distributed
from infm_sota2.env import Environment

# Eingaben im Datenordner, siehe docs/dachs.md
INPUT_FILES = ("mini-book.txt", "The-lord-of-the-rings.txt", "gpt2-small-124M.pth")
# W&B und PyPI: Was ein Job aus dem Internet bräuchte
INTERNET_HOSTS = ("api.wandb.ai", "pypi.org")
INTERNET_TIMEOUT_S = 5


def collect_info(env: Environment, check_internet: bool) -> dict[str, object]:
    """Alles, was Job 1 über Rechner, GPU und Einrichtung wissen will."""
    on_gpu = env.device.type == "cuda"
    return {
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": env.gpu_name,
        "compute_capability": _compute_capability(env.device) if on_gpu else None,
        "driver": _driver_version(),
        "bf16_supported": has_bf16(env.device) if on_gpu else None,
        "flash_attention": _flash_attention(env.device) if on_gpu else None,
        "nccl": _nccl_version() if on_gpu else None,
        "c_compiler": _c_compiler(),
        "internet": _internet() if check_internet else None,
        "tiktoken_cache": _tiktoken_cache(),
        "data_dir": str(env.data_dir),
        "data_files": [name for name in INPUT_FILES if (env.data_dir / name).is_file()],
    }


def format_info(info: dict[str, object]) -> str:
    width = max(map(len, info))
    return "\n".join(
        f"{key:<{width}}  {_format_value(value)}" for key, value in info.items()
    )


def _format_value(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "ja" if value else "nein"
    if isinstance(value, dict):
        return ", ".join(f"{key} {_format_value(item)}" for key, item in value.items())
    if isinstance(value, list):
        return ", ".join(map(str, value)) or "keine"
    return str(value)


def _compute_capability(device: torch.device) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    return f"{major}.{minor}"


def _driver_version() -> str | None:
    """Version des NVIDIA-Treibers laut nvidia-smi, ``None`` ohne nvidia-smi."""
    if shutil.which("nvidia-smi") is None:
        return None
    output = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout
    return output.splitlines()[0].strip()


def _flash_attention(device: torch.device) -> str:
    """Ein kleiner Aufruf, der nur den Flash-Kernel erlaubt: "ok" oder der Fehler."""
    dtype = torch.bfloat16 if has_bf16(device) else torch.float16
    q = torch.randn(1, 2, 16, 64, device=device, dtype=dtype)
    try:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(q, q, q, is_causal=True)
        torch.cuda.synchronize(device)
    except RuntimeError as error:  # kein Flash-Kernel für diese GPU
        return f"Fehler: {error}"
    return "ok" if bool(torch.isfinite(out).all()) else "Fehler: Ergebnis nicht endlich"


def _nccl_version() -> str:
    return ".".join(str(part) for part in nccl.version())


def _c_compiler() -> str | None:
    """Der C-Compiler, den Triton für ``torch.compile`` sucht: CC, gcc, clang."""
    return os.environ.get("CC") or shutil.which("gcc") or shutil.which("clang")


def _internet() -> dict[str, bool]:
    reachable = {}
    for host in INTERNET_HOSTS:
        try:
            with socket.create_connection((host, 443), timeout=INTERNET_TIMEOUT_S):
                reachable[host] = True
        except OSError:  # kein Name, keine Antwort oder abgelehnt
            reachable[host] = False
    return reachable


def _tiktoken_cache() -> str | None:
    """Ordner aus ``TIKTOKEN_CACHE_DIR`` und die Zahl der Dateien darin."""
    directory = os.environ.get("TIKTOKEN_CACHE_DIR")
    if directory is None:
        return None
    path = Path(directory)
    files = sum(1 for item in path.iterdir() if item.is_file()) if path.is_dir() else 0
    return f"{path} ({files} Dateien)"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Umgebung prüfen (Job 1)")
    parser.add_argument(
        "--no-internet", action="store_true", help="Internet nicht prüfen"
    )
    args = parser.parse_args(argv)
    check_internet = not args.no_internet

    env = Environment.detect(smoke_test=False)
    if env.world_size == 1:
        print(format_info(collect_info(env, check_internet)), flush=True)
        return

    # Unter torchrun mit mehreren Prozessen: eine GPU pro Prozess
    init_distributed(env)
    try:
        host, gpu = socket.gethostname(), env.gpu_name or "keine GPU"
        print(f"Rang {env.rank} von {env.world_size}: {host}, {gpu}", flush=True)
        if env.rank == 0:
            print(format_info(collect_info(env, check_internet)), flush=True)
        total = torch.tensor([env.rank + 1], device=env.device)
        dist.all_reduce(total)  # Summe 1 + 2 + ... + world_size
        expected = env.world_size * (env.world_size + 1) // 2
        if env.rank == 0:
            print(
                f"all_reduce über {env.world_size} Prozesse: {int(total.item())} "
                f"(erwartet {expected})",
                flush=True,
            )
        if int(total.item()) != expected:
            raise RuntimeError(
                f"all_reduce over {env.world_size} processes gave "
                f"{int(total.item())}, expected {expected}"
            )
    finally:
        destroy_distributed(env)


if __name__ == "__main__":
    main()
