"""GPT-2-Small-Konfiguration und Download der OpenAI-Gewichte."""

import hashlib
import shutil
import urllib.request
from pathlib import Path

GPT2_SMALL_CONFIG: dict[str, int | float | bool] = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True,
}

# OpenAI-Gewichte im Format von GPTModel (Raschka, "Build a LLM from Scratch").
# Fester Commit statt "main", damit immer dieselbe Datei geladen wird.
GPT2_SMALL_WEIGHTS_URL = (
    "https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/"
    "402728fa68de0d3bb91e3e148ae351511eb7f4db/gpt2-small-124M.pth"
)
GPT2_SMALL_WEIGHTS_SHA256 = (
    "24a9078c5b27137fb2706d2206349c759952a7de8f72ef8e305dc02511bcabf8"
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_weights(url: str, sha256: str, dest: Path) -> Path:
    """Lädt ``url`` nach ``dest`` und prüft die SHA-256.

    Liegt ``dest`` schon da, wird nur geprüft. Eine falsche Prüfsumme bricht ab,
    statt ein anderes Modell zu laden.
    """
    if dest.exists():
        _check_sha256(dest, sha256)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    # Timeout pro Lese-Vorgang: eine hängende Verbindung bricht ab statt ewig zu warten
    with (
        urllib.request.urlopen(url, timeout=60) as response,
        partial.open("wb") as file,
    ):
        shutil.copyfileobj(response, file)
    try:
        _check_sha256(partial, sha256)
    except ValueError:
        partial.unlink()
        raise
    partial.replace(dest)
    return dest


def _check_sha256(path: Path, expected: str) -> None:
    actual = sha256_of(path)
    if actual != expected:
        raise ValueError(
            f"SHA-256 of {path} is {actual}, expected {expected}. "
            "Delete the file and download it again."
        )
