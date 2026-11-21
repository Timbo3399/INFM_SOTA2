"""GPT-2-Konfigurationen, Download und Laden der OpenAI-Gewichte (Small)."""

import hashlib
import shutil
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn

from infm_sota2.model.gpt_v2 import GPTModelV2, convert_v1_state_dict

GPT2_SMALL_CONFIG: dict[str, int | float | bool] = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True,
}
# GPT-2-Paper, Tabelle 2: rund 350 und 760 Mio. Parameter mit geteiltem Embedding,
# in unserem GPTModel mit eigenem out_head rund 405 und 840 Mio.
GPT2_MEDIUM_CONFIG = {
    **GPT2_SMALL_CONFIG,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 24,
}
GPT2_LARGE_CONFIG = {
    **GPT2_SMALL_CONFIG,
    "emb_dim": 1280,
    "n_heads": 20,
    "n_layers": 36,
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


def load_pretrained[M: nn.Module](
    model_cls: Callable[[dict[str, Any]], M],
    config: dict[str, Any],
    weights_path: Path,
    device: torch.device,
) -> M:
    """Baut ein Modell und lädt die Gewichte aus ``weights_path`` (strikt).

    Setzt bewusst keinen Seed: Das Bauen verbraucht dieselben Zufallszahlen wie
    ``model_cls(config)``, ein Training danach bekommt also dieselben wie bisher.
    """
    model = model_cls(config)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(model, GPTModelV2):
        state = convert_v1_state_dict(state)  # die Datei hat das Format von GPTModel
    model.load_state_dict(state)
    return model.to(device)


def _check_sha256(path: Path, expected: str) -> None:
    actual = sha256_of(path)
    if actual != expected:
        raise ValueError(
            f"SHA-256 of {path} is {actual}, expected {expected}. "
            "Delete the file and download it again."
        )
