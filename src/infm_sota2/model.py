"""GPT-2 in zwei Fassungen, die GPT-2-Konfigurationen und die OpenAI-Gewichte.

``GPTModel`` ist unser eigenes Modell aus Woche 4 (Raschka, Kapitel 4) mit selbst
geschriebenen Bausteinen. ``GPTModelV2`` aus Woche 5 hat dieselbe Architektur aus
den fertigen Bausteinen von PyTorch. Seine Attention nutzt
``F.scaled_dot_product_attention`` und damit ab der A100-Generation
FlashAttention. ``convert_v1_state_dict`` macht aus Gewichten von ``GPTModel``
welche für ``GPTModelV2``.
"""

import hashlib
import shutil
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

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


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    mask: torch.Tensor

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        # Reduce the projection dim to match desired output dim
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        batch, num_tokens, d_in = x.shape
        assert num_tokens <= self.mask.shape[0], (
            f"Number of tokens {num_tokens} exceeds context length {self.mask.shape[0]}"
        )

        keys = self.W_key(x)  # Shape: (batch, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out)
        # -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(batch, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(batch, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (batch, num_tokens, num_heads, head_dim)
        # -> (batch, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (batch, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(batch, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class CausalSelfAttention(nn.Module):
    def __init__(self, emb_dim: int, n_heads: int, drop_rate: float, qkv_bias: bool):
        super().__init__()
        if emb_dim % n_heads != 0:
            raise ValueError(f"emb_dim {emb_dim} is not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.drop_rate = drop_rate
        # Q, K und V für alle Köpfe in einer einzigen Matrixmultiplikation
        self.qkv = nn.Linear(emb_dim, 3 * emb_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, x: Tensor) -> Tensor:
        batch, tokens, emb_dim = x.shape
        # (batch, tokens, 3 · emb) -> (3, batch, heads, tokens, head_dim)
        q, k, v = (
            self.qkv(x)
            .view(batch, tokens, 3, self.n_heads, emb_dim // self.n_heads)
            .permute(2, 0, 3, 1, 4)
        )
        # Maske, Skalierung, Softmax und Dropout in einem Kernel, ohne T×T-Matrix
        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.drop_rate if self.training else 0.0,
            is_causal=True,
        )
        # (batch, heads, tokens, head_dim) -> (batch, tokens, emb)
        return self.out_proj(context.transpose(1, 2).reshape(batch, tokens, emb_dim))


class TransformerBlockV2(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        emb_dim = cfg["emb_dim"]
        self.norm1 = nn.LayerNorm(emb_dim, eps=1e-5)
        self.att = CausalSelfAttention(
            emb_dim, cfg["n_heads"], cfg["drop_rate"], cfg["qkv_bias"]
        )
        self.norm2 = nn.LayerNorm(emb_dim, eps=1e-5)
        # tanh-Näherung der GELU wie im Original-GPT-2
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * emb_dim, emb_dim),
        )
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_shortcut(self.att(self.norm1(x)))
        return x + self.drop_shortcut(self.ff(self.norm2(x)))


class GPTModelV2(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlockV2(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = nn.LayerNorm(cfg["emb_dim"], eps=1e-5)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx: Tensor) -> Tensor:
        positions = torch.arange(in_idx.shape[1], device=in_idx.device)
        x = self.drop_emb(self.tok_emb(in_idx) + self.pos_emb(positions))
        return self.out_head(self.final_norm(self.trf_blocks(x)))


def convert_v1_state_dict(state: dict[str, Tensor]) -> dict[str, Tensor]:
    """Macht aus einem ``state_dict`` von ``GPTModel`` eines für ``GPTModelV2``.

    Q, K und V werden zu ``qkv`` zusammengefügt, ``scale``/``shift`` heißen
    ``weight``/``bias``, und die Masken fallen weg, weil V2 keine braucht.
    """
    converted: dict[str, Tensor] = {}
    for key, value in state.items():
        if key.endswith(".att.mask") or ".att.W_key." in key or ".att.W_value." in key:
            continue
        if ".att.W_query." in key:
            prefix, _, suffix = key.partition(".att.W_query.")
            parts = [
                state[f"{prefix}.att.W_{name}.{suffix}"]
                for name in ("query", "key", "value")
            ]
            converted[f"{prefix}.att.qkv.{suffix}"] = torch.cat(parts, dim=0)
            continue
        key = key.replace(".ff.layers.", ".ff.")
        if key.endswith(".scale"):
            key = key.removesuffix(".scale") + ".weight"
        elif key.endswith(".shift"):
            key = key.removesuffix(".shift") + ".bias"
        converted[key] = value
    return converted


# Die Werte von model_class in den TOML-Dateien
MODEL_CLASSES = {"GPTModel": GPTModel, "GPTModelV2": GPTModelV2}


def build_model[M: nn.Module](
    model_cls: Callable[[dict[str, Any]], M],
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> M:
    """Neues Modell mit Zufallsgewichten aus ``seed``, gleicher Seed heißt gleiche
    Gewichte."""
    torch.manual_seed(seed)
    return model_cls(config).to(device)


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
