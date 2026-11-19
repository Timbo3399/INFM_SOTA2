"""GPT-2 aus den fertigen Bausteinen von PyTorch (Woche 5, Attention V2).

Dieselbe Architektur wie ``GPTModel``, aber mit ``nn.LayerNorm``, ``nn.GELU`` und
``F.scaled_dot_product_attention``. Letztere wählt selbst den schnellsten Kernel,
ab der A100-Generation FlashAttention, und nutzt dabei die Tensor Cores.
"""

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


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
