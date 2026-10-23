import torch
from torch import nn


class Embeddings(nn.Module):
    """Token + Position Embeddings."""

    def __init__(self, vocab_size: int, emb_dim: int, max_len: int = 512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(max_len, emb_dim)
        nn.init.normal_(self.token_emb.weight, 0, 0.02)
        nn.init.normal_(self.pos_emb.weight, 0, 0.02)

    def forward(self, x):
        seq_len = x.shape[1]
        assert seq_len <= self.pos_emb.num_embeddings, (
            f"Sequence length {seq_len} exceeds maximum length {self.pos_emb.num_embeddings}"
        )
        positions = torch.arange(seq_len, device=x.device)
        return self.token_emb(x) + self.pos_emb(positions)
