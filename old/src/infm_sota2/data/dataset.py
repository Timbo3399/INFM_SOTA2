import tiktoken
import torch
from torch.utils.data import Dataset


class BPEDataset(Dataset):
    """Sliding Window Dataset mit BPE Tokenization."""

    def __init__(
        self,
        text: str,
        seq_len: int = 128,
        stride: int | None = None,
        encoding: str = "gpt2",
    ):
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if stride is not None and stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")

        self.tokenizer = tiktoken.get_encoding(encoding)
        self.tokens = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        if len(self.tokens) < seq_len + 1:
            raise ValueError(
                f"Text must contain at least {seq_len + 1} tokens, "
                f"got {len(self.tokens)}"
            )

        self.seq_len = seq_len
        self.stride = seq_len if stride is None else stride
        self.n_samples = (len(self.tokens) - seq_len - 1) // self.stride + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        seq = self.tokens[start : start + self.seq_len + 1]
        return seq[:-1], seq[1:]

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

    @property
    def vocab_size(self):
        return self.tokenizer.n_vocab
