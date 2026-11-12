"""Text erzeugen mit Temperature und Top-k-Sampling (Raschka, Kapitel 5.3)."""

import tiktoken
import torch
from torch import Tensor, nn


def generate(
    model: nn.Module,
    idx: Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float,
    top_k: int | None,
    generator: torch.Generator | None,
) -> Tensor:
    """Hängt ``max_new_tokens`` Tokens an ``idx`` (Shape ``(batch, tokens)``) an.

    ``temperature == 0`` wählt immer das wahrscheinlichste Token. Sonst wird aus
    ``softmax(logits / temperature)`` gezogen, bei ``top_k`` nur aus den k besten.
    ``generator`` muss auf demselben Gerät liegen wie ``idx``.
    """
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    if top_k is not None and top_k < 1:
        raise ValueError(f"top_k must be >= 1 or None, got {top_k}")

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(idx[:, -context_size:])[:, -1, :]
            if top_k is not None:
                kth_best = torch.topk(logits, top_k).values[:, -1:]
                logits = logits.masked_fill(logits < kth_best, -torch.inf)
            if temperature == 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1, generator=generator)
            idx = torch.cat([idx, next_id], dim=1)
    model.train(was_training)
    return idx


def complete(
    model: nn.Module,
    tokenizer: tiktoken.Encoding,
    prompt: str,
    max_new_tokens: int,
    context_size: int,
    temperature: float,
    top_k: int | None,
    generator: torch.Generator | None,
) -> str:
    """Gibt nur die Fortsetzung von ``prompt`` zurück, ohne den Prompt selbst."""
    device = next(model.parameters()).device
    prompt_ids = tokenizer.encode(prompt)
    output = generate(
        model,
        torch.tensor([prompt_ids], dtype=torch.long, device=device),
        max_new_tokens=max_new_tokens,
        context_size=context_size,
        temperature=temperature,
        top_k=top_k,
        generator=generator,
    )
    return tokenizer.decode(output[0, len(prompt_ids) :].tolist())
