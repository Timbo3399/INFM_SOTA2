"""Modelle vergleichen: Test-Loss, Perplexity, erzeugte Texte und wie viel davon
wörtlich aus dem Trainingstext stammt.

``generate`` und ``complete`` erzeugen Text mit Temperature und Top-k-Sampling
(Raschka, Kapitel 5.3). ``longest_copied_span`` findet die längste Wortfolge, die
genau so im Trainingstext steht, ein Zeichen für Auswendiglernen.
"""

import math
import re

import tiktoken
import torch
from torch import Tensor, nn

from infm_sota2.data import TextSplits
from infm_sota2.loop import evaluate


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


# Wörter inklusive Akzenten und innerem Apostroph (don't, Lórien), ohne "_"
_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)*")


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower().replace("’", "'"))


def longest_copied_span(text: str, reference: str) -> str:
    """Längste Wortfolge aus ``text``, die genau so in ``reference`` vorkommt.

    Verglichen wird Wort für Wort, ohne Groß-/Kleinschreibung und Satzzeichen;
    typografische Apostrophe zählen wie ``'``.
    Ergebnis in Kleinbuchstaben, ``""`` wenn kein einziges Wort vorkommt.
    """
    words = _words(text)
    haystack = " " + " ".join(_words(reference)) + " "
    longest: list[str] = []
    for start in range(len(words)):
        end = start + len(longest) + 1
        # Nur Folgen prüfen, die länger wären als der bisherige Fund
        while end <= len(words) and f" {' '.join(words[start:end])} " in haystack:
            longest = words[start:end]
            end += 1
    return " ".join(longest)


class ModelReport:
    def __init__(
        self,
        data: TextSplits,
        device: torch.device,
        batch_size: int,
        seed: int,
        prompt: str,
        samples: int,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
    ) -> None:
        """Alle Modelle werden auf demselben Testtext gemessen und schreiben mit
        denselben Einstellungen und Seeds."""
        self.data = data
        self.device = device
        self.batch_size = batch_size
        self.seed = seed
        self.prompt = prompt
        self.samples = samples
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self._test_losses: dict[str, float] = {}
        self._texts: dict[str, list[str]] = {}

    def add(self, name: str, model: nn.Module, context_size: int) -> None:
        """Misst den Test-Loss, erzeugt ``samples`` Texte und gibt beides aus."""
        test_loss = evaluate(
            model, self.data.loader("test", self.batch_size), self.device
        )
        self._test_losses[name] = test_loss
        self._texts[name] = [
            complete(
                model,
                self.data.tokenizer,
                self.prompt,
                max_new_tokens=self.max_new_tokens,
                context_size=context_size,
                temperature=self.temperature,
                top_k=self.top_k,
                generator=torch.Generator(device=self.device).manual_seed(
                    self.seed + i
                ),
            )
            for i in range(self.samples)
        ]
        perplexity = math.exp(test_loss)
        print(f"{name}: Test-Loss {test_loss:.3f}, Perplexity {perplexity:,.1f}")
        for i, text in enumerate(self._texts[name], 1):
            print(f"  {i}. {self.prompt}{text}")

    def test_loss(self, name: str) -> float:
        return self._test_losses[name]

    def completions(self) -> list[dict[str, object]]:
        """Alle Texte mit der längsten Wortfolge, die wörtlich im Trainingstext
        steht."""
        rows: list[dict[str, object]] = []
        for name, texts in self._texts.items():
            for i, text in enumerate(texts, 1):
                copied = longest_copied_span(text, self.data.train)
                rows.append(
                    {
                        "Modell": name,
                        "Nr.": i,
                        "Text": self.prompt + text,
                        "Kopiert (Wörter)": len(copied.split()),
                        "Längste kopierte Folge": copied,
                    }
                )
        return rows
