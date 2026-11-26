"""Modelle vergleichen: Test-Loss, Perplexity und erzeugte Texte."""

import math

import torch
from torch import nn

from infm_sota2.data.splits import TextSplits
from infm_sota2.training.analysis import longest_copied_span
from infm_sota2.training.generate import complete
from infm_sota2.training.loop import evaluate


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
            model, self.data.loader("test", self.batch_size), self.device, None
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

    def add_texts(self, name: str, texts: list[str]) -> None:
        """Fortsetzungen von außerhalb, z. B. von Claude. Ohne Loss."""
        self._texts[name] = texts

    def test_loss(self, name: str) -> float:
        return self._test_losses[name]

    def losses(self) -> list[dict[str, object]]:
        return [
            {"Modell": name, "Test-Loss": loss, "Perplexity": math.exp(loss)}
            for name, loss in self._test_losses.items()
        ]

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
