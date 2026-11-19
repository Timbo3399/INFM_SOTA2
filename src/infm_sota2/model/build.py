from collections.abc import Callable
from typing import Any

import torch
from torch import nn


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
