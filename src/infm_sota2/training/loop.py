"""Trainingsschleife, Evaluation und Checkpoints für GPTModel (Raschka, Kapitel 5.2).

Alle Hyperparameter kommen vom Aufrufer, es gibt bewusst keine Default-Werte.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader


def batch_loss(model: nn.Module, inputs: Tensor, targets: Tensor) -> Tensor:
    """Mittlere Cross-Entropy über alle Positionen, in float32 berechnet."""
    logits = model(inputs)  # (batch, tokens, vocab)
    return F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int | None
) -> float:
    """Mittlerer Loss pro Token über die ersten ``max_batches`` Batches.

    ``max_batches=None`` nimmt alle Batches.
    """
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            loss = batch_loss(model, inputs.to(device), targets.to(device))
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()
    model.train(was_training)
    if total_tokens == 0:
        raise ValueError("Loader yielded no batches")
    return total_loss / total_tokens


@dataclass(frozen=True)
class TrainResult:
    best_val_loss: float
    best_step: int
    steps: int
    tokens_seen: int


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    eval_every: int,
    eval_batches: int | None,
    grad_clip: float,
    use_amp: bool,
    checkpoint_path: Path | None,
    checkpoint_metadata: dict[str, object],
    log: Callable[[dict[str, float], int], None],
) -> TrainResult:
    """Trainiert ``epochs`` Epochen und prüft alle ``eval_every`` Schritte sowie am
    Ende jeder Epoche den Validierungs-Loss.

    Bei jedem neuen Bestwert wird nach ``checkpoint_path`` gespeichert (falls nicht
    ``None``). ``log(metrics, step)`` bekommt ``train/loss`` (Mittel seit der letzten
    Prüfung), ``val/loss``, ``epoch`` und ``tokens_seen``.
    """
    if use_amp and device.type != "cuda":
        raise ValueError(f"Mixed precision needs CUDA, got device {device}")
    if eval_every < 1:
        raise ValueError(f"eval_every must be >= 1, got {eval_every}")
    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise ValueError("Train loader yielded no batches")

    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    best_val_loss = math.inf
    best_step = -1
    step = 0
    tokens_seen = 0
    running_loss = 0.0
    running_steps = 0

    def check_validation(epoch_progress: float) -> None:
        nonlocal best_val_loss, best_step, running_loss, running_steps
        val_loss = evaluate(model, val_loader, device, eval_batches)
        log(
            {
                "train/loss": running_loss / running_steps,
                "val/loss": val_loss,
                "epoch": epoch_progress,
                "tokens_seen": tokens_seen,
            },
            step,
        )
        running_loss = 0.0
        running_steps = 0
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_step = step
            if checkpoint_path is not None:
                save_checkpoint(
                    checkpoint_path,
                    model,
                    {**checkpoint_metadata, "step": step, "val_loss": val_loss},
                )

    model.train()
    for epoch in range(epochs):
        for batch_index, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                loss = batch_loss(model, inputs, targets)
            loss_value = loss.item()  # einziger GPU-Sync pro Schritt
            if not math.isfinite(loss_value):
                raise FloatingPointError(f"Loss is {loss_value} at step {step}")

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            step += 1
            tokens_seen += targets.numel()
            running_loss += loss_value
            running_steps += 1
            if step % eval_every == 0:
                check_validation(epoch + (batch_index + 1) / steps_per_epoch)
        if running_steps > 0:
            check_validation(epoch + 1.0)

    return TrainResult(best_val_loss, best_step, step, tokens_seen)


def save_checkpoint(path: Path, model: nn.Module, metadata: dict[str, object]) -> None:
    """Speichert nur Gewichte und einfache Metadaten, damit ``weights_only`` reicht."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, path)


def load_checkpoint(path: Path, model: nn.Module) -> dict[str, object]:
    """Lädt die Gewichte in ``model`` (strikt) und gibt die Metadaten zurück."""
    device = next(model.parameters()).device
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint["metadata"]
