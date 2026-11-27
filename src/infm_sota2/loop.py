"""Trainingsschleife, Evaluation und Checkpoints für GPTModel (Raschka, Kapitel 5.2).

Alle Hyperparameter kommen vom Aufrufer, es gibt bewusst keine Default-Werte.
"""

import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, DistributedSampler

TrainPrecision = Literal["fp32", "fp16-mixed", "bf16-mixed"]
# fp16 braucht den GradScaler, bf16 hat den Wertebereich von fp32 und kommt ohne aus
_AUTOCAST_DTYPE = {
    "fp32": None,
    "fp16-mixed": torch.float16,
    "bf16-mixed": torch.bfloat16,
}


def batch_loss(model: nn.Module, inputs: Tensor, targets: Tensor) -> Tensor:
    """Mittlere Cross-Entropy über alle Positionen, in float32 berechnet."""
    logits = model(inputs)  # (batch, tokens, vocab)
    return F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Mittlerer Loss pro Token über alle Batches von ``loader``."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for inputs, targets in loader:
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
    # Median der Tokens/s dieses Prozesses über die Intervalle zwischen zwei
    # Prüfungen, ohne das erste (dort kompiliert torch.compile). Ohne die Prüfung
    # selbst. None bei weniger als zwei Intervallen.
    tokens_per_second: float | None
    val_losses: tuple[tuple[int, float], ...]  # (Schritt, Validierungs-Loss)


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    eval_every: int,
    grad_clip: float,
    precision: TrainPrecision,
    checkpoint_path: Path | None,
    checkpoint_metadata: dict[str, object],
    log: Callable[[dict[str, float], int], None],
    eval_model: nn.Module,
) -> TrainResult:
    """Trainiert ``epochs`` Epochen und prüft alle ``eval_every`` Schritte sowie am
    Ende jeder Epoche den Validierungs-Loss.

    ``model`` ist das Modell, mit dem trainiert wird, auch mit DDP oder kompiliert.
    ``eval_model`` ist dasselbe Modell ohne diese Hüllen: Darauf laufen Auswertung
    und Checkpoint. So haben Checkpoints die Schlüssel eines normalen Modells, und
    die Auswertung startet weder Kompilieren noch eine Operation über alle Ränge.

    Bei jedem neuen Bestwert wird nach ``checkpoint_path`` gespeichert (falls nicht
    ``None``). ``log(metrics, step)`` bekommt ``train/loss`` (Mittel seit der letzten
    Prüfung), ``val/loss``, ``epoch``, ``tokens_seen`` und
    ``throughput/tokens_per_second`` (dieses Prozesses, ohne die Prüfung).
    """
    if precision not in get_args(TrainPrecision):
        allowed = ", ".join(get_args(TrainPrecision))
        raise ValueError(f"Unknown precision {precision!r}, allowed: {allowed}")
    if precision == "fp16-mixed" and device.type != "cuda":
        raise ValueError(f"fp16-mixed needs CUDA (GradScaler), got device {device}")
    if eval_every < 1:
        raise ValueError(f"eval_every must be >= 1, got {eval_every}")
    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise ValueError("Train loader yielded no batches")

    autocast_dtype = _AUTOCAST_DTYPE[precision]
    scaler = torch.amp.GradScaler(device.type, enabled=precision == "fp16-mixed")
    best_val_loss = math.inf
    best_step = -1
    step = 0
    tokens_seen = 0
    running_loss = 0.0
    running_steps = 0
    val_losses: list[tuple[int, float]] = []
    interval_rates: list[float] = []
    interval_tokens = 0
    interval_start = time.perf_counter()

    def check_validation(epoch_progress: float) -> None:
        nonlocal best_val_loss, best_step, running_loss, running_steps
        nonlocal interval_tokens, interval_start
        if device.type == "cuda":
            torch.cuda.synchronize(device)  # alle Schritte des Intervalls fertig
        rate = interval_tokens / (time.perf_counter() - interval_start)
        interval_rates.append(rate)
        val_loss = evaluate(eval_model, val_loader, device)
        val_losses.append((step, val_loss))
        log(
            {
                "train/loss": running_loss / running_steps,
                "val/loss": val_loss,
                "epoch": epoch_progress,
                "tokens_seen": tokens_seen,
                "throughput/tokens_per_second": rate,
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
                    eval_model,
                    {**checkpoint_metadata, "step": step, "val_loss": val_loss},
                )
        # Die Prüfung samt Checkpoint zählt nicht zum nächsten Intervall
        interval_tokens = 0
        interval_start = time.perf_counter()

    model.train()
    for epoch in range(epochs):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)  # sonst jede Epoche gleich
        for batch_index, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            # fp32: autocast aus, wie bisher mit use_amp=False
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if autocast_dtype is None else autocast_dtype,
                enabled=autocast_dtype is not None,
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
            interval_tokens += targets.numel()
            running_loss += loss_value
            running_steps += 1
            if step % eval_every == 0:
                check_validation(epoch + (batch_index + 1) / steps_per_epoch)
        if running_steps > 0:
            check_validation(epoch + 1.0)

    measured = interval_rates[1:]  # ohne das erste Intervall mit dem Kompilieren
    return TrainResult(
        best_val_loss,
        best_step,
        step,
        tokens_seen,
        statistics.median(measured) if measured else None,
        tuple(val_losses),
    )


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
