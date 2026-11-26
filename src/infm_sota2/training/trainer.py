"""Ein Trainingslauf mit Optimizer, W&B-Run und bestem Checkpoint.

Unter ``torchrun`` mit mehreren Prozessen trainiert ``fit`` mit DDP. Die
Prozessgruppe startet das Einstiegsskript vorher (``init_distributed``).
"""

import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import torch
import wandb
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from infm_sota2.data.splits import TextSplits
from infm_sota2.env import WANDB_ENTITY, WANDB_PROJECT, Environment
from infm_sota2.training.loop import (
    TrainPrecision,
    TrainResult,
    load_checkpoint,
    train,
)


@dataclass(frozen=True)
class TrainConfig:
    """Werte, die für alle Läufe eines Experiments gleich bleiben."""

    weight_decay: float
    batch_size: int  # über alle Prozesse zusammen
    eval_every: int
    grad_clip: float
    precision: TrainPrecision
    compile_backend: str | None  # None: nicht kompilieren
    fused_adamw: bool  # Optimizer-Schritt in zusammengefassten Kernels, nur CUDA
    seed: int


class Trainer:
    def __init__(self, env: Environment, wandb_config: dict[str, object]) -> None:
        """``wandb_config`` geht in jeden Run, damit nachvollziehbar ist, womit er
        lief."""
        self.env = env
        self.wandb_config = wandb_config

    def fit(
        self,
        model: nn.Module,
        data: TextSplits,
        config: TrainConfig,
        learning_rate: float,
        epochs: int,
        name: str,
        group: str,
        checkpoint_name: str | None,
    ) -> TrainResult:
        """Trainiert ``model`` mit AdamW und loggt nach W&B.

        Mit ``checkpoint_name`` steht danach der beste Stand im Modell und liegt
        zusätzlich im bleibenden Checkpoint-Ordner.

        Unter ``torchrun`` kommen die Fenster aus ``distributed_train_loader``, auch
        bei einem einzigen Prozess, so sieht jede Zahl der Prozesse dieselben
        Batches. Ab zwei Prozessen trainiert DDP, und nur Rang 0 loggt, gibt aus
        und schreibt den Checkpoint. Tokens und Tokens/s gelten für alle Prozesse.
        """
        env = self.env
        if config.batch_size % env.world_size != 0:
            raise ValueError(
                f"batch_size {config.batch_size} is not divisible by "
                f"world_size {env.world_size}"
            )
        if config.fused_adamw and env.device.type != "cuda":
            raise ValueError(f"fused AdamW needs CUDA, got device {env.device}")
        local_batch_size = config.batch_size // env.world_size
        if env.launched_by_torchrun:
            train_loader = data.distributed_train_loader(
                local_batch_size, config.seed, env.rank, env.world_size
            )
        else:
            train_loader = data.train_loader(config.batch_size, config.seed)

        # Wie PyTorch es empfiehlt: erst DDP, dann compile. Ein einzelner Prozess
        # läuft ohne DDP, er ist die Basis für den Speedup.
        step_model = model
        if env.world_size > 1:
            device_ids = [env.device.index] if env.device.type == "cuda" else None
            # Die Masken von GPTModel sind konstant, also nichts zu verteilen
            step_model = DistributedDataParallel(
                model, device_ids=device_ids, broadcast_buffers=False
            )
        if config.compile_backend is not None:
            # Auf ein Modul angewendet liefert torch.compile wieder ein Modul
            step_model = cast(
                nn.Module, torch.compile(step_model, backend=config.compile_backend)
            )
        # fused=None ist der Default von AdamW, so rechnet das Notebook wie bisher
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=config.weight_decay,
            fused=True if config.fused_adamw else None,
        )

        main_process = env.rank == 0
        run = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            group=group,
            name=name,
            mode=env.wandb_mode if main_process else "disabled",
            config={
                **self.wandb_config,
                "run": {"learning_rate": learning_rate, "epochs": epochs},
                "world_size": env.world_size,
                "batch_size_per_process": local_batch_size,
            },
        )
        checkpoint_path = (
            env.run_dir / checkpoint_name
            if checkpoint_name is not None and main_process
            else None
        )

        def log(metrics: dict[str, float], step: int) -> None:
            # Jeder Prozess zählt nur seine eigenen Tokens
            scaled = {
                **metrics,
                "tokens_seen": metrics["tokens_seen"] * env.world_size,
                "throughput/tokens_per_second": metrics["throughput/tokens_per_second"]
                * env.world_size,
            }
            run.log(scaled, step=step)

        result = train(
            step_model,
            optimizer,
            train_loader,
            data.loader("val", config.batch_size),
            env.device,
            epochs=epochs,
            eval_every=config.eval_every,
            grad_clip=config.grad_clip,
            precision=config.precision,
            checkpoint_path=checkpoint_path,
            checkpoint_metadata={"name": name, "learning_rate": learning_rate},
            log=log,
            eval_model=model,
        )
        result = replace(
            result,
            tokens_seen=result.tokens_seen * env.world_size,
            tokens_per_second=(
                None
                if result.tokens_per_second is None
                else result.tokens_per_second * env.world_size
            ),
        )
        run.summary["best_val_loss"] = result.best_val_loss
        run.summary["best_step"] = result.best_step
        run.finish()
        if main_process:
            print(
                f"{name}: bester Validierungs-Loss {result.best_val_loss:.3f} "
                f"bei Schritt {result.best_step} von {result.steps}"
            )
        if checkpoint_path is not None:
            load_checkpoint(checkpoint_path, model)
            print("Checkpoint:", self._keep(checkpoint_path))
        return result

    def _keep(self, path: Path) -> Path:
        """Kopiert von der schnellen Platte an den bleibenden Ort (in Colab: Drive)."""
        target = self.env.checkpoint_dir / path.name
        if target != path:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        return target
