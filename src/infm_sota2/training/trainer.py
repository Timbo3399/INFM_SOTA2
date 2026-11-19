"""Ein Trainingslauf mit Optimizer, W&B-Run und bestem Checkpoint."""

import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
import wandb
from torch import nn

from infm_sota2.data.splits import TextSplits
from infm_sota2.env import WANDB_ENTITY, WANDB_PROJECT, Environment
from infm_sota2.training.loop import TrainResult, load_checkpoint, train


@dataclass(frozen=True)
class TrainConfig:
    """Werte, die für alle Läufe eines Experiments gleich bleiben."""

    weight_decay: float
    batch_size: int
    eval_every: int
    eval_batches: int | None
    grad_clip: float
    use_amp: bool
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
        """
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=config.weight_decay
        )
        run = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            group=group,
            name=name,
            mode=self.env.wandb_mode,
            config={
                **self.wandb_config,
                "run": {"learning_rate": learning_rate, "epochs": epochs},
            },
        )
        checkpoint_path = (
            None if checkpoint_name is None else self.env.run_dir / checkpoint_name
        )
        result = train(
            model,
            optimizer,
            data.train_loader(config.batch_size, config.seed),
            data.loader("val", config.batch_size),
            self.env.device,
            epochs=epochs,
            eval_every=config.eval_every,
            eval_batches=config.eval_batches,
            grad_clip=config.grad_clip,
            use_amp=config.use_amp,
            checkpoint_path=checkpoint_path,
            checkpoint_metadata={"name": name, "learning_rate": learning_rate},
            log=lambda metrics, step: run.log(metrics, step=step),
        )
        run.summary["best_val_loss"] = result.best_val_loss
        run.summary["best_step"] = result.best_step
        run.finish()
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
