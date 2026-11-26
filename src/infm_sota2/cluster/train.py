"""Smoke-Lauf und Läufe Nr. 2 bis 8: ein Trainingslauf nach einer TOML-Datei.

Liest die Konfiguration, baut Daten und Modell, misst auf Wunsch GPT-2 original,
trainiert mit ``fit`` und schreibt eine JSON-Datei mit allen Einstellungen
und Ergebnissen nach ``results/week06/<benutzer>/<gruppe>/``.

Unter ``torchrun`` mit mehreren Prozessen trainieren alle Ränge mit DDP. Nur
Rang 0 misst danach, erzeugt die Texte und schreibt die Datei.

Aufruf: ``python -m infm_sota2 train --config configs/smoke-cpu.toml``, auf
dem Cluster über ``slurm/train.sbatch``. Die Suche wählt mit ``--grid-index`` eine
Kombination aus Lernrate und Batchgröße.
"""

import argparse
import getpass
import math
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import torch
import wandb
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from infm_sota2.benchmark import flops_per_token, peak_tflops
from infm_sota2.cluster.check import collect_info
from infm_sota2.cluster.config import (
    RunConfig,
    TrainSettings,
    config_dict,
    load_config,
    run_label,
    with_grid,
)
from infm_sota2.cluster.distributed import destroy_distributed, init_distributed
from infm_sota2.cluster.results import git_commit, result_path, write_result
from infm_sota2.data import TextSplits
from infm_sota2.env import WANDB_ENTITY, WANDB_PROJECT, Environment
from infm_sota2.model import (
    GPT2_SMALL_CONFIG,
    GPT2_SMALL_WEIGHTS_SHA256,
    GPT2_SMALL_WEIGHTS_URL,
    GPTModel,
    GPTModelV2,
    build_model,
    download_weights,
    load_pretrained,
)
from infm_sota2.training import ModelReport, TrainResult, load_checkpoint, train

MODEL_CLASSES = {"GPTModel": GPTModel, "GPTModelV2": GPTModelV2}
PRETRAINED = "GPT-2 original"
PRETRAINED_FILE = "gpt2-small-124M.pth"
# Der Klon, aus dem dieser Code läuft (uv sync installiert das Paket editierbar).
# Sein Git-Commit kommt in jede Ergebnisdatei, egal aus welchem Ordner der Lauf kam.
REPO_DIR = Path(__file__).resolve().parents[3]


def run(
    config: RunConfig,
    env: Environment,
    user: str,
    job_id: str | None,
    repo_dir: Path,
) -> dict[str, object]:
    """Ein Lauf von den Daten bis zur JSON-Datei, gibt den Datensatz zurück.

    Auf Rang 0 ist das der geschriebene Datensatz, die anderen Ränge trainieren
    nur mit und geben ``{}`` zurück. Bei ``FloatingPointError`` (Loss NaN) und
    ``torch.OutOfMemoryError`` landet der Lauf mit dem Fehler und ohne Loss in der
    Datei, danach geht der Fehler weiter. So bleibt ein gescheiterter Lauf in der
    Auswertung sichtbar.
    """
    if config.train.precision == "fp16-mixed" and env.device.type != "cuda":
        raise ValueError("fp16-mixed needs CUDA (GradScaler)")
    book = env.data_dir / config.data.file
    if not book.is_file():
        raise FileNotFoundError(
            f"{book} not found. `bash slurm/setup.sh` writes the mini book and lists "
            "what is missing; set SOTA2_DATA_DIR if the data lives in another folder."
        )
    data_fields = asdict(config.data)
    seq_len = data_fields.pop("seq_len")
    data = TextSplits.from_book(env.data_dir, seq_len=seq_len, **data_fields)

    label = run_label(config, env.world_size)
    main_process = env.rank == 0
    context_length = config.model["context_length"]
    assert isinstance(context_length, int)  # load_config hat den Typ geprüft
    report = None
    if main_process and (config.report or config.evaluate_pretrained):
        report = ModelReport(
            data,
            env.device,
            batch_size=config.train.batch_size,
            seed=config.seed,
            **asdict(config.generation),
        )
    record: dict[str, Any] = {}
    if main_process:
        record = _empty_record(config, env, user, label, job_id, repo_dir)
    path = result_path(env.base_dir, user, config.group, label, job_id)
    try:
        if config.evaluate_pretrained and report is not None:
            weights = download_weights(
                GPT2_SMALL_WEIGHTS_URL,
                GPT2_SMALL_WEIGHTS_SHA256,
                env.data_dir / PRETRAINED_FILE,
            )
            pretrained = load_pretrained(
                GPTModel, GPT2_SMALL_CONFIG, weights, env.device
            )
            pretrained_context = GPT2_SMALL_CONFIG["context_length"]  # 1024
            assert isinstance(pretrained_context, int)
            report.add(PRETRAINED, pretrained, pretrained_context)
            del pretrained
            record["pretrained"] = _report_entry(report, PRETRAINED)

        model = build_model(
            MODEL_CLASSES[config.model_class], config.model, config.seed, env.device
        )
        if env.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(env.device)
        start = time.perf_counter()
        result = fit(
            env,
            model,
            data,
            config.train,
            config.seed,
            name=f"{label}-{user}",
            group=config.group,
            checkpoint_name=f"{label}.pth" if config.save_checkpoint else None,
            wandb_config=config_dict(config) | {"user": user, "job_id": job_id},
        )
        if not main_process:
            return {}
        record["fit_seconds"] = time.perf_counter() - start
        record["result"] = asdict(result)
        record["mfu"] = _mfu(model, config, env, result.tokens_per_second)
        if env.device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(env.device) / 1024**3
            record["peak_memory_gb"] = peak
        if config.report:
            assert report is not None
            report.add(label, model, context_length)
            record |= _report_entry(report, label)
    except (FloatingPointError, torch.OutOfMemoryError) as error:
        if main_process:
            record |= {
                "error": f"{type(error).__name__}: {error}",
                "result": None,
                "mfu": None,
                "test_loss": None,
                "perplexity": None,
                "texts": None,
                "copied": None,
                "fit_seconds": None,
            }
            write_result(path, record)
            print(f"Fehler, Ergebnis mit Fehler: {path}")
        raise
    write_result(path, record)
    print(f"Ergebnis: {path}")
    return record


def fit(
    env: Environment,
    model: nn.Module,
    data: TextSplits,
    settings: TrainSettings,
    seed: int,
    name: str,
    group: str,
    checkpoint_name: str | None,
    wandb_config: dict[str, object],
) -> TrainResult:
    """Trainiert ``model`` mit AdamW nach ``settings`` und loggt nach W&B.

    Jeder Prozess nimmt die Fenster seines Rangs aus ``data.train_loader``, ohne
    ``torchrun`` ist das Rang 0 von 1. So sieht jede Zahl der Prozesse dieselben
    Batches. Ab zwei Prozessen trainiert DDP, und nur Rang 0 loggt, gibt aus und
    schreibt den Checkpoint. Tokens und Tokens/s gelten für alle Prozesse.

    Mit ``checkpoint_name`` steht danach der beste Stand im Modell und in
    ``env.checkpoint_dir``. ``wandb_config`` geht in den W&B-Run, damit
    nachvollziehbar ist, womit er lief.
    """
    if settings.batch_size % env.world_size != 0:
        raise ValueError(
            f"batch_size {settings.batch_size} is not divisible by "
            f"world_size {env.world_size}"
        )
    if settings.fused_adamw and env.device.type != "cuda":
        raise ValueError(f"fused AdamW needs CUDA, got device {env.device}")
    local_batch_size = settings.batch_size // env.world_size
    train_loader = data.train_loader(local_batch_size, seed, env.rank, env.world_size)

    # Wie PyTorch es empfiehlt: erst DDP, dann compile. Ein einzelner Prozess
    # läuft ohne DDP, er ist die Basis für den Speedup.
    step_model = model
    if env.world_size > 1:
        device_ids = [env.device.index] if env.device.type == "cuda" else None
        # Die Masken von GPTModel sind konstant, also nichts zu verteilen
        step_model = DistributedDataParallel(
            model, device_ids=device_ids, broadcast_buffers=False
        )
    if settings.compile_backend is not None:
        # Auf ein Modul angewendet liefert torch.compile wieder ein Modul
        step_model = cast(
            nn.Module, torch.compile(step_model, backend=settings.compile_backend)
        )
    # fused=None ist der Default von AdamW
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
        fused=True if settings.fused_adamw else None,
    )

    main_process = env.rank == 0
    wandb_run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=group,
        name=name,
        mode=env.wandb_mode if main_process else "disabled",
        config={
            **wandb_config,
            "world_size": env.world_size,
            "batch_size_per_process": local_batch_size,
        },
    )
    checkpoint_path = (
        env.checkpoint_dir / checkpoint_name
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
        wandb_run.log(scaled, step=step)

    result = train(
        step_model,
        optimizer,
        train_loader,
        data.loader("val", settings.batch_size),
        env.device,
        epochs=settings.epochs,
        eval_every=settings.eval_every,
        grad_clip=settings.grad_clip,
        precision=settings.precision,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata={"name": name, "learning_rate": settings.learning_rate},
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
    wandb_run.summary["best_val_loss"] = result.best_val_loss
    wandb_run.summary["best_step"] = result.best_step
    wandb_run.finish()
    if main_process:
        print(
            f"{name}: bester Validierungs-Loss {result.best_val_loss:.3f} "
            f"bei Schritt {result.best_step} von {result.steps}"
        )
    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model)
        print("Checkpoint:", checkpoint_path)
    return result


def _empty_record(
    config: RunConfig,
    env: Environment,
    user: str,
    label: str,
    job_id: str | None,
    repo_dir: Path,
) -> dict[str, Any]:
    """Der Datensatz vor dem Training: Einstellungen und Umgebung, noch ohne Loss."""
    return {
        "user": user,
        "label": label,
        "group": config.group,
        "config": config_dict(config),
        "environment": collect_info(env, check_internet=False)
        | {
            "job_id": job_id,
            "nodelist": os.environ.get("SLURM_JOB_NODELIST"),
            "world_size": env.world_size,
            "git_commit": git_commit(repo_dir),
        },
        "result": None,
        "mfu": None,
        "test_loss": None,
        "perplexity": None,
        "texts": None,
        "copied": None,
        "pretrained": None,
        "peak_memory_gb": None,
        "fit_seconds": None,
        "error": None,
    }


def _mfu(
    model: nn.Module,
    config: RunConfig,
    env: Environment,
    tokens_per_second: float | None,
) -> float | None:
    """Anteil der Spitzenleistung pro GPU wie in Woche 5, ``None`` ohne bekannte GPU.

    ``tokens_per_second`` gilt für alle Prozesse zusammen, alle auf derselben GPU.
    """
    peak = peak_tflops(env.gpu_name, config.train.precision)
    if peak is None or tokens_per_second is None:
        return None
    flops = flops_per_token(model, config.model, config.data.seq_len)
    return tokens_per_second * flops / (env.world_size * peak * 1e12)


def _report_entry(report: ModelReport, name: str) -> dict[str, object]:
    """Test-Loss, Perplexity, Texte und längste kopierte Folgen eines Modells."""
    test_loss = report.test_loss(name)
    rows = [row for row in report.completions() if row["Modell"] == name]
    return {
        "test_loss": test_loss,
        "perplexity": math.exp(test_loss),
        "texts": [row["Text"] for row in rows],
        "copied": [row["Längste kopierte Folge"] for row in rows],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m infm_sota2 train",
        description="Ein Trainingslauf nach einer TOML-Datei aus configs/",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--grid-index",
        type=int,
        help="Kombination der Suche, im Job-Array SLURM_ARRAY_TASK_ID",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.grid_index is not None:
        config = with_grid(config, args.grid_index)
    env = Environment.detect(smoke_test=False)
    init_distributed(env)
    try:
        run(config, env, getpass.getuser(), os.environ.get("SLURM_JOB_ID"), REPO_DIR)
    finally:
        destroy_distributed(env)


if __name__ == "__main__":
    main()
